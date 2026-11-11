"""Consul's dataset: tasks, teacher critiques, ground-truth and judge checks, the final set.

Reuses Emperor's shard writer, local teacher client and loopback guard. Nothing here calls a
hosted API: the teacher and the judge are the local open-weight model behind llama-server.
"""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from emperor.data.generate import Chat
from emperor.data.shards import ShardWriter, done_ids, iter_records, read_jsonl
from emperor.data.teacher import TEACHER_LICENCE, TEACHER_MODEL, TeacherError, TeacherOutputError

from consul import prompts
from consul.overlap import DEFAULT_THRESHOLD, OverlapIndex, overlap_report
from consul.schema import CritiqueParseError, extract_json_object, schema_errors
from consul.score import score_critique
from consul.tasks import CritiqueTask, dumps, generate_tasks

HELDOUT_SEED_OFFSET = 1_000_003  # held-out seeds start far from the training range
MIN_RECALL = 1.0
MAX_FALSE_RATE = 0.0
JUDGE_MEAN = 4.0
JUDGE_MIN = 3
# The teacher's reasoning is stored and becomes the student's thinking: low effort keeps it short.
GENERATE_EFFORT = "low"


# -- tasks ---------------------------------------------------------------------------------


def load_tasks(path: Path) -> list[CritiqueTask]:
    return [CritiqueTask.from_dict(r) for r in read_jsonl(path)] if path.exists() else []


def write_train_tasks(data_dir: Path, count: int, seed: int) -> dict:
    """Append training tasks (training vocabulary) until `prompts.jsonl` holds `count`."""
    path = data_dir / "prompts.jsonl"
    existing = load_tasks(path)
    need = count - len(existing)
    if need > 0:
        tasks = generate_tasks(need, seed + len(existing), seen_ids={t.id for t in existing})
        data_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                handle.write(dumps(task) + "\n")
    return {"tasks": max(count, len(existing)), "added": max(need, 0)}


def write_heldout(
    data_dir: Path,
    count: int,
    seed: int = HELDOUT_SEED_OFFSET,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """The held-out set: another seed range and vocabulary, never overlapping the training set.

    Candidates that duplicate or closely copy a training task are skipped; the overlap report
    (`data/eval/overlap.json`) is written for the record. An existing set is kept as it is.
    """
    out = data_dir / "eval"
    target = out / "heldout.jsonl"
    train = load_tasks(data_dir / "prompts.jsonl")
    if not target.exists():
        index = OverlapIndex(train)
        tasks = generate_tasks(
            count, seed, held=True, avoid=lambda task: index.closest(task)[0] >= threshold
        )
        out.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(dumps(t) + "\n" for t in tasks), encoding="utf-8", newline="\n")
    report = overlap_report(train, load_tasks(target), threshold)
    (out / "overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# -- teacher critiques ---------------------------------------------------------------------


def critique_text(critique: dict) -> str:
    """The stored assistant answer: the critique as indented JSON."""
    return json.dumps(critique, indent=1, ensure_ascii=False)


def make_sample(task: CritiqueTask, critique: dict, reasoning: str) -> dict:
    """A training sample. The assistant turn carries the teacher's reasoning as
    `reasoning_content` beside the critique; Emperor's trainer renders it as the thinking
    model's `<think>` block, so the student learns to reason before it critiques."""
    return {
        "id": task.id,
        "task_class": task.task_class,
        "kind": task.kind,
        "messages": [
            {"role": "system", "content": prompts.CONSUL_SYSTEM},
            {"role": "user", "content": prompts.user_message(task)},
            {
                "role": "assistant",
                "content": critique_text(critique),
                "reasoning_content": reasoning,
            },
        ],
        "meta": {
            "teacher": TEACHER_MODEL,
            "licence": TEACHER_LICENCE,
            "seed": task.seed,
            "host": "local llama.cpp",
            "clean": task.clean,
        },
    }


def teach_one(
    task: CritiqueTask, teacher: Chat, *, temperature: float = 0.4
) -> tuple[dict | None, str | None]:
    """One critique from the teacher with its reasoning, or (None, reason). Checks format,
    citations and that there is reasoning to learn from; the ground truth is applied at the
    judge step."""
    reply = teacher.chat(
        [
            {"role": "system", "content": prompts.teacher_system()},
            {"role": "user", "content": prompts.user_message(task)},
        ],
        temperature=temperature,
        seed=task.seed,
        max_tokens=8192,  # gpt-oss reasons first; 3000 often left no room for the answer
    )
    try:
        critique = extract_json_object(reply.get("content") or "")
    except CritiqueParseError:
        return None, "unparsable"
    if schema_errors(critique):
        return None, "schema_invalid"
    if score_critique(task, critique).bad_citations:
        return None, "invented_citation"
    reasoning = (reply.get("reasoning_content") or "").strip()
    if not reasoning:
        return None, "empty_reasoning"
    return make_sample(task, critique, reasoning), None


def run_generation(
    data_dir: Path,
    teacher: Chat,
    *,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    traces = ShardWriter(data_dir / "traces", "traces")
    rejects = ShardWriter(data_dir / "rejects", "rejects")
    finished = done_ids(data_dir / "traces") | done_ids(data_dir / "rejects")
    todo = [t for t in load_tasks(data_dir / "prompts.jsonl") if t.id not in finished]
    if limit is not None:
        todo = todo[:limit]
    counts = {"todo": len(todo), "accepted": 0, "rejected": 0, "stopped": False}
    for task in todo:
        if should_stop():
            counts["stopped"] = True
            break
        try:
            sample, reason = teach_one(task, teacher)
        except TeacherOutputError:
            sample, reason = None, "teacher_output_unparsable"
        except TeacherError as error:
            counts.update(stopped=True, error=str(error))
            break
        except Exception as error:  # one bad sample must not end an overnight run
            sample, reason = None, f"error:{type(error).__name__}"
        if sample:
            traces.write(sample)
            counts["accepted"] += 1
        else:
            rejects.write({"id": task.id, "task_class": task.task_class, "reason": reason})
            counts["rejected"] += 1
        log(f"{task.id} {'ok' if sample else reason}")
    return counts


# -- judging -------------------------------------------------------------------------------


def parse_judge(text: str) -> dict:
    data = extract_json_object(text)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise CritiqueParseError("missing scores")
    clean = {}
    for name in prompts.JUDGE_CRITERIA:
        value = scores.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not 1 <= value <= 5:
            raise CritiqueParseError(f"bad score for {name}")
        clean[name] = int(value)
    if not isinstance(data.get("pass"), bool):
        raise CritiqueParseError("pass must be true or false")
    return {"scores": clean, "pass": data["pass"], "reason": str(data.get("reason", ""))}


def decide(
    truth: dict,
    judge: dict,
    *,
    threshold: float = JUDGE_MEAN,
    min_recall: float = MIN_RECALL,
    max_false_rate: float = MAX_FALSE_RATE,
) -> tuple[bool, list[str]]:
    """Keep a critique only if it matches the ground truth and the judge approves."""
    reasons = []
    recall, false_rate = truth["recall"], truth["false_rate"]
    if recall is not None and recall < min_recall:
        reasons.append("missed_defect")
    if false_rate is not None and false_rate > max_false_rate:
        reasons.append("false_finding")
    if not judge["pass"]:
        reasons.append("judge_fail")
    scores = judge["scores"]
    if sum(scores.values()) / len(scores) < threshold:
        reasons.append("below_threshold")
    reasons += [f"low_{n}" for n, v in scores.items() if v < JUDGE_MIN]
    return not reasons, reasons


def run_judging(
    data_dir: Path,
    client: Chat,
    *,
    threshold: float = JUDGE_MEAN,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    writer = ShardWriter(data_dir / "judged", "judged")
    finished = done_ids(data_dir / "judged")
    tasks = {t.id: t for t in load_tasks(data_dir / "prompts.jsonl")}
    counts = {"judged": 0, "kept": 0, "stopped": False}
    for sample in iter_records(data_dir / "traces"):
        if sample["id"] in finished or sample["id"] not in tasks:
            continue
        if limit is not None and counts["judged"] >= limit:
            break
        if should_stop():
            counts["stopped"] = True
            break
        task = tasks[sample["id"]]
        answer = sample["messages"][-1]["content"]
        score = score_critique(task, extract_json_object(answer))
        truth = {
            "recall": score.recall,
            "false_rate": score.false_rate,
            "missed": score.missed,
            "false_findings": score.false_findings,
        }
        try:
            reply = client.chat(
                [
                    {"role": "system", "content": prompts.JUDGE_SYSTEM},
                    {"role": "user", "content": prompts.judge_user(task, answer)},
                ],
                temperature=0.0,
                seed=task.seed,
                max_tokens=4096,  # room for gpt-oss reasoning; 1024 often cut the verdict off
            )
            judge = parse_judge(reply.get("content") or "")
            keep, reasons = decide(truth, judge, threshold=threshold)
            verdict = {**judge, "model": TEACHER_MODEL, "keep": keep, "rejection": reasons}
        except (TeacherOutputError, CritiqueParseError) as error:
            verdict = {"keep": False, "rejection": ["judge_unparseable"], "reason": str(error)}
        except TeacherError as error:
            counts.update(stopped=True, error=str(error))
            break
        sample["meta"]["truth"] = truth
        sample["meta"]["judge"] = verdict
        writer.write(sample)
        counts["judged"] += 1
        counts["kept"] += bool(verdict["keep"])
        log(f"{task.id} {'keep' if verdict['keep'] else verdict['rejection']}")
    return counts


# -- final set and report ------------------------------------------------------------------


def filter_kept(data_dir: Path) -> dict:
    out = data_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "final.jsonl"
    seen: set[str] = set()
    kept = dropped = no_reasoning = 0
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in iter_records(data_dir / "judged"):
            judge = sample.get("meta", {}).get("judge", {})
            try:
                valid = not schema_errors(extract_json_object(sample["messages"][-1]["content"]))
            except CritiqueParseError:
                valid = False
            if not judge.get("keep") or sample["id"] in seen or not valid:
                dropped += 1
                continue
            if not (sample["messages"][-1].get("reasoning_content") or "").strip():
                no_reasoning += 1  # from before reasoning was kept; a thinking model needs it
                dropped += 1
                continue
            seen.add(sample["id"])
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    temporary.replace(target)
    return {"kept": kept, "dropped": dropped, "no_reasoning": no_reasoning, "path": str(target)}


def build_report(data_dir: Path) -> dict:
    traces = list(iter_records(data_dir / "traces"))
    rejects = list(iter_records(data_dir / "rejects"))
    judged = list(iter_records(data_dir / "judged"))
    kept_by_class: Counter = Counter()
    judged_by_class: Counter = Counter()
    rejection: Counter = Counter()
    for sample in judged:
        verdict = sample["meta"].get("judge", {})
        judged_by_class[sample["task_class"]] += 1
        if verdict.get("keep"):
            kept_by_class[sample["task_class"]] += 1
        rejection.update(verdict.get("rejection") or [])
    kept = sum(kept_by_class.values())
    return {
        "prompts": len(load_tasks(data_dir / "prompts.jsonl")),
        "generated": len(traces),
        "generation_rejected": len(rejects),
        "generation_rejection_reasons": dict(Counter(r.get("reason", "?") for r in rejects)),
        "judged": len(judged),
        "kept": kept,
        "pass_rate": round(kept / len(judged), 4) if judged else None,
        "kept_by_class": dict(sorted(kept_by_class.items())),
        "judged_by_class": dict(sorted(judged_by_class.items())),
        "judge_rejection_reasons": dict(rejection.most_common()),
        "teacher": TEACHER_MODEL,
        "licence": TEACHER_LICENCE,
    }


def write_report(data_dir: Path) -> dict:
    report = build_report(data_dir)
    (data_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
