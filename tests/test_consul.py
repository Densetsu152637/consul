"""Consul: tasks, schema, scoring, dataset steps, eval and pipeline wiring.

No GPU, no model and no network: the teacher is a stub that answers with the reference critique.
"""

import json
from pathlib import Path

import pytest
from emperor.pipeline.idle import IdleGate
from emperor.pipeline.processes import OwnProcesses
from emperor.pipeline.run import Settings
from emperor.train.config import load_config

from consul import data, prompts
from consul.eval.cli import main as eval_main
from consul.overlap import OverlapIndex, overlap_report
from consul.pipeline.cli import ConsulPipeline
from consul.schema import citation_errors, extract_json_object, schema_errors
from consul.score import anchor_hit, reference_critique, score_critique
from consul.tasks import CritiqueTask, generate_tasks, make_task

ROOT = Path(__file__).parent.parent


def test_tasks_are_deterministic_diverse_and_carry_ground_truth():
    assert make_task(7).to_dict() == make_task(7).to_dict()
    tasks = generate_tasks(300, 1)
    assert len({t.id for t in tasks}) == 300
    assert {t.kind for t in tasks} == {"plan", "diff", "report"}
    assert 0.05 < sum(t.clean for t in tasks) / 300 < 0.35
    for task in tasks:
        assert task.artifact and task.urls() >= {s.url for s in task.sources}
        assert all(d.where and d.keywords for d in task.defects)
        assert task.clean == (not [d for d in task.defects if d.acceptance_changing])
    assert len({d.kind for t in tasks for d in t.defects}) >= 15


def test_reference_critique_scores_perfectly_on_every_task():
    for task in generate_tasks(200, 5):
        critique = reference_critique(task)
        assert schema_errors(critique) == []
        score = score_critique(task, critique)
        assert score.format_ok and not score.missed and not score.false_findings


def test_task_roundtrips_through_json():
    task = make_task(11, kind="plan", clean=False)
    again = CritiqueTask.from_dict(json.loads(json.dumps(task.to_dict())))
    assert again == task


def test_schema_rejects_inconsistent_results():
    task = make_task(3, kind="plan", clean=False)
    good = reference_critique(task)
    assert schema_errors(good) == []
    accept = {**good, "verdict": "accept"}
    assert any("accept" in e for e in schema_errors(accept))
    assert any("verdict" in e for e in schema_errors({**good, "findings": [], "verdict": "revise"}))
    assert schema_errors({"verdict": "accept"})  # missing keys
    dup = {**good, "findings": [good["findings"][0], good["findings"][0]]}
    assert any("duplicate" in e for e in schema_errors(dup))
    sourced = json.loads(json.dumps(good))
    sourced["findings"][0].update(basis="source", sources=[])
    assert any("source" in e for e in schema_errors(sourced))


def test_extract_json_object_handles_fences_and_prose():
    assert extract_json_object('text ```json\n{"a": {"b": "}"}}\n``` more') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json_object("no object here")


def test_citations_must_come_from_the_context():
    task = next(t for t in generate_tasks(50, 1) if t.sources)
    critique = reference_critique(task)
    critique["findings"][0]["sources"] = [task.sources[0].url, "https://invented.example/x"]
    assert citation_errors(critique, task.urls()) == [
        "cited URL not in the context: https://invented.example/x"
    ]
    score = score_critique(task, critique)
    assert score.cited == 2 and len(score.bad_citations) == 1


def test_scoring_counts_missed_defects_and_false_blocking_findings():
    task = next(t for t in generate_tasks(100, 1) if len(t.defects) >= 1 and not t.clean)
    empty = {"verdict": "accept", "summary": "fine", "findings": []}
    score = score_critique(task, empty)
    assert score.recall == 0.0 and score.missed
    bogus = {
        "verdict": "revise",
        "summary": "x",
        "findings": [
            {
                "id": "f1",
                "severity": "acceptance_changing",
                "location": "Step 99",
                "impact": "nothing relevant",
                "confidence": "low",
                "basis": "judgement",
                "fix_or_check": "none",
            }
        ],
    }
    score = score_critique(task, bogus)
    assert score.false_rate == 1.0
    assert score_critique(task, None).parsed is False
    clean = make_task(9, kind="plan", clean=True)
    assert score_critique(clean, bogus).false_rate == 1.0
    assert score_critique(clean, empty).false_rate is None


def test_anchor_matching():
    assert anchor_hit("Step 4", "Step 4|Step 3")
    assert anchor_hit("step 3 and 4", "Step 4")
    assert not anchor_hit("Step 5", "Step 4")
    assert anchor_hit("Plan", "Step 4")  # a general location is allowed; keywords decide
    assert anchor_hit("paging.py:L12", "paging.py:L13")
    assert not anchor_hit("paging.py:L40", "paging.py:L13")
    assert anchor_hit("Acceptance criteria", "Acceptance")
    assert anchor_hit("Claim 2", "Claim 2")


def test_heldout_uses_other_vocabulary_and_does_not_overlap_training(tmp_path):
    data.write_train_tasks(tmp_path, 400, 1)
    report = data.write_heldout(tmp_path, 25)
    assert report["ok"] and report["exact_duplicates"] == []
    assert report["max_overlap"] < data.DEFAULT_THRESHOLD
    train = data.load_tasks(tmp_path / "prompts.jsonl")
    held = data.load_tasks(tmp_path / "eval" / "heldout.jsonl")
    assert len(held) == 25 and not {t.id for t in train} & {t.id for t in held}
    assert min(t.seed for t in held) >= data.HELDOUT_SEED_OFFSET > max(t.seed for t in train)
    # A copy of a training task is caught by both checks.
    assert not overlap_report(train, [train[0]], 0.6)["ok"]
    assert OverlapIndex(train).closest(train[3])[0] == 1.0


def test_train_tasks_append_and_resume(tmp_path):
    assert data.write_train_tasks(tmp_path, 30, 1)["added"] == 30
    assert data.write_train_tasks(tmp_path, 50, 1)["added"] == 20
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    assert len(tasks) == 50 and len({t.id for t in tasks}) == 50


class StubTeacher:
    """Answers critiques with the reference critique (good), judge calls with a pass."""

    def __init__(self, tasks, garble=()):
        self.by_text = {prompts.user_message(t): t for t in tasks}
        self.garble = set(garble)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == prompts.JUDGE_SYSTEM:
            scores = {c: 5 for c in prompts.JUDGE_CRITERIA}
            return {"content": json.dumps({"scores": scores, "pass": True, "reason": "ok"})}
        task = self.by_text[user]
        if task.id in self.garble:
            return {"content": "I think it is fine."}
        return {"content": json.dumps(reference_critique(task))}


def test_generate_judge_filter_end_to_end(tmp_path):
    data.write_train_tasks(tmp_path, 40, 1)
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    broken = tasks[0].id
    teacher = StubTeacher(tasks, garble=[broken])
    counts = data.run_generation(tmp_path, teacher)
    assert counts["accepted"] == 39 and counts["rejected"] == 1
    assert data.run_generation(tmp_path, teacher)["todo"] == 0  # resumable
    judged = data.run_judging(tmp_path, teacher)
    assert judged["judged"] == 39 and judged["kept"] == 39
    final = data.filter_kept(tmp_path)
    assert final["kept"] == 39
    report = data.write_report(tmp_path)
    assert report["generation_rejection_reasons"] == {"unparsable": 1}
    sample = json.loads((tmp_path / "final" / "final.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    assert sample["messages"][0]["content"] == prompts.CONSUL_SYSTEM  # no teacher guidance
    assert "keywords" not in json.dumps(sample["messages"])  # ground truth stays out


def test_judging_drops_critiques_that_miss_a_defect(tmp_path):
    data.write_train_tasks(tmp_path, 40, 1)
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    target = next(t for t in tasks if not t.clean)

    class Lazy(StubTeacher):
        def chat(self, messages, **kwargs):
            if messages[0]["content"] != prompts.JUDGE_SYSTEM and (
                self.by_text[messages[1]["content"]].id == target.id
            ):
                return {
                    "content": json.dumps({"verdict": "accept", "summary": "ok", "findings": []})
                }
            return super().chat(messages, **kwargs)

    teacher = Lazy(tasks)
    data.run_generation(tmp_path, teacher)
    data.run_judging(tmp_path, teacher)
    kept = {s["id"] for s in data.iter_records(tmp_path / "judged") if s["meta"]["judge"]["keep"]}
    assert target.id not in kept and len(kept) == 39
    reasons = data.build_report(tmp_path)["judge_rejection_reasons"]
    assert reasons == {"missed_defect": 1}


def test_eval_dry_run_scores_perfectly_and_refuses_overlap(tmp_path, capsys):
    data.write_train_tasks(tmp_path, 60, 1)
    data.write_heldout(tmp_path, 10)
    args = [
        "--tasks", str(tmp_path / "eval" / "heldout.jsonl"),
        "--train-data", str(tmp_path / "prompts.jsonl"),
        "--dry-run", "--output", str(tmp_path / "report.json"),
    ]  # fmt: skip
    assert eval_main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert metrics["defect_recall"]["score"] == 1.0 and metrics["defect_recall"]["passed"]
    assert metrics["false_findings"]["score"] == 0.0 and metrics["false_findings"]["passed"]
    assert metrics["format"]["score"] == 1.0
    assert metrics["loop"]["note"] == "TODO" and report["overlap"]["ok"]
    # Evaluating on the training tasks themselves must be refused.
    copy = ["--tasks", str(tmp_path / "prompts.jsonl")] + args[2:4]
    assert eval_main(copy + ["--dry-run"]) == 2
    assert "overlaps" in capsys.readouterr().err


def test_consul_config_is_emperors_recipe_with_consul_names():
    config = load_config(ROOT / "configs" / "sft.toml")
    assert config.merge.output_dir == "weights/consul-4b-bf16"
    assert config.release.name == "consul-4b" and config.release.eval_alias == "consul"
    assert config.model.base.endswith("Qwen3-4B-Instruct-2507")


def make_pipeline(root: Path, **kwargs) -> ConsulPipeline:
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "sft.toml").write_text(
        (ROOT / "configs" / "sft.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_dir = root / "runs" / "r1"
    run_dir.mkdir(parents=True)
    settings = Settings(root=root, run_dir=run_dir, data_dir=root / "data", prompts=30, **kwargs)
    return ConsulPipeline(settings, IdleGate(asleep=True), OwnProcesses(), eval_tasks=6)


def test_pipeline_prompts_filter_and_eval_command(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.phase_prompts()
    assert result["status"] == "done" and result["tasks"] == 30 and result["heldout"]["ok"]
    assert pipeline._pending("prompts.jsonl", "traces") == 30
    assert pipeline.phase_filter()["kept"] == 0
    command = pipeline.eval_command(Path("eval.json"))
    assert command[1:3] == ["-m", "consul.eval.cli"] and "llamacpp:consul" in command


def test_pipeline_runs_without_gpu_phases(tmp_path):
    pipeline = make_pipeline(
        tmp_path, skip={"generate", "judge", "train", "merge", "export", "eval", "card"}
    )
    summary = pipeline.run()
    assert summary["status"] == "done" and summary["phases"]["prompts"]["status"] == "done"
    assert summary["teacher"] == {"model": "gpt-oss-20b", "licence": "Apache-2.0"}


def test_pipeline_cli_dispatch_and_extra_options():
    from consul.pipeline.cli import split_extra

    known, rest = split_extra(["--detach", "--eval-tasks", "9", "--asleep", "--eval-seed=5"])
    assert (known.eval_tasks, known.eval_seed, rest) == (9, 5, ["--detach", "--asleep"])


def test_prompt_example_is_a_valid_critique():
    from consul.schema import CRITIQUE_EXAMPLE

    assert schema_errors(CRITIQUE_EXAMPLE) == []
    assert '"verdict": "revise"' in prompts.CONSUL_SYSTEM
