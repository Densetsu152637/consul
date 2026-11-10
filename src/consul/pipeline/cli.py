"""`uv run pipeline`: the whole Consul build, gated on idleness, on Emperor's orchestrator.

Same command line, idle gate, STOP file, partial-GPU fit, resume and phases as Emperor's:
prompts -> generate -> judge -> filter -> train -> merge -> export -> eval -> card.
Consul replaces what is specific to a critic: the tasks (seeded defects) and a held-out set in
`prompts`, the teacher critiques in `generate`, the ground-truth and judge checks in `judge`,
the final set in `filter`, and the held-out harness in `eval`. Training, merge, export and
cards are Emperor's code, driven by this repository's `configs/sft.toml`.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from emperor.data.shards import done_ids, iter_records
from emperor.data.teacher import ChatClient
from emperor.pipeline import cli as base
from emperor.pipeline.run import Pipeline

from consul import data

MODULE = "consul.pipeline.cli"
log = logging.getLogger("consul")


class ConsulPipeline(Pipeline):
    def __init__(
        self,
        settings,
        gate,
        own,
        eval_tasks: int = 60,
        eval_seed: int = data.HELDOUT_SEED_OFFSET,
    ):
        super().__init__(settings, gate, own)
        self.eval_tasks = eval_tasks
        self.eval_seed = eval_seed

    def phase_prompts(self) -> dict:
        """Training tasks, then the held-out set checked against them."""
        result = data.write_train_tasks(self.s.data_dir, self.s.prompts, self.s.seed)
        overlap = data.write_heldout(self.s.data_dir, self.eval_tasks, self.eval_seed)
        status = "done" if overlap["ok"] else "failed"
        return {"status": status, **result, "heldout": overlap}

    def _teacher_work(self, kind: str):
        def work(server, should_stop):
            client = ChatClient(
                f"http://127.0.0.1:{self.s.teacher_port}/v1",
                reasoning_effort=self.s.reasoning_effort,
            )
            try:
                if kind == "generate":
                    counts = data.run_generation(
                        self.s.data_dir, client, should_stop=should_stop, log=log.debug
                    )
                else:
                    counts = data.run_judging(
                        self.s.data_dir,
                        client,
                        threshold=self.s.threshold,
                        should_stop=should_stop,
                        log=log.debug,
                    )
            finally:
                client.close()
            return {**counts, "stopped": counts.get("stopped", False)}

        return work

    def _pending(self, source: str, done: str) -> int:
        finished = done_ids(self.s.data_dir / done)
        if source == "prompts.jsonl":
            finished |= done_ids(self.s.data_dir / "rejects")
            tasks = data.load_tasks(self.s.data_dir / source)
            return sum(1 for t in tasks if t.id not in finished)
        return sum(1 for r in iter_records(self.s.data_dir / source) if r["id"] not in finished)

    def phase_filter(self) -> dict:
        result = data.filter_kept(self.s.data_dir)
        report = data.write_report(self.s.data_dir)
        return {"status": "done", **result, "pass_rate": report["pass_rate"]}

    def eval_command(self, report: Path) -> list[str]:
        return [
            sys.executable, "-m", "consul.eval.cli",
            "--model", f"llamacpp:{self.sft.release.eval_alias}",
            "--base-url", f"http://127.0.0.1:{self.s.eval_port}/v1",
            "--tasks", str(self.s.data_dir / "eval" / "heldout.jsonl"),
            "--train-data", str(self.s.data_dir / "prompts.jsonl"),
            "--context-tier", self.s.eval_tier,
            "--output", str(report),
        ]  # fmt: skip


def split_extra(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Consul's own options (held-out size and seed), and the line without them."""
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--eval-tasks", type=int, default=60, help="held-out tasks to generate")
    extra.add_argument("--eval-seed", type=int, default=data.HELDOUT_SEED_OFFSET)
    known, rest = extra.parse_known_args(argv)
    return known, rest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known, rest = split_extra(argv)
    parser = base.build_parser("pipeline", __doc__.strip().splitlines()[0])
    args, _ = parser.parse_known_args(rest)
    if args.detach:  # relaunch with Consul's own options kept, as this module
        root = args.root.resolve()
        run_dir = (
            args.resume.resolve()
            if args.resume
            else root / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        return base.detach(argv, run_dir, MODULE)

    def factory(settings, gate, own):
        return ConsulPipeline(settings, gate, own, known.eval_tasks, known.eval_seed)

    return base.main(
        rest,
        pipeline_class=factory,
        module=MODULE,
        prog="pipeline",
        description=__doc__.strip().splitlines()[0],
    )


if __name__ == "__main__":
    sys.exit(main())
