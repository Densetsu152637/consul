# Consul

Training, evaluation, and export for **Consul**, an open-weight critic model for
[Imperium](https://github.com/Densetsu152637/Imperium-AI). Consul keeps
[Emperor](https://github.com/Densetsu152637/emperor), Imperium's orchestrator model, in check:
given a plan, a diff or an agent report, it returns ranked findings (location, impact,
confidence, fix or check) and a verdict in a strict JSON result schema.

The base is the dense **Qwen3-4B-Thinking-2507**, trained with a **bf16 LoRA** (no QLoRA, no
4/8-bit) and quantised only after training, so it runs on the CPU from system RAM, on the GPU,
or split between them. Consul thinks before it critiques: each training sample keeps the
teacher's reasoning as `reasoning_content` on the assistant turn (Emperor's trainer renders it
as the `<think>` block), and the eval removes the thinking before it scores the answer.

> **Status:** the data, training and evaluation pipeline is implemented and tested without a
> GPU. Nothing has been trained or evaluated on a model yet. The plan is
> [`docs/plans/consul.md`](docs/plans/consul.md).

## Layout

| Path | Holds |
| ---- | ----- |
| `src/consul/schema.py`, `prompts.py` | The critique result schema and Consul's system prompt |
| `src/consul/tasks.py` | Seeded-defect plans, diffs and reports with recorded ground truth |
| `src/consul/data.py`, `data_cli.py` | Teacher critiques, schema, citation, ground-truth and judge checks, final set |
| `src/consul/score.py`, `overlap.py` | Scoring a critique against ground truth; held-out vs training overlap |
| `src/consul/eval/` | The held-out harness: C1 (recall, false findings), C3 (format), C2 (citations) |
| `src/consul/pipeline/` | The idle-gated pipeline, a subclass of Emperor's |
| `configs/sft.toml` | The training recipe and release text (output `weights/consul-4b-bf16/`) |
| `data/`, `runs/`, `outputs/`, `weights/`, `tools/` | Generated locally and ignored by Git |

Training, merge, GGUF export, model cards and the guarded publish are
[Emperor's code](https://github.com/Densetsu152637/emperor) (a git dependency pinned in
`pyproject.toml`), driven by this repository's config.

## Development

```bash
uv sync
bash scripts/check.sh   # what CI runs
uv sync --group data --group train   # data generation and training (multi-GB torch)
```

Emperor installs scripts with the same names (`pipeline`, `eval`, `data`). They hand over to
Consul's through `[tool.emperor.entrypoints]` in `pyproject.toml`, so run the commands from the
repository root.

## Build the model

Everything runs on this machine. No Claude, Anthropic or OpenAI API generates or judges data;
the teacher is a local open-weight model (gpt-oss-20b, Apache-2.0) behind `llama-server`.

```powershell
# Start the whole build in the background (writes runs/<timestamp>/ and returns at once).
uv run --group data --group train pipeline --detach
# Same, but the user is asleep: ignore the idle gate.
uv run --group data --group train pipeline --detach --asleep
# Watch it, stop it cleanly, or continue a run.
Get-Content runs/<timestamp>/pipeline.log -Wait
New-Item runs/<timestamp>/STOP
uv run --group data --group train pipeline --detach --resume runs/<timestamp>
# Export GGUF files and write cards for a run that finished training.
uv run --group data --group train pipeline --detach --resume runs/<timestamp> --only export --only card
```

Phases (`--skip` and `--only` take these names): `prompts` (training tasks and a held-out set
checked against them for overlap), `generate` (teacher critiques with their reasoning; a reply
without reasoning is rejected as `empty_reasoning`), `judge` (ground-truth check plus local
judge, which sees only the critique), `filter`, `train` (bf16 LoRA, resumable, fits into free
VRAM and system RAM), `merge`, `export` (bf16 and Q8_0, Q6_K, Q4_K_M GGUF), `eval` (the
held-out harness against a `llama-server` started with `--fit on`), `card`. The idle gate, STOP file, pauses and options
are those of Emperor's pipeline (see its README); `--prompts N` sets the training tasks,
`--eval-tasks N` and `--eval-seed S` the held-out set, and `--generate-reasoning-effort`
(default `low`, for short reasoning traces) the teacher's effort while it writes critiques; the
judge keeps `--reasoning-effort`.

Steps can also be run alone: `uv run data {tasks,heldout,generate,judge,filter,report}`,
`uv run --group train train {show,fit,merge}`, `uv run release {export,card,publish}`.

## Evaluation

```bash
uv run data tasks --count 2000 && uv run data heldout --count 60
uv run eval --dry-run                      # reference critiques, no model: must score 1.0
uv run eval --model llamacpp:consul --base-url http://127.0.0.1:8080/v1 --output runs/eval.json
```

| Metric | Goal | Scored as |
| ------ | ---- | --------- |
| `defect_recall` | C1 | Share of seeded acceptance-changing defects found (target 0.80) |
| `false_findings` | C1 | Share of acceptance-changing findings matching no seeded defect (at most 0.15) |
| `format` | C3 | Share of answers that parse and validate against the schema (0.98) |
| `citations` | C2 | Share of cited URLs that appear in the provided context (0.90); the full precedent-and-link judge is future work |
| `loop` | C4 | TODO: delta-review convergence |

The model's `<think>...</think>` block (or a dangling one, if it was cut off) is removed before
the answer is parsed, and `--max-tokens` (default 16384) leaves room for the thinking.

A finding identifies a seeded defect when its location points at the defect and its text uses
one of the defect's keywords; critiques that word a defect very differently are missed, so
recall is a lower bound. The eval refuses to run if the held-out tasks overlap the training
tasks.

## Licence

[Apache-2.0](LICENSE)
