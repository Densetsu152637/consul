# Consul

Training, evaluation, and export for **Consul**, an open-weight critic model for
[Imperium](https://github.com/Densetsu152637/Imperium-AI). Consul keeps
[Emperor](https://github.com/Densetsu152637/emperor), Imperium's orchestrator model, in check:
it challenges Emperor's plans and results with evidence, using tools such as web search to
test them against precedent, and it plugs into the Agent-Custom-Workflow critic loop.

Consul runs on the CPU from system RAM (a ~18 GB mixture-of-experts model fits comfortably on
a 64 GB machine), so it critiques while the GPU keeps serving Emperor.

> **Status:** planning only. Training is future work; the plan is
> [`docs/plans/consul.md`](docs/plans/consul.md).

## Licence

[Apache-2.0](LICENSE)
