# The Regression & Telemetry Gate 

An automated eval and tracing harness built on top of Week 4's
Persistent Operator: every agent request is traced end-to-end
(Langfuse/OpenTelemetry), unit economics (cost/tokens/latency) are
tracked per step, and a curated golden dataset is replayed through the
real agent graph on every pull request — scored with DeepEval,
RAGAS, and a custom trajectory metric — to **block a merge the moment
a change measurably degrades quality**.

This is Week 5 of a 10-week self-study AI Engineering program. See
`week-5-notes.md` (provided separately) for the concept notes this
project implements.

## What this project demonstrates

| Concept | Where it lives |
|---|---|
| End-to-end tracing (Langfuse, OTel-native SDK) | `telemetry/tracing.py`, wired into `app/graph.py`'s nodes |
| Unit economics (cost/token/latency per step) | `telemetry/usage_aggregator.py`, exposed via `GET /telemetry` |
| Golden dataset | `evaluation/golden_dataset.py` |
| LLM-as-a-Judge (GEval) | `evaluation/geval_metrics.py` |
| RAG faithfulness (RAGAS) | `evaluation/ragas_faithfulness_metric.py` |
| Agent tool-call trajectory evaluation | `evaluation/trajectory_metric.py` |
| Evaluation runner (replays goldens through the real graph) | `evaluation/eval_runner.py` |
| Regression testing & CI/CD gate | `tests/test_regression.py`, `.github/workflows/regression.yml` |
| Everything from Week 4 (agent loop, tools, Redis short/long-term memory) | `app/` — unchanged except `graph.py`/`main.py`'s new, optional tracer wiring |

## Project structure

```
regression-telemetry-gate/
├── app/                              # Week 4's Persistent Operator (unchanged agent logic)
│   ├── main.py                        # UPDATED: builds/wires AgentTracer, adds GET /telemetry
│   ├── graph.py                       # UPDATED: nodes optionally wrapped in tracing spans
│   ├── config.py, schemas.py, state.py, tools.py, safety.py, llm.py
│   └── memory/customer_memory_store.py
├── telemetry/                        # NEW
│   ├── tracing.py                     # Langfuse/OTel-native span & generation helpers
│   └── usage_aggregator.py            # Local cost/latency/token recording + aggregation
├── evaluation/                       # NEW
│   ├── golden_dataset.py              # Curated golden examples
│   ├── trajectory_metric.py           # Deterministic tool-call-sequence metric
│   ├── ragas_faithfulness_metric.py   # RAGAS Faithfulness wrapped as a DeepEval metric
│   ├── geval_metrics.py               # GEval correctness + tone metrics
│   └── eval_runner.py                 # Replays a golden example through the real graph
├── tests/
│   ├── test_regression.py             # THE CI/CD GATE — run via `deepeval test run`
│   ├── test_trajectory_metric.py      # NEW: unit tests for trajectory comparison logic
│   ├── test_usage_aggregator.py       # NEW: unit tests for cost/latency aggregation
│   └── (Week 4's test_tools.py, test_safety.py, test_graph_routing.py, test_customer_memory_store.py — unchanged)
├── .github/workflows/regression.yml  # NEW: the actual CI gate
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Setup

1. **Get an OpenRouter API key**: https://openrouter.ai/keys
2. **(Optional) Get a Langfuse project**: https://cloud.langfuse.com — if you skip this, tracing degrades gracefully to local-only recording (see `telemetry/tracing.py`'s `AgentTracer` docstring); the eval suite and `/telemetry` endpoint work either way.
3. **Create your environment file:**
   ```bash
   cp .env.example .env
   # edit .env: paste OPENROUTER_API_KEY, and LANGFUSE_* if you have a project
   ```
4. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. **Run Redis Stack** (only needed for the LIVE service, not the eval suite):
   ```bash
   docker run -d -p 6379:6379 redis/redis-stack:latest
   ```
6. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

### Chat with tracing active
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "conv-1", "customer_id": "cust-42", "message": "Whats the status of order ORD-1001?"}'
```
If Langfuse is configured, open your Langfuse dashboard to see the full trace: `load_context` span → `call_model` generation (with token usage) → `execute_tools` span → `call_model` generation again.

### Check cumulative unit economics
```bash
curl http://localhost:8000/telemetry
```
Returns running totals (cost, tokens, latency) across every request this process has served — see `telemetry/usage_aggregator.py`.

### Run the regression suite locally
```bash
deepeval test run tests/test_regression.py
# or
pytest tests/test_regression.py -v
```
This replays every golden example (`evaluation/golden_dataset.py`) through the real agent graph and scores it — exactly what CI runs on every PR.

### Watch the gate actually block a regression
Deliberately break something — e.g. edit `app/graph.py`'s `call_model` system prompt to remove the instruction to look up an order before refunding it — then re-run the regression suite. The `refund_requires_lookup_first` golden example's `ToolTrajectoryMetric` should now fail, since the agent may call `issue_refund` without `get_order_status` first (or in the wrong order), demonstrating the gate catching exactly the kind of regression it exists to catch.

### Health check
```bash
curl http://localhost:8000/health
```

## Running unit tests (fast, no API key needed)

```bash
pytest tests/ --ignore=tests/test_regression.py
```
Covers the trajectory metric's comparison logic, usage aggregation math, and everything inherited from Week 4 (tools, safety valves, graph routing, customer memory store) — all pure logic, no LLM calls.

## CI/CD

`.github/workflows/regression.yml` runs two jobs on every push/PR:
1. `unit-tests` — the fast suite above, no secrets required.
2. `regression-eval` — `deepeval test run tests/test_regression.py`, requiring an `OPENROUTER_API_KEY` repository secret. This is the actual quality gate: with branch protection requiring this check, a golden-dataset regression blocks the merge automatically.

## Definition of Done (Week 5 milestone)

- [x] A single `/chat` request produces one complete, correctly-nested trace (verified via Langfuse if configured, or via `/telemetry`'s aggregated totals otherwise) showing every node and tool call with cost/token/latency attached
- [x] The golden dataset covers a simple lookup, a dependent multi-tool-call action, and a cross-thread memory recall — one example per major Week 4 capability
- [x] The CI workflow fails when a deliberately-introduced regression is present (see "Watch the gate actually block a regression" above) and passes on the unmodified baseline
- [x] At least one trajectory-based metric (deterministic) and one RAGAS metric (LLM-as-a-Judge) are both wired into the suite, not just GEval correctness alone

## Known limitations (intentional — addressed in later weeks)

- No online/production evaluation — this harness is entirely offline/golden-dataset-based; continuously scoring live traffic is a natural extension once this offline foundation is solid.
- Judge-score noise calibration (how many repeated judge calls, threshold tolerance) is not deeply tuned — start with these defaults and revisit as false-positive/negative CI failures are observed in practice, per week-5-notes.md section 11.
- No prompt-injection-aware evaluation — Week 8 (AI Reliability & Safety) territory.
- `langchain-openrouter`, `langfuse`, `deepeval`, and `ragas` are all fast-moving packages — if any API here has shifted, each module's docstring names the specific API surface used and where to check current docs.
