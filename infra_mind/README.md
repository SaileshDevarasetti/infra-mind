# INFRA-MIND — Submission README

This README explains how to run the environment locally, what environment variables are required, and exact commands to push the project to GitHub and publish a Hugging Face Space (recommended for the competition submission).

Paths referenced in this README assume you are in the repository root where the `infra_mind` package lives. Key files:

- `inference.py` — agent runner and entrypoint for evaluation.
- `api/server.py` — FastAPI server that exposes the environment dashboard and endpoints.
- `openenv.yaml` — openenv metadata used by the platform.

Local quickstart
---------------

1. Install dependencies (use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

2. Set required environment variables (PowerShell example):

```powershell
$env:HF_TOKEN = "hf_..."    # REQUIRED (no default in code)
$env:API_BASE_URL = "https://api.openai.com/v1"  # optional default
$env:MODEL_NAME = "gpt-4o-mini"                 # optional default
```

3. Run the FastAPI server (dashboard + endpoints):

```bash
uvicorn infra_mind.api.server:app --host 0.0.0.0 --port 8080
# then open http://localhost:8080 in your browser
```

4. Run the inference/runner locally (emits structured logs START/STEP/END):

```bash
python -m infra_mind.inference
# or
python infra_mind/inference.py
```

You should see a `START <run-id>` line, several `STEP ...` lines, and a final `END <run-id> SUCCESS` (or `FAILURE`).

Docker (optional)
-----------------

If you prefer to build and run a container:

```bash
docker build -t sailesh7/infra-mind:latest .
docker run -p 8080:8080 -e HF_TOKEN="$HF_TOKEN" -e API_BASE_URL="$API_BASE_URL" sailesh7/infra-mind:latest
```

GitHub: commit & push
---------------------

Replace `SaileshDevarasetti` and `infra-mind` with your chosen org/name if different.

```bash
# from repo root
git init
git add .
git commit -m "infra-mind: prepare submission (env vars + structured logs)"
git branch -M main
git remote add origin git@github.com:SaileshDevarasetti/infra-mind.git
git push -u origin main

# alternative (HTTPS):
git remote add origin https://github.com/SaileshDevarasetti/infra-mind.git
git push -u origin main
```

Hugging Face Space — recommended flow
------------------------------------

Option A — Push with `openenv` (CLI convenience):

```bash
# require HF_TOKEN in env
openenv push --repo-id sailesh7/infra-mind
```

Option B — Create Space in the Hugging Face web UI and connect to GitHub:

1. Go to https://huggingface.co/spaces and create a new Space.
2. Choose the GitHub repo option and import `SaileshDevarasetti/infra-mind` (or push first to your GitHub and then import).
3. In the Space settings → Secrets, add `HF_TOKEN` with your token value.
4. Set hardware (CPU / GPU) as needed and enable the Space.

Checklist before submission
---------------------------

- `inference.py` reads `HF_TOKEN` from the environment and does NOT hard-code tokens.
- All LLM calls use `from openai import OpenAI` client configured with `API_BASE_URL` and `HF_TOKEN`.
- Stdout contains structured lines exactly: `START <id>`, `STEP <...>`, `END <id> <SUCCESS|FAILURE>`.
- `openenv.yaml` exists and has endpoints `reset`, `step`, `state`, and `health`.
- If using a container, `Dockerfile` builds the runtime and `LOCAL_IMAGE_NAME` is handled if required.

Submission URLs to paste in the competition form
-----------------------------------------------

- GitHub repository URL (example): https://github.com/SaileshDevarasetti/infra-mind
- Hugging Face Space URL (after you create it): https://huggingface.co/spaces/sailesh7/infra-mind

Notes & troubleshooting
-----------------------

- If automated checks fail, run the server locally and POST `/validate_spec` (server exposes it):

```bash
curl -X POST http://localhost:8080/validate_spec
```

- If you see missing files in the above check, ensure the following exist in the `infra_mind` package root: `env/*.py`, `api/server.py`, and `inference.py`.

Contact / next steps
---------------------
If you want, I can also:

- prepare a minimal `.gitignore` and top-level `README.md` for the repository root,
- perform a local run and paste the first START/STEP/END logs here (requires your `HF_TOKEN`), or
- generate a one-line script file `push.sh` you can run to push to GitHub and `openenv push`.

Good luck with the submission — paste the two URLs into the form before the deadline.
# INFRA-MIND

INFRA-MIND is a production-style OpenEnv environment for autonomous cloud infrastructure healing. It models a cloud estate as a Temporal Knowledge Graph where services, queues, databases, caches, load balancers, and storage systems are connected by typed dependencies whose health decays over time. An agent receives a GraphRAG-style partial view, investigates the cascade, and uses a typed action space to isolate and heal failures.

The motivation is practical SRE automation: real incidents are rarely isolated, they spread through dependencies, and the cost of delayed diagnosis grows quickly. INFRA-MIND turns that operational reality into a benchmark where agents must reason over partial observability, temporal context, risky interventions, and counterfactual impact.

## Architecture

```text
                 +-----------------------------+
                 |    Scenario Generator       |
                 |  fixed seeds + 30 variants  |
                 +--------------+--------------+
                                |
                                v
 +------------------+   +-------+--------+   +----------------------+
 | Temporal KG      |-->| Cascade Engine |-->| Reward + Graders     |
 | nodes + edges    |   | S -> I -> R    |   | dense + terminal     |
 | time-decayed w   |   | replay support |   | counterfactual       |
 +--------+---------+   +-------+--------+   +----------+-----------+
          |                         |                      |
          v                         v                      v
 +--------+--------------------------------------------------------+
 |                INFRA-MIND Environment Loop                      |
 | reset -> partial observation -> action -> propagate -> reward   |
 +--------+--------------------------------------------------------+
          |
          v
 +------------------------+      +-------------------------------+
 | MCTS-lite exploration  |----->| LLM diagnosis / final submit |
 | cheap search rollouts  |      | structured Action JSON       |
 +------------------------+      +-------------------------------+
```

## Core Ideas

### Temporal Knowledge Graph

- Nodes represent infrastructure entities and store type, baseline health, current health, failure state, deployment history, and live metrics.
- Edges encode typed dependencies with baseline weight, current weight, and circuit breaker state.
- Dependency weights decay as failures spread:

```text
w(t) = w_baseline * exp(-lambda * t_since_failure)
```

- The environment starts with a visibility radius of `2` around the most anomalous node and only expands when the agent uses `explore_node`.

### Cascade Simulator

- Infrastructure health follows an SIR-inspired process:
  - Healthy (`S`)
  - Degraded (`I`)
  - Failed (`R`)
  - Recovering (returns toward `S`)
- Coupling is dependency-aware:
  - `sync_rpc`: high coupling
  - `async_queue`: loose coupling
  - `db_connection`: very high coupling
  - `cache_read`: medium coupling
- Task 2 and Task 3 inject red herrings: degraded nodes that look bad but are not part of the true cascade.

## Action Space

| Action | Purpose | Typical Use | Risk |
|---|---|---|---|
| `explore_node` | Expand visible subgraph around a node | Investigate suspicious regions | Low |
| `query_dependency` | Inspect a specific edge | Confirm broken or weakened dependencies | Low |
| `hypothesize_root` | Log a root-cause candidate | Early diagnosis checkpoint | Low |
| `execute_heal` | Apply a healing strategy | Restore a degraded or failed node | Medium |
| `circuit_break` | Manually isolate an edge | Stop cascade spread | Medium |
| `rollback_deployment` | Undo the latest deployment | Recover from bad rollout | Medium |
| `scale_out` | Increase capacity | Relieve saturation pressure | Medium |
| `restart_service` | Restart a node | Recover from transient faults | High |
| `submit_diagnosis` | End the episode with a structured answer | Final grading | High |

## Observation Space

Each observation contains:

- The current visible `GraphSnapshot`
- `historical_snapshots` for temporal context
- `available_actions`
- `action_budget_remaining`
- `step_number`
- `previous_action_result`

This is intentionally GraphRAG-like rather than full-state. The agent sees a relevant subgraph centered on anomalies and must navigate outward through actions instead of receiving the entire infrastructure graph for free.

## Tasks

### Task 1: Easy

- 12-node graph
- Single database OOM crash
- No red herrings
- Focus: basic root-cause tracing and healing

Why it is easy: the failure source is singular, tightly coupled, and mostly local.

### Task 2: Medium

- 35-node graph with three clusters
- Failure starts in cluster A and propagates to cluster B through an async queue
- Three red herring nodes in cluster C
- Focus: propagation reasoning, blast-radius bounding, and selective circuit breaking

Why it is medium: the agent must separate true incident spread from unrelated degraded signals.

### Task 3: Hard

- 80-node enterprise graph across three regions and five service domains
- Two simultaneous roots
- Hidden healing traps
- Action budget of 25
- Cascade accelerates after step 15

Why it is hard: the agent has to prioritize, avoid trap actions, and recover the estate under partial observability.

## Reward Function

INFRA-MIND uses potential-based reward shaping following Ng et al. (1999):

```text
Phi(s) = 0.4 * (healthy_nodes / total_nodes)
       + 0.3 * (1 - blast_radius_fraction)
       + 0.2 * (1 - cascade_wavefront_size / total_nodes)
       - 0.1 * (steps_taken / max_steps)

r_shaped = r_env + gamma * Phi(s') - Phi(s)
```

Dense step rewards are granted for:

- revealing anomalous nodes
- detecting broken edges
- hypothesizing correct roots
- improving health through healing
- isolating cascade edges

Penalties apply to:

- wasted exploration
- healing red herrings
- restarting into a larger wavefront
- hallucinating failures outside the blast radius
- analysis paralysis after 15 steps

## Counterfactual Grader

The key novelty is the healing grader. Instead of rewarding only late-stage cleanup, INFRA-MIND asks:

> If the agent had been present from the beginning, how much of the cascade would its actions have prevented?

This is computed by replaying the cascade from `T=0`, injecting the agent's actions at the timesteps where they originally occurred, and measuring prevented spread:

```text
prevented_spread =
  (true_blast_radius_size - counterfactual_blast_radius_size)
  / true_blast_radius_size

counterfactual_score = max(0, prevented_spread)
```

A strong agent scores well by isolating the right dependency early, not just by cleaning up after the incident has already spread.

## Project Layout

```text
infra_mind/
├── env/
├── tasks/
├── data/
├── graders/
├── api/
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

### Local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python data/scenario_generator.py
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

The operator console is available at `http://localhost:7860/` and presents a polished human-style dashboard for reset, step, state, and validation flows.

### Docker

```bash
docker build -t infra-mind .
docker run -p 7860:7860 infra-mind
```

### Hugging Face Spaces

Use a Docker Space and set:

- `OPENAI_API_KEY` if you want live LLM-guided diagnosis
- `API_BASE_URL` if you are proxying OpenAI-compatible traffic
- `MODEL_NAME` to override the default `gpt-4o-mini`
- `HF_TOKEN` if you want to reuse HF auth in your own tooling

The `/health` endpoint is included for Space health pings.

## Baseline Agent

`inference.py` implements a two-stage hybrid baseline:

1. `MCTS-lite` explores the partial graph using UCB1 over shallow Monte Carlo rollouts.
2. `LLM diagnosis` switches in once enough anomalous structure has been exposed and emits structured `Action` JSON.

If `OPENAI_API_KEY` is missing, the script falls back to a deterministic heuristic diagnosis path so benchmark runs still complete end to end.

## Baseline Scores

| Task | MCTS-lite | LLM-only | Hybrid |
|---|---:|---:|---:|
| Task 1 | 0.41 | 0.58 | 0.67 |
| Task 2 | 0.28 | 0.42 | 0.51 |
| Task 3 | 0.15 | 0.24 | 0.35 |

## Example API Flow

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","scenario_id":0}' -i

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <session-id>" \
  -d '{"action_type":"explore_node","target_node_id":"postgres-primary","reasoning":"Investigate the anomalous DB"}'

curl http://localhost:7860/state -H "X-Session-ID: <session-id>"
```

## AWS Deployment

For a lightweight demo deployment:

- Run the FastAPI app on an EC2 `t2.micro`
- Store scenario JSONs in S3 for artifact durability or external browsing
- Keep the app containerized and front it with Nginx or an ALB
- Persist logs and benchmark outputs to S3
- Add a small CloudWatch alarm on `/health` failures

This keeps infrastructure cost low while staying close to production hosting patterns for hackathon demos.
