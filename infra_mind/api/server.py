from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import InfraMindEnvironment
from env.models import Action, Observation, Reward
from tasks import TASK_REGISTRY


class ResetRequest(BaseModel):
    task_id: str
    scenario_id: int


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


app = FastAPI(title="INFRA-MIND", version="1.0.0")
SESSIONS: Dict[str, InfraMindEnvironment] = {}


def _require_session(session_id: str | None) -> InfraMindEnvironment:
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header.")
    env = SESSIONS.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>INFRA-MIND Console</title>
  <style>
    :root {
      --bg: #09111f;
      --panel: rgba(14, 26, 46, 0.88);
      --panel-strong: #102440;
      --line: rgba(137, 189, 255, 0.16);
      --text: #e7eef8;
      --muted: #91a4be;
      --accent: #5ec0ff;
      --accent-2: #54e0b7;
      --danger: #ff7b7b;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(94, 192, 255, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(84, 224, 183, 0.14), transparent 24%),
        linear-gradient(180deg, #0b1525 0%, #09111f 55%, #060b15 100%);
      min-height: 100vh;
    }
    .shell {
      width: min(1280px, calc(100% - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .hero {
      padding: 28px;
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.06) 42%, transparent 70%);
      pointer-events: none;
    }
    .hero-grid, .panel-grid {
      display: grid;
      gap: 18px;
    }
    .hero-grid { grid-template-columns: 1.6fr 1fr; align-items: end; }
    .panel-grid { grid-template-columns: repeat(3, 1fr); }
    h1, h2, h3, p { margin: 0; }
    h1 {
      font-size: clamp(2rem, 4vw, 3.2rem);
      letter-spacing: -0.04em;
      max-width: 10ch;
      line-height: 0.95;
    }
    .subtitle {
      color: var(--muted);
      margin-top: 12px;
      max-width: 60ch;
      line-height: 1.5;
    }
    .kpis {
      display: grid;
      gap: 12px;
    }
    .kpi {
      padding: 16px;
      border-radius: 16px;
      background: rgba(9, 17, 31, 0.45);
      border: 1px solid rgba(255,255,255,0.06);
    }
    .kpi span {
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .kpi strong { font-size: 1.4rem; }
    .panel { padding: 20px; }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .section-head small { color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    input, select, textarea, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(4, 10, 20, 0.92);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    textarea { min-height: 126px; resize: vertical; }
    button {
      background: linear-gradient(135deg, var(--accent), #2b8cff);
      color: #08111d;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.16s ease, box-shadow 0.16s ease;
      box-shadow: 0 12px 28px rgba(46, 142, 255, 0.28);
    }
    button:hover { transform: translateY(-1px); }
    .secondary {
      background: linear-gradient(135deg, #172b45, #1b375a);
      color: var(--text);
      box-shadow: none;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      background: rgba(5, 10, 18, 0.84);
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.05);
      padding: 16px;
      min-height: 240px;
      overflow: auto;
      color: #dce8f9;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(84, 224, 183, 0.12);
      color: #b8f0de;
      border: 1px solid rgba(84, 224, 183, 0.24);
      font-size: 0.88rem;
    }
    .session {
      color: var(--accent);
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.88rem;
    }
    .pillbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }
    .pill {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.06);
      font-size: 0.82rem;
    }
    @media (max-width: 960px) {
      .hero-grid, .panel-grid { grid-template-columns: 1fr; }
      .shell { width: min(100% - 20px, 1280px); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="status">Live Operator Console</div>
          <h1>INFRA-MIND</h1>
          <p class="subtitle">
            Autonomous cloud healing over a temporal knowledge graph. Reset scenarios, drive the agent step loop,
            and inspect blast radius changes through the API without leaving this console.
          </p>
          <div class="pillbar">
            <span class="pill">GraphRAG visibility</span>
            <span class="pill">SIR cascade replay</span>
            <span class="pill">Counterfactual grading</span>
          </div>
        </div>
        <div class="kpis">
          <div class="kpi"><span>Session</span><strong id="sessionLabel">Not started</strong></div>
          <div class="kpi"><span>Current Task</span><strong id="taskLabel">Idle</strong></div>
          <div class="kpi"><span>Last Reward</span><strong id="rewardLabel">0.0000</strong></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Episode Control</h2>
        <small>Reset creates a fresh isolated session.</small>
      </div>
      <div class="controls">
        <label>Task
          <select id="taskId">
            <option value="easy">easy</option>
            <option value="medium">medium</option>
            <option value="hard">hard</option>
          </select>
        </label>
        <label>Scenario ID
          <input id="scenarioId" type="number" value="0" min="0" max="9" />
        </label>
        <label>&nbsp;
          <button onclick="resetEnv()">Reset Environment</button>
        </label>
        <label>&nbsp;
          <button class="secondary" onclick="loadTasks()">Load Tasks</button>
        </label>
      </div>
      <div class="pillbar">
        <span class="session" id="sessionId">X-Session-ID: none</span>
      </div>
    </section>

    <section class="panel-grid">
      <section class="panel">
        <div class="section-head">
          <h3>Action Payload</h3>
          <small>POST /step</small>
        </div>
        <textarea id="actionPayload">{
  "action_type": "explore_node",
  "target_node_id": "postgres-primary",
  "reasoning": "Expand around the most anomalous database node."
}</textarea>
        <div class="pillbar">
          <button onclick="stepEnv()">Send Action</button>
          <button class="secondary" onclick="submitDiagnosis()">Submit Diagnosis</button>
        </div>
      </section>

      <section class="panel">
        <div class="section-head">
          <h3>State</h3>
          <small>GET /state</small>
        </div>
        <div class="pillbar">
          <button class="secondary" onclick="fetchState()">Refresh State</button>
          <button class="secondary" onclick="validateSpec()">Validate Spec</button>
        </div>
        <pre id="stateOutput">State output will appear here.</pre>
      </section>

      <section class="panel">
        <div class="section-head">
          <h3>Response Log</h3>
          <small>Observation + reward</small>
        </div>
        <pre id="logOutput">Console idle.</pre>
      </section>
    </section>
  </main>
  <script>
    let sessionId = null;

    function setSession(id) {
      sessionId = id;
      document.getElementById("sessionId").textContent = `X-Session-ID: ${id || "none"}`;
      document.getElementById("sessionLabel").textContent = id ? id.slice(0, 8) + "..." : "Not started";
    }

    function pretty(data) {
      return JSON.stringify(data, null, 2);
    }

    async function api(url, options = {}) {
      const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
      if (sessionId) headers["X-Session-ID"] = sessionId;
      const response = await fetch(url, { ...options, headers });
      const text = await response.text();
      let payload;
      try { payload = text ? JSON.parse(text) : {}; } catch { payload = { raw: text }; }
      if (!response.ok) throw new Error(pretty(payload));
      const newSession = response.headers.get("X-Session-ID");
      if (newSession) setSession(newSession);
      return payload;
    }

    async function resetEnv() {
      const task_id = document.getElementById("taskId").value;
      const scenario_id = Number(document.getElementById("scenarioId").value);
      const payload = await api("/reset", { method: "POST", body: JSON.stringify({ task_id, scenario_id }) });
      document.getElementById("taskLabel").textContent = payload.task_name;
      document.getElementById("logOutput").textContent = pretty(payload);
      document.getElementById("rewardLabel").textContent = "0.0000";
    }

    async function stepEnv() {
      const payload = JSON.parse(document.getElementById("actionPayload").value);
      const result = await api("/step", { method: "POST", body: JSON.stringify(payload) });
      document.getElementById("logOutput").textContent = pretty(result);
      document.getElementById("rewardLabel").textContent = result.reward.step_reward.toFixed(4);
      document.getElementById("taskLabel").textContent = result.observation.task_name;
    }

    async function fetchState() {
      const result = await api("/state");
      document.getElementById("stateOutput").textContent = pretty(result);
    }

    async function loadTasks() {
      const result = await api("/tasks");
      document.getElementById("stateOutput").textContent = pretty(result);
    }

    async function validateSpec() {
      const result = await api("/validate_spec", { method: "POST", body: "{}" });
      document.getElementById("stateOutput").textContent = pretty(result);
    }

    function submitDiagnosis() {
      document.getElementById("actionPayload").value = JSON.stringify({
        action_type: "submit_diagnosis",
        diagnosis: {
          root_causes: ["postgres-primary"],
          blast_radius: ["postgres-primary", "order-service", "payment-service"],
          isolated_edges: [["rabbitmq", "event-consumer-b"]],
          trap_triggered_nodes: []
        },
        reasoning: "Submit the current best diagnosis."
      }, null, 2);
    }
  </script>
</body>
</html>
    """


@app.post("/reset", response_model=Observation)
def reset_environment(request: ResetRequest, response: Response) -> Observation:
    if request.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {request.task_id}")
    env = InfraMindEnvironment(task_id=request.task_id, scenario_id=request.scenario_id)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = env
    response.headers["X-Session-ID"] = session_id
    return env.reset(request.task_id, request.scenario_id)


@app.post("/step", response_model=StepResponse)
def step_environment(action: Action, x_session_id: str | None = Header(default=None)) -> StepResponse:
    env = _require_session(x_session_id)
    result = env.step(action)
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def get_state(x_session_id: str | None = Header(default=None)) -> Dict[str, Any]:
    env = _require_session(x_session_id)
    state = env.get_state()
    return {
        "graph_snapshot": state["full_snapshot"].model_dump(mode="json"),
        "visible_state": state["visible_snapshot"].model_dump(mode="json"),
        "task": state["task"],
        "action_budget_remaining": state["action_budget_remaining"],
        "step_number": state["step_number"],
        "time_elapsed_seconds": state["time_elapsed_seconds"],
        "done": state["done"],
        "success": state["success"],
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": task_cls.metadata.task_id,
                "name": task_cls.metadata.task_name,
                "description": task_cls.metadata.task_description,
                "action_budget": task_cls.metadata.action_budget,
                "max_steps": task_cls.metadata.max_steps,
            }
            for task_cls in TASK_REGISTRY.values()
        ]
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/validate_spec")
def validate_spec() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    spec_path = project_root / "openenv.yaml"
    required_files = [
        project_root / "env" / "environment.py",
        project_root / "env" / "models.py",
        project_root / "env" / "graph_engine.py",
        project_root / "env" / "cascade_simulator.py",
        project_root / "env" / "reward_engine.py",
        project_root / "api" / "server.py",
        project_root / "inference.py",
    ]
    issues = []
    if not spec_path.exists():
        issues.append("Missing openenv.yaml.")
    else:
        config = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if config.get("name") != "infra-mind":
            issues.append("openenv.yaml name must be infra-mind.")
        endpoint_keys = set((config.get("endpoints") or {}).keys())
        if endpoint_keys != {"reset", "step", "state", "health"}:
            issues.append("openenv.yaml endpoints block is incomplete.")
    for file_path in required_files:
        if not file_path.exists():
            issues.append(f"Missing required file: {file_path.name}")
    return {"passed": not issues, "issues": issues}
