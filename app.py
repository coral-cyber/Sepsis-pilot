"""
SepsisPilot — FastAPI HTTP Server
Exposes the OpenEnv API over HTTP so inference.py (and any agent) can interact
with the environment via standard REST calls.

Endpoints:
  POST /reset              → start new episode
  POST /step               → take one action
  GET  /state              → current state
  GET  /grade              → grade completed episode
  GET  /tasks              → list available tasks
  GET  /health             → liveness check
  GET  /                   → visual dashboard (HTML)
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from environment import SepsisPilotEnv, AVAILABLE_TASKS
from environment.models import (
    ActionRequest, GraderResult, PatientState, ResetRequest,
    StepResult, TaskInfo,
)


# ── Session state (single-session server; suitable for hackathon eval) ──
_env: Optional[SepsisPilotEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = SepsisPilotEnv()
    yield


app = FastAPI(
    title="SepsisPilot OpenEnv",
    description=(
        "Reinforcement learning environment for optimal sepsis treatment sequencing. "
        "Trains AI agents to learn antibiotic + vasopressor policies in simulated ICU patients."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# OpenEnv API
# ──────────────────────────────────────────────

@app.post("/reset", response_model=PatientState, tags=["OpenEnv"])
async def reset(body: ResetRequest):
    """Reset the environment and begin a new episode."""
    if body.task not in AVAILABLE_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{body.task}'. Available: {AVAILABLE_TASKS}",
        )
    state = _env.reset(task=body.task, seed=body.seed)
    return state


@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
async def step(body: ActionRequest):
    """Apply an action. Returns next state, reward, done flag, and info."""
    try:
        result = _env.step(body.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.get("/state", response_model=PatientState, tags=["OpenEnv"])
async def state():
    """Return the current environment state without advancing the simulation."""
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade", response_model=GraderResult, tags=["OpenEnv"])
async def grade():
    """Grade a completed episode. Returns a score in [0.0, 1.0]."""
    try:
        return _env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", response_model=list[TaskInfo], tags=["OpenEnv"])
async def tasks():
    """List all available tasks with descriptions."""
    return SepsisPilotEnv.task_list()


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "env": "SepsisPilot", "version": "1.0.0"}


# ──────────────────────────────────────────────
# Visual Dashboard
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SepsisPilot — ICU AI Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e17;
    --surface: #111827;
    --border: #1f2d44;
    --accent: #00d4aa;
    --accent2: #ff6b6b;
    --accent3: #ffd166;
    --text: #e2e8f0;
    --muted: #64748b;
    --critical: #ef4444;
    --warning: #f59e0b;
    --ok: #10b981;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 24px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }
  .logo {
    width: 44px; height: 44px;
    background: var(--accent);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
  }
  h1 { font-family: var(--mono); font-size: 22px; color: var(--accent); letter-spacing: -0.5px; }
  .subtitle { color: var(--muted); font-size: 13px; }
  .badge { background: var(--border); color: var(--accent); font-family: var(--mono); font-size: 11px; padding: 3px 8px; border-radius: 4px; margin-left: 8px; }

  .grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .panel-title {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  .task-btn {
    width: 100%;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    margin-bottom: 8px;
    transition: all 0.15s;
  }
  .task-btn:hover { border-color: var(--accent); color: var(--accent); }
  .task-btn.active { border-color: var(--accent); background: rgba(0,212,170,0.08); color: var(--accent); }
  .task-difficulty {
    font-size: 10px;
    font-family: var(--mono);
    padding: 2px 6px;
    border-radius: 3px;
    float: right;
  }
  .easy   { background: rgba(16,185,129,0.15); color: var(--ok); }
  .medium { background: rgba(245,158,11,0.15); color: var(--warning); }
  .hard   { background: rgba(239,68,68,0.15);  color: var(--critical); }

  .vitals-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
  }
  .vital-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .vital-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .vital-value { font-family: var(--mono); font-size: 22px; }
  .vital-unit  { font-size: 11px; color: var(--muted); }
  .vital-bar   { height: 3px; background: var(--border); border-radius: 2px; margin-top: 8px; }
  .vital-fill  { height: 3px; border-radius: 2px; transition: width 0.4s; }

  .ok-color   { color: var(--ok); }
  .warn-color { color: var(--warning); }
  .crit-color { color: var(--critical); }
  .ok-bar     { background: var(--ok); }
  .warn-bar   { background: var(--warning); }
  .crit-bar   { background: var(--critical); }

  .action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 20px;
  }
  .action-btn {
    padding: 10px 8px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
    line-height: 1.3;
  }
  .action-btn:hover  { border-color: var(--accent); background: rgba(0,212,170,0.06); }
  .action-btn:active { transform: scale(0.97); }
  .action-btn.last-action { border-color: var(--accent2); background: rgba(255,107,107,0.08); }

  .log-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    height: 140px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
  .log-line { margin-bottom: 4px; }
  .log-reward-pos { color: var(--ok); }
  .log-reward-neg { color: var(--critical); }
  .log-step { color: var(--accent); }

  .score-bar-wrap { margin-top: 16px; }
  .score-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }
  .score-bar { height: 8px; background: var(--border); border-radius: 4px; }
  .score-fill { height: 8px; border-radius: 4px; background: var(--accent); transition: width 0.5s; }

  .ctrl-row { display: flex; gap: 10px; margin-bottom: 16px; }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn-primary { background: var(--accent); color: #000; }
  .btn-danger  { background: var(--accent2); color: #fff; }
  .btn-neutral { background: var(--border); color: var(--text); }

  .status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-family: var(--mono);
    margin-left: 12px;
  }
  .status-active { background: rgba(0,212,170,0.15); color: var(--accent); }
  .status-done   { background: rgba(239,68,68,0.15);  color: var(--critical); }
  .status-idle   { background: var(--border); color: var(--muted); }

  .sofa-score { font-family: var(--mono); font-size: 28px; margin: 8px 0; }
  .step-counter { font-family: var(--mono); color: var(--muted); font-size: 13px; }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--ok);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }
  .dead-dot { background: var(--critical); animation: none; }
</style>
</head>
<body>
<header>
  <div class="logo">🫀</div>
  <div>
    <h1>SepsisPilot <span class="badge">OpenEnv v1.0</span></h1>
    <div class="subtitle">ICU sepsis treatment RL environment — interactive demo</div>
  </div>
</header>

<div class="grid">
  <!-- Left: controls -->
  <div>
    <div class="panel" style="margin-bottom:16px">
      <div class="panel-title">Select Task</div>
      <button class="task-btn active" id="btn-task-0" onclick="selectTask('mild_sepsis')">
        Mild Sepsis <span class="task-difficulty easy">Easy</span>
      </button>
      <button class="task-btn" id="btn-task-1" onclick="selectTask('septic_shock')">
        Septic Shock <span class="task-difficulty medium">Medium</span>
      </button>
      <button class="task-btn" id="btn-task-2" onclick="selectTask('severe_mods')">
        Severe MODS <span class="task-difficulty hard">Hard</span>
      </button>
      <div style="margin-top:12px">
        <button class="btn btn-primary" style="width:100%" onclick="doReset()">↺ Reset Episode</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">SOFA Score</div>
      <div class="sofa-score" id="sofa-display">—</div>
      <div class="step-counter" id="step-display">Step — / —</div>
      <div class="score-bar-wrap">
        <div class="score-label"><span>Episode Score</span><span id="score-val">—</span></div>
        <div class="score-bar"><div class="score-fill" id="score-fill" style="width:0%"></div></div>
      </div>
    </div>
  </div>

  <!-- Right: vitals + actions -->
  <div>
    <div class="panel" style="margin-bottom:16px">
      <div class="panel-title">
        Patient Vitals
        <span class="status-pill status-idle" id="status-pill">IDLE</span>
      </div>
      <div class="vitals-grid">
        <div class="vital-card">
          <div class="vital-label">MAP</div>
          <div class="vital-value" id="v-map">—</div>
          <div class="vital-unit">mmHg &nbsp;goal ≥65</div>
          <div class="vital-bar"><div class="vital-fill" id="b-map" style="width:0%"></div></div>
        </div>
        <div class="vital-card">
          <div class="vital-label">Lactate</div>
          <div class="vital-value" id="v-lac">—</div>
          <div class="vital-unit">mmol/L &nbsp;goal &lt;2</div>
          <div class="vital-bar"><div class="vital-fill" id="b-lac" style="width:0%"></div></div>
        </div>
        <div class="vital-card">
          <div class="vital-label">WBC</div>
          <div class="vital-value" id="v-wbc">—</div>
          <div class="vital-unit">k/uL &nbsp;normal 4–12</div>
          <div class="vital-bar"><div class="vital-fill" id="b-wbc" style="width:0%"></div></div>
        </div>
        <div class="vital-card">
          <div class="vital-label">Temperature</div>
          <div class="vital-value" id="v-temp">—</div>
          <div class="vital-unit">°C &nbsp;goal 36–38</div>
          <div class="vital-bar"><div class="vital-fill" id="b-temp" style="width:0%"></div></div>
        </div>
        <div class="vital-card">
          <div class="vital-label">Heart Rate</div>
          <div class="vital-value" id="v-hr">—</div>
          <div class="vital-unit">bpm &nbsp;goal &lt;100</div>
          <div class="vital-bar"><div class="vital-fill" id="b-hr" style="width:0%"></div></div>
        </div>
        <div class="vital-card">
          <div class="vital-label">Creatinine</div>
          <div class="vital-value" id="v-cr">—</div>
          <div class="vital-unit">mg/dL &nbsp;normal &lt;1.2</div>
          <div class="vital-bar"><div class="vital-fill" id="b-cr" style="width:0%"></div></div>
        </div>
      </div>
    </div>

    <div class="panel" style="margin-bottom:16px">
      <div class="panel-title">Choose Treatment Action</div>
      <div class="action-grid" id="action-grid">
        <button class="action-btn" onclick="doStep(0)">🚫<br>No Treatment</button>
        <button class="action-btn" onclick="doStep(1)">💊<br>Broad AB</button>
        <button class="action-btn" onclick="doStep(2)">💉<br>Narrow AB</button>
        <button class="action-btn" onclick="doStep(3)">⬆️<br>Low Vaso</button>
        <button class="action-btn" onclick="doStep(4)">⬆️⬆️<br>High Vaso</button>
        <button class="action-btn" onclick="doStep(5)">💊+⬆️<br>Broad+Low</button>
        <button class="action-btn" onclick="doStep(6)">💊+⬆️⬆️<br>Broad+High</button>
        <button class="action-btn" onclick="doStep(7)">💉+⬆️<br>Narrow+Low</button>
        <button class="action-btn" onclick="doStep(8)">💉+⬆️⬆️<br>Narrow+High</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Step Log</div>
      <div class="log-box" id="log-box">
        <div class="log-line" style="color:var(--accent)">Select a task and click Reset to begin.</div>
      </div>
    </div>
  </div>
</div>

<script>
const BASE = '';
let currentTask = 'mild_sepsis';
let episodeDone = false;

function selectTask(t) {
  currentTask = t;
  ['mild_sepsis','septic_shock','severe_mods'].forEach((task, i) => {
    document.getElementById('btn-task-' + i).classList.toggle('active', task === t);
  });
}

async function doReset() {
  const r = await fetch(BASE + '/reset', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({task: currentTask, seed: 42})
  });
  const data = await r.json();
  episodeDone = false;
  updateUI(data, null, null);
  clearLog();
  logLine(`Episode started: <span style="color:var(--accent)">${currentTask}</span>`, 'plain');
  document.getElementById('status-pill').textContent = '● LIVE';
  document.getElementById('status-pill').className = 'status-pill status-active';
  document.getElementById('score-val').textContent = '—';
  document.getElementById('score-fill').style.width = '0%';
}

async function doStep(action) {
  if (episodeDone) { logLine('Episode done. Reset to play again.', 'plain'); return; }
  const r = await fetch(BASE + '/step', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action})
  });
  const data = await r.json();
  updateUI(data.state, data.reward, action);
  const rewardClass = data.reward >= 0 ? 'log-reward-pos' : 'log-reward-neg';
  logLine(
    `<span class="log-step">Step ${data.state.step}</span>  action=${action}  ` +
    `MAP=${data.state.vitals.map_mmhg.toFixed(1)}  ` +
    `Lac=${data.state.vitals.lactate.toFixed(2)}  ` +
    `reward=<span class="${rewardClass}">${data.reward >= 0 ? '+' : ''}${data.reward.toFixed(3)}</span>`,
    'plain'
  );
  if (data.done) {
    episodeDone = true;
    document.getElementById('status-pill').textContent = data.state.alive ? '✓ STABLE' : '✗ DEAD';
    document.getElementById('status-pill').className = 'status-pill ' + (data.state.alive ? 'status-active' : 'status-done');
    await fetchGrade();
  }
  // Highlight last action
  document.querySelectorAll('.action-btn').forEach((b, i) => b.classList.toggle('last-action', i === action));
}

async function fetchGrade() {
  const r = await fetch(BASE + '/grade');
  const g = await r.json();
  document.getElementById('score-val').textContent = (g.score * 100).toFixed(1) + '%';
  document.getElementById('score-fill').style.width = (g.score * 100) + '%';
  logLine(`[GRADE] score=${g.score.toFixed(4)} — ${g.reason}`, 'plain');
}

function updateUI(state, reward, action) {
  const v = state.vitals;
  setVital('map', v.map_mmhg, 'mmHg', 40, 100, 65, 100);
  setVital('lac', v.lactate,  'mmol/L', 0, 12, 0.5, 2.0, true);
  setVital('wbc', v.wbc,      'k/uL', 0, 30, 4, 12);
  setVital('temp', v.temperature, '°C', 33, 42, 36.5, 38.0);
  setVital('hr', v.heart_rate, 'bpm', 20, 160, 60, 100);
  setVital('cr', v.creatinine, 'mg/dL', 0, 6, 0.6, 1.2);
  document.getElementById('sofa-display').textContent = v.sofa_score.toFixed(1);
  document.getElementById('step-display').textContent = `Step ${state.step} / ${state.max_steps}`;
}

function setVital(id, value, unit, min, max, goodMin, goodMax, invertBad=false) {
  const el   = document.getElementById('v-' + id);
  const bar  = document.getElementById('b-' + id);
  const pct  = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const good = value >= goodMin && value <= goodMax;
  const warn = !good && (value > goodMin * 0.7 && value < goodMax * 1.5);

  el.textContent = typeof value === 'number' ? value.toFixed(1) : '—';
  el.className   = 'vital-value ' + (good ? 'ok-color' : warn ? 'warn-color' : 'crit-color');
  bar.className  = 'vital-fill ' + (good ? 'ok-bar' : warn ? 'warn-bar' : 'crit-bar');
  bar.style.width = pct + '%';
}

function logLine(html, type) {
  const box = document.getElementById('log-box');
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function clearLog() {
  document.getElementById('log-box').innerHTML = '';
}

// Auto-reset on load for first-time visitors
window.addEventListener('load', doReset);
</script>
</body>
</html>
"""
