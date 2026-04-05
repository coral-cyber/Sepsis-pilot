---
title: SepsisPilot
emoji: 🫀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# SepsisPilot — OpenEnv ICU Agent

> Meta PyTorch OpenEnv Hackathon 2026

An **OpenEnv-compliant reinforcement learning environment** for optimal sepsis treatment sequencing. An AI agent observes ICU patient vitals and learns which antibiotic and vasopressor combinations to administer each hour.

---

## What It Does

SepsisPilot simulates realistic sepsis physiology across three difficulty levels. At each step (= 1 ICU hour), the agent receives patient vitals and chooses one of 9 treatment actions. The environment returns a dense reward signal and a graded score at episode end.

```
mild_sepsis   → gram-negative UTI     → 24 steps  [EASY]
septic_shock  → gram-positive MRSA    → 48 steps  [MEDIUM]
severe_mods   → mixed drug-resistant  → 72 steps  [HARD]
```

---

## OpenEnv API

The environment runs as an HTTP server on port 7860.

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Current patient state |
| `/grade` | GET | Score a completed episode |
| `/tasks` | GET | List all tasks |
| `/health` | GET | Liveness check |
| `/` | GET | Interactive dashboard |

### Reset
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "mild_sepsis", "seed": 42}'
```

### Step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 5}'
```

### Grade
```bash
curl http://localhost:7860/grade
```

---

## Action Space

| Action | Description |
|---|---|
| 0 | No treatment |
| 1 | Broad-spectrum antibiotics (pip-tazo) |
| 2 | Narrow-spectrum antibiotics (vancomycin) |
| 3 | Low-dose vasopressor (0.1 mcg/kg/min) |
| 4 | High-dose vasopressor (0.3 mcg/kg/min) |
| 5 | Broad AB + Low vasopressor |
| 6 | Broad AB + High vasopressor |
| 7 | Narrow AB + Low vasopressor |
| 8 | Narrow AB + High vasopressor |

---

## Observation Space

8 continuous vitals per step:

| Field | Unit | Sepsis Goal |
|---|---|---|
| `map_mmhg` | mmHg | ≥ 65 |
| `lactate` | mmol/L | < 2.0 |
| `wbc` | k/uL | 4–12 |
| `temperature` | °C | 36–38 |
| `heart_rate` | bpm | < 100 |
| `creatinine` | mg/dL | < 1.2 |
| `sofa_score` | 0–24 | lower is better |
| `resistance` | 0–1 | lower is better |

---

## Local Setup

```bash
git clone https://github.com/<your-username>/SepsisPilot
cd SepsisPilot

python -m venv venv
.\venv\Scripts\activate        # Windows PowerShell
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Run the environment server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run the demo (no internet required)
```bash
python demo.py
```

### Run the inference agent
```bash
# Offline (heuristic only — no API key needed)
python inference.py --episodes 1 --seed 42

# With LLM (set your HuggingFace token)
set HF_TOKEN=your_token_here        # Windows
export HF_TOKEN=your_token_here     # Mac/Linux
python inference.py --episodes 1 --seed 42
```

---

## Docker

```bash
docker build -t sepsispilot .
docker run -p 7860:7860 sepsispilot
```

---

## Inference Output Format

The inference script emits exactly this format to stdout:

```
[START] task=mild_sepsis env=sepsis_pilot model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=broad_ab_low_vaso reward=0.42 done=false error=null
[STEP] step=2 action=broad_ab_low_vaso reward=0.38 done=false error=null
...
[END] success=true steps=12 score=0.87 rewards=0.42,0.38,...
```

---

## Grading Logic

Each task uses a composite grader (score in [0.0, 1.0]):

**mild_sepsis**: 40% survival + 25% MAP + 20% lactate + 10% WBC + 5% speed

**septic_shock**: 30% survival + 20% MAP + 15% lactate + 15% correct antibiotic + 5% vasopressor + 5% creatinine + 5% WBC + 5% speed

**severe_mods**: 25% survival + 20% vitals + 15% antibiotic sequencing + 15% resistance management + 15% renal protection + 10% speed

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace / API key | — |
| `API_BASE_URL` | LLM endpoint URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `LOCAL_IMAGE_NAME` | Docker image name | — |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## Project Structure

```
SepsisPilot/
├── app.py              # FastAPI server (OpenEnv HTTP API + dashboard)
├── inference.py        # Inference agent (OpenAI client + heuristic fallback)
├── demo.py             # Standalone demo (no server needed)
├── Dockerfile          # HuggingFace Spaces / Docker deployment
├── requirements.txt    # Python dependencies
├── environment/
│   ├── __init__.py
│   ├── env.py          # SepsisPilotEnv (reset/step/grade)
│   ├── patient_sim.py  # Physiology simulator
│   ├── graders.py      # Episode graders (3 tasks)
│   └── models.py       # Pydantic models (OpenEnv spec)
├── test_env.py         # Unit tests
└── validate.py         # Pre-submission OpenEnv compliance checker
```

---

## Built With

- [OpenEnv](https://github.com/huggingface/openenv) — Meta × HuggingFace RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) — HTTP server
- [Pydantic v2](https://docs.pydantic.dev) — data validation
- [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM client
