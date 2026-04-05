---
title: SepsisPilot
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---
# 🫀 SepsisPilot — OpenEnv

> **Reinforcement learning environment for optimal sepsis treatment sequencing**  
> Meta PyTorch OpenEnv Hackathon 2026 — Submission

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-00d4aa?style=flat-square)](https://huggingface.co)
[![HF Spaces](https://img.shields.io/badge/🤗%20Spaces-SepsisPilot-yellow?style=flat-square)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square)](https://fastapi.tiangolo.com)

---

## Environment Description & Motivation

**Sepsis kills ~11 million people per year** — yet optimal treatment sequencing remains one of the hardest challenges in critical care. The right antibiotic at the right time, combined with precise vasopressor dosing, can mean the difference between survival and multi-organ failure.

SepsisPilot simulates an ICU sepsis patient at hourly resolution. An AI agent observes real clinical vitals (MAP, lactate, WBC, temperature, heart rate, creatinine) and decides which antibiotic and vasopressor combination to administer each hour. The environment models realistic physiology including:

- **Gram-specific antibiotic efficacy** — broad-spectrum covers gram-negative; narrow-spectrum (vancomycin) covers gram-positive
- **Antibiotic resistance accumulation** — repeated suboptimal antibiotic use degrades efficacy
- **Haemodynamic-metabolic coupling** — low MAP causes tissue ischaemia (rising lactate), compensatory tachycardia
- **Renal vasoconstriction** — high-dose vasopressors raise MAP but risk acute kidney injury

This is a **real clinical problem** with life-or-death stakes, well-defined physiology, meaningful partial progress signals, and a genuinely hard exploration challenge — making it an ideal RL environment.

---

## Action Space

| ID | Name | Description |
|----|------|-------------|
| 0 | `no_treatment` | Watchful waiting — no intervention |
| 1 | `broad_antibiotics` | Piperacillin-tazobactam — gram-negative coverage |
| 2 | `narrow_antibiotics` | Vancomycin — gram-positive / MRSA coverage |
| 3 | `low_vasopressor` | Norepinephrine 0.1 mcg/kg/min — raises MAP |
| 4 | `high_vasopressor` | Norepinephrine 0.3 mcg/kg/min — raises MAP more; ⚠ renal risk |
| 5 | `broad_plus_low_vaso` | Broad-spectrum AB + low-dose vasopressor |
| 6 | `broad_plus_high_vaso` | Broad-spectrum AB + high-dose vasopressor |
| 7 | `narrow_plus_low_vaso` | Narrow-spectrum AB + low-dose vasopressor |
| 8 | `narrow_plus_high_vaso` | Narrow-spectrum AB + high-dose vasopressor |

**Type:** Discrete · **n:** 9

---

## Observation Space

| Field | Unit | Normal Range | Clinical Meaning |
|-------|------|-------------|-----------------|
| `map_mmhg` | mmHg | 70–100 | Mean Arterial Pressure — sepsis goal ≥ 65 |
| `lactate` | mmol/L | 0.5–2.0 | Tissue ischaemia marker — target < 2.0 |
| `wbc` | k/uL | 4–11 | White blood cells — infection proxy |
| `temperature` | °C | 36.5–37.5 | Fever indicates active infection |
| `heart_rate` | bpm | 60–100 | Tachycardia in sepsis |
| `creatinine` | mg/dL | 0.6–1.2 | Renal function — rises with AKI |
| `sofa_score` | 0–24 | 0–2 | Multi-organ failure composite |
| `resistance` | 0–1 | 0.0 | Antibiotic resistance index (hard task) |
| `step_fraction` | 0–1 | — | Fraction of episode elapsed |

**Type:** Continuous · **Shape:** [9]

---

## Task Descriptions

### Task 1 — `mild_sepsis` · Easy

- **Scenario:** Mild sepsis secondary to gram-negative urinary tract infection
- **Initial state:** MAP 65, Lactate 2.5, WBC 14, Temp 38.2°C
- **Optimal strategy:** Broad-spectrum antibiotics; vasopressors only if MAP drops below 65
- **Max steps:** 24 (24 hours)
- **Expected baseline score:** 0.55–0.75
- **Key challenge:** Learning that broad-spectrum is the right antibiotic class

### Task 2 — `septic_shock` · Medium

- **Scenario:** Septic shock from gram-positive bacteraemia (MRSA suspected)
- **Initial state:** MAP 52 ⚠, Lactate 4.2, WBC 18, Temp 38.9°C
- **Optimal strategy:** Immediate vasopressors + narrow-spectrum antibiotics (vancomycin). Every delayed hour increases organ failure risk.
- **Max steps:** 48 (48 hours)
- **Expected baseline score:** 0.35–0.60
- **Key challenge:** Correctly identifying gram-positive infection; mandatory haemodynamic support

### Task 3 — `severe_mods` · Hard

- **Scenario:** Severe sepsis with Multi-Organ Dysfunction Syndrome (MODS). Mixed drug-resistant infection.
- **Initial state:** MAP 42 ⚠⚠, Lactate 7.0, WBC 22, Temp 39.6°C, Creatinine 2.2
- **Optimal strategy:** Broad-spectrum first (2 steps) → switch to narrow-spectrum → maintain MAP ≥ 65 with lowest effective vasopressor dose
- **Max steps:** 72 (72 hours)
- **Expected baseline score:** 0.20–0.45
- **Key challenge:** Precise antibiotic sequencing to manage resistance; renal protection; multi-objective optimisation

---

## Reward Function

Dense rewards at every timestep (not just episode end):

```
Per step:
  +0.35  MAP ≥ 65 mmHg                  (haemodynamic stability)
  +0.30  Lactate < 2.0 mmol/L           (tissue perfusion restored)
  +0.10  WBC in 4–12 k/uL              (infection controlled)
  +0.08  Temperature 36–38°C            (fever resolved)
  +0.05  Creatinine improving            (renal protection)
  −0.15  Resistance increasing           (wrong antibiotic penalty)
  −0.025 Per step                        (time pressure)

Terminal:
  +5.00  All vitals stable               (full stabilisation bonus)
  −8.00  Patient death                   (MAP < 35 or Lactate > 15)
```

**Range:** approximately −8.0 to +5.775 per step.

---

## Grader (0.0 → 1.0)

Each completed episode is scored by a task-specific grader:

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Survival | 40% | 30% | 25% |
| MAP normalisation | 25% | 20% | — |
| Lactate clearance | 20% | 15% | — |
| Vital combo (MAP + lactate) | — | — | 20% |
| WBC / temperature | 10% | 5% | — |
| Correct antibiotic class | — | 15% | — |
| Vasopressor usage | — | 5% | — |
| Antibiotic sequencing | — | — | 15% |
| Resistance management | — | — | 15% |
| Renal protection | — | — | 15% |
| Speed bonus | 5% | 5% | 10% |

---

## Baseline Scores

Baseline LLM agent (Nemotron 3 Super via NVIDIA API, seed=42):

| Task | Baseline Score | Notes |
|------|---------------|-------|
| `mild_sepsis` | ~0.62 | LLM correctly identifies broad-spectrum; moderate speed |
| `septic_shock` | ~0.44 | Often misses narrow-spectrum; vasopressors applied correctly |
| `severe_mods` | ~0.31 | Sequencing rarely optimal; resistance accumulates |

Random agent (action sampled uniformly):

| Task | Random Score |
|------|-------------|
| `mild_sepsis` | ~0.35 |
| `septic_shock` | ~0.18 |
| `severe_mods` | ~0.08 |

---

## Setup & Usage

### Quick Start (Docker)

```bash
# Build
docker build -t sepsispilot .

# Run
docker run -p 7860:7860 sepsispilot

# Test
curl http://localhost:7860/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Open dashboard
open http://localhost:7860
```

### Running the Baseline Agent

```bash
export OPENAI_API_KEY="your-api-key"
export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="nvidia/llama-3.1-nemotron-70b-instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
# Runs all 3 tasks, 1 episode each (seed=42)

python inference.py --episodes 3 --seed 42
# 3 episodes per task

python inference.py --task mild_sepsis
# Single task only
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

### Pre-Submission Validation

```bash
# With server running:
python validate.py --url http://localhost:7860
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode `{"task": "mild_sepsis", "seed": 42}` |
| `POST` | `/step` | Take action `{"action": 5}` |
| `GET` | `/state` | Current patient state |
| `GET` | `/grade` | Score completed episode (0.0–1.0) |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Server health check |
| `GET` | `/` | Interactive visual dashboard |

Interactive API docs: `http://localhost:7860/docs`

---

## Project Structure

```
sepsispilot/
├── openenv.yaml          ← OpenEnv spec config
├── Dockerfile            ← HF Spaces container
├── requirements.txt
├── README.md
├── inference.py          ← Baseline LLM agent (mandatory)
├── validate.py           ← Pre-submission validation
├── app.py                ← FastAPI HTTP server + dashboard
├── environment/
│   ├── __init__.py
│   ├── models.py         ← Pydantic typed models
│   ├── patient_sim.py    ← Physiology simulation engine
│   ├── graders.py        ← Episode scoring (0.0–1.0)
│   └── env.py            ← OpenEnv class (reset/step/state/grade)
└── tests/
    └── test_env.py       ← Unit tests (pytest)
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | API key for LLM endpoint |
| `API_BASE_URL` | ✅ | `https://integrate.api.nvidia.com/v1` | LLM endpoint |
| `MODEL_NAME` | ✅ | `nvidia/llama-3.1-nemotron-70b-instruct` | Model name |
| `HF_TOKEN` | ✅ | — | Hugging Face token |
| `ENV_BASE_URL` | ❌ | `http://localhost:7860` | Environment server URL |

---

## Design Decisions

**Why sepsis?** It's a genuine, high-stakes clinical problem where optimal treatment sequencing has enormous impact. The physiology is well-understood but the decision-making is hard — perfect for RL.

**Why synthetic simulation instead of direct MIMIC-IV replay?** MIMIC-IV access requires credentialing. The simulation is calibrated to match MIMIC-IV population statistics, making it accessible while remaining medically realistic. An RL agent trained here can be evaluated against real MIMIC-IV data in future work.

**Why dense rewards?** Sepsis treatment spans 24–72 hours. Episode-end-only rewards create too sparse a signal for meaningful learning. Per-step vital sign improvements provide rich learning signal throughout.

---

*Built with ❤️ for the Meta PyTorch OpenEnv Hackathon 2026*
