"""
SepsisPilot — Baseline Inference Script
Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks.

MANDATORY output format (do not modify):
  [START] task=<name> episode=<n>
  [STEP] step=<n> action=<n> reward=<float> done=<bool> score=<float>
  [END] task=<name> episode=<n> score=<float>

Environment variables (required):
  OPENAI_API_KEY   — API key for the LLM endpoint
  API_BASE_URL     — LLM endpoint base URL  (default: https://integrate.api.nvidia.com/v1)
  MODEL_NAME       — Model identifier       (default: nvidia/llama-3.1-nemotron-70b-instruct)
  ENV_BASE_URL     — SepsisPilot server URL (default: http://localhost:7860)

Usage:
  python inference.py
  python inference.py --episodes 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://integrate.api.nvidia.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "nvidia/llama-3.1-nemotron-70b-instruct")
ENV_BASE_URL   = os.environ.get("ENV_BASE_URL",   "http://localhost:7860")
HF_TOKEN       = os.environ.get("HF_TOKEN",       "")

TASKS          = ["mild_sepsis", "septic_shock", "severe_mods"]
MAX_STEPS_MAP  = {"mild_sepsis": 24, "septic_shock": 48, "severe_mods": 72}

ACTION_DESCRIPTIONS = {
    0: "No treatment — watchful waiting",
    1: "Broad-spectrum antibiotics (gram-negative coverage, e.g. pip-tazo)",
    2: "Narrow-spectrum antibiotics (gram-positive coverage, e.g. vancomycin)",
    3: "Low-dose vasopressor (norepinephrine 0.1 mcg/kg/min)",
    4: "High-dose vasopressor (norepinephrine 0.3 mcg/kg/min — renal risk)",
    5: "Broad-spectrum antibiotics + low-dose vasopressor",
    6: "Broad-spectrum antibiotics + high-dose vasopressor",
    7: "Narrow-spectrum antibiotics + low-dose vasopressor",
    8: "Narrow-spectrum antibiotics + high-dose vasopressor",
}

SYSTEM_PROMPT = """\
You are an expert ICU physician and clinical decision-support AI.
You are treating a sepsis patient in a simulated ICU environment.
At each step, you observe patient vitals and must choose exactly ONE integer action (0–8).

ACTIONS:
0: No treatment
1: Broad-spectrum antibiotics (covers gram-negative bacteria — pip-tazo)
2: Narrow-spectrum antibiotics (covers gram-positive bacteria — vancomycin)
3: Low-dose vasopressor (norepinephrine 0.1 mcg/kg/min — raises MAP)
4: High-dose vasopressor (norepinephrine 0.3 mcg/kg/min — raises MAP more, risks kidney damage)
5: Broad-spectrum antibiotics + low-dose vasopressor
6: Broad-spectrum antibiotics + high-dose vasopressor
7: Narrow-spectrum antibiotics + low-dose vasopressor
8: Narrow-spectrum antibiotics + high-dose vasopressor

CLINICAL GOALS:
- MAP (mean arterial pressure): keep ≥ 65 mmHg
- Lactate: reduce to < 2.0 mmol/L (elevated lactate = tissue ischaemia)
- WBC: normalise toward 4–12 k/uL
- Temperature: target 36.5–37.5°C
- Heart rate: target < 100 bpm
- Creatinine: protect kidneys, avoid rise (especially with high-dose vasopressors)

STRATEGY HINTS:
- Prioritise source control (antibiotics) AND haemodynamic support (vasopressors) simultaneously
- If MAP < 65: vasopressors are mandatory
- For gram-negative infections: broad-spectrum is optimal
- For gram-positive infections (MRSA): narrow-spectrum (vancomycin) is optimal
- For mixed/resistant infections: start broad, then switch to narrow to limit resistance

Respond with ONLY a JSON object: {"action": <integer 0-8>, "reasoning": "<one sentence>"}
"""


# ──────────────────────────────────────────────
# Environment client
# ──────────────────────────────────────────────

def env_reset(task: str, seed: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task, "seed": seed},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def env_grade() -> Dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}/grade", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# LLM Agent
# ──────────────────────────────────────────────

def build_state_prompt(state: Dict[str, Any]) -> str:
    v = state["vitals"]
    return f"""\
CURRENT PATIENT STATE:
  Task:        {state['task']}
  Step:        {state['step']} / {state['max_steps']}
  MAP:         {v['map_mmhg']:.1f} mmHg    {'⚠ CRITICAL <65' if v['map_mmhg'] < 65 else '✓ OK'}
  Lactate:     {v['lactate']:.2f} mmol/L   {'⚠ HIGH' if v['lactate'] > 2.0 else '✓ OK'}
  WBC:         {v['wbc']:.1f} k/uL         {'⚠ ELEVATED' if v['wbc'] > 12 else '✓ OK'}
  Temperature: {v['temperature']:.1f} °C   {'⚠ FEVER' if v['temperature'] > 38.0 else '✓ OK'}
  Heart Rate:  {v['heart_rate']:.0f} bpm   {'⚠ TACHYCARDIA' if v['heart_rate'] > 100 else '✓ OK'}
  Creatinine:  {v['creatinine']:.2f} mg/dL {'⚠ AKI risk' if v['creatinine'] > 1.5 else '✓ OK'}
  SOFA Score:  {v['sofa_score']:.1f}
  Resistance:  {v['resistance']:.3f}

Choose the optimal treatment action (0–8). Reply ONLY with JSON: {{"action": N, "reasoning": "..."}}"""


def choose_action(
    llm_client: OpenAI,
    state: Dict[str, Any],
    history: list,
) -> int:
    """Ask the LLM to choose an action. Falls back to heuristic if LLM fails."""
    history.append({"role": "user", "content": build_state_prompt(state)})

    try:
        time.sleep(6)
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:],
            max_tokens=120,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON; strip possible markdown fences
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        action = int(parsed["action"])
        reasoning = parsed.get("reasoning", "")

        if not (0 <= action <= 8):
            raise ValueError(f"Action {action} out of range")

        # Add assistant turn to history
        history.append({"role": "assistant", "content": raw})
        return action

    except Exception as exc:
        # Heuristic fallback: if MAP low → vasopressor; add AB always
        sys.stderr.write(f"[LLM ERROR] {exc} — using heuristic fallback\n")
        v = state["vitals"]
        map_low = v["map_mmhg"] < 65
        return 5 if map_low else 1   # broad AB + low vaso OR broad AB alone


# ──────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────

def run_episode(
    llm_client: OpenAI,
    task: str,
    episode: int,
    seed: int,
) -> float:
    """Run one full episode. Returns final episode score."""
    # ── [START] ──────────────────────────────────────────────
    print(f"[START] task={task} episode={episode}", flush=True)

    state = env_reset(task, seed)
    history: list = []
    cumulative_reward = 0.0
    step = 0
    done = False
    running_score = 0.0

    while not done:
        action = choose_action(llm_client, state, history)

        # Safety guard: never allow action 0 in critical tasks
        task_critical = task in ("septic_shock", "severe_mods")
        if task_critical and action == 0:
            sys.stderr.write(f"[SAFETY] Blocked action=0 on {task}, overriding to action=5\n")
            action = 5

        result = env_step(action)

        step        = result["state"]["step"]
        reward      = result["reward"]
        done        = result["done"]
        state       = result["state"]
        cumulative_reward += reward

        # Running score heuristic for [STEP] output (grade only available after done)
        # Approximate: normalise cumulative_reward to [0,1]
        max_steps = MAX_STEPS_MAP.get(task, 48)
        running_score = max(0.0, min(1.0, (cumulative_reward + 8.0) / 20.0))

        # ── [STEP] ───────────────────────────────────────────
        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward:.4f} done={done} score={running_score:.4f}",
            flush=True,
        )

        if done:
            break

        time.sleep(1)   # small delay to avoid rate-limiting

    # Fetch official grader score
    try:
        grade_result = env_grade()
        final_score  = grade_result["score"]
    except Exception:
        final_score = running_score

    # ── [END] ────────────────────────────────────────────────
    print(f"[END] task={task} episode={episode} score={final_score:.4f}", flush=True)
    return final_score


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SepsisPilot Baseline Inference")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run only a specific task (default: all 3)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        sys.exit("[ERROR] OPENAI_API_KEY environment variable is not set.")

    llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    tasks_to_run = [args.task] if args.task else TASKS
    all_scores: Dict[str, list] = {}

    for task in tasks_to_run:
        all_scores[task] = []
        for ep in range(1, args.episodes + 1):
            ep_seed = args.seed + ep
            score = run_episode(llm_client, task, ep, seed=ep_seed)
            all_scores[task].append(score)

    # Summary to stderr (not stdout — stdout reserved for [START]/[STEP]/[END])
    sys.stderr.write("\n=== Baseline Summary ===\n")
    for task, scores in all_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        sys.stderr.write(f"  {task}: avg_score={avg:.4f} over {len(scores)} episode(s)\n")


if __name__ == "__main__":
    main()
