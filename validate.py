"""
SepsisPilot — Pre-Submission Validation Script
Run this before submitting to verify OpenEnv spec compliance.

Usage: python validate.py [--url http://localhost:7860]
"""

from __future__ import annotations
import argparse
import sys
import requests

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = 0


def check(label: str, condition: bool, msg: str = ""):
    global errors
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label} {msg}")
        errors += 1


def section(title: str):
    print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


def validate(base_url: str):
    global errors
    print(f"\n🔬 SepsisPilot OpenEnv Validation\n   Target: {base_url}\n")

    # ── 1. Health ───────────────────────────────
    section("1. Health Check")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        check("GET /health returns 200", r.status_code == 200)
        data = r.json()
        check("Response contains 'status'", "status" in data)
        check("Status is 'ok'", data.get("status") == "ok")
    except Exception as e:
        check("Server reachable", False, str(e))
        print("\n  [ABORT] Server not reachable. Start the server first.\n")
        sys.exit(1)

    # ── 2. Tasks ────────────────────────────────
    section("2. Task Listing")
    try:
        r = requests.get(f"{base_url}/tasks", timeout=10)
        check("GET /tasks returns 200", r.status_code == 200)
        tasks = r.json()
        check("Returns a list", isinstance(tasks, list))
        check("At least 3 tasks", len(tasks) >= 3)
        task_names = [t["name"] for t in tasks]
        check("mild_sepsis present", "mild_sepsis" in task_names)
        check("septic_shock present", "septic_shock" in task_names)
        check("severe_mods present", "severe_mods" in task_names)
        for t in tasks:
            check(f"  Task '{t['name']}' has difficulty", "difficulty" in t)
            check(f"  Task '{t['name']}' has description", "description" in t)
            check(f"  Task '{t['name']}' has max_steps", "max_steps" in t)
    except Exception as e:
        check("Tasks endpoint works", False, str(e))

    # ── 3. Episode — mild_sepsis ─────────────
    section("3. Episode Flow — mild_sepsis (Easy)")
    _validate_episode(base_url, "mild_sepsis", max_steps=24)

    # ── 4. Episode — septic_shock ────────────
    section("4. Episode Flow — septic_shock (Medium)")
    _validate_episode(base_url, "septic_shock", max_steps=48)

    # ── 5. Episode — severe_mods ─────────────
    section("5. Episode Flow — severe_mods (Hard)")
    _validate_episode(base_url, "severe_mods", max_steps=72)

    # ── 6. Grader variance ──────────────────
    section("6. Grader Score Variance (anti-trivial check)")
    scores = []
    actions_list = [
        [5, 5, 5, 1, 1, 1],   # broad + low vaso (good)
        [0, 0, 0, 0, 0, 0],   # no treatment (bad)
        [4, 4, 4, 4, 4, 4],   # high vaso only (wrong)
    ]
    for i, actions in enumerate(actions_list):
        try:
            r = requests.post(f"{base_url}/reset", json={"task": "mild_sepsis", "seed": 42}, timeout=10)
            for a in actions:
                r = requests.post(f"{base_url}/step", json={"action": a}, timeout=10)
                if r.json().get("done"):
                    break
            # Force episode end
            while not r.json().get("done"):
                r = requests.post(f"{base_url}/step", json={"action": 0}, timeout=10)
            grade = requests.get(f"{base_url}/grade", timeout=10).json()
            scores.append(grade["score"])
        except Exception as e:
            scores.append(None)
            print(f"  {WARN} Strategy {i} failed: {e}")

    valid_scores = [s for s in scores if s is not None]
    check("Grader returns different scores for different strategies", 
          len(set(round(s, 2) for s in valid_scores)) > 1,
          f"(scores: {[round(s,4) for s in valid_scores]})")
    check("All scores in [0.0, 1.0]",
          all(0.0 <= s <= 1.0 for s in valid_scores))

    # ── 7. Reproducibility ──────────────────
    section("7. Reproducibility (same seed = same result)")
    try:
        scores_run1, scores_run2 = [], []
        for run_scores in (scores_run1, scores_run2):
            requests.post(f"{base_url}/reset", json={"task": "mild_sepsis", "seed": 99}, timeout=10)
            for _ in range(5):
                r = requests.post(f"{base_url}/step", json={"action": 5}, timeout=10)
                run_scores.append(round(r.json()["reward"], 4))
                if r.json()["done"]:
                    break
        check("Reward sequences are identical across runs", scores_run1 == scores_run2,
              f"\n    run1={scores_run1}\n    run2={scores_run2}")
    except Exception as e:
        check("Reproducibility check", False, str(e))

    # ── 8. Error handling ───────────────────
    section("8. Error Handling")
    try:
        r = requests.post(f"{base_url}/step", json={"action": 99}, timeout=10)
        check("Invalid action returns 4xx", r.status_code in (400, 422))
    except Exception as e:
        check("Invalid action error handling", False, str(e))

    # ── Summary ─────────────────────────────
    print(f"\n{'═'*50}")
    if errors == 0:
        print(f"  ✅  All checks passed. Ready for submission!")
    else:
        print(f"  ❌  {errors} check(s) failed. Fix before submitting.")
    print(f"{'═'*50}\n")
    sys.exit(0 if errors == 0 else 1)


def _validate_episode(base_url: str, task: str, max_steps: int):
    """Run a short episode and verify all OpenEnv contracts."""
    try:
        # Reset
        r = requests.post(f"{base_url}/reset", json={"task": task, "seed": 42}, timeout=10)
        check(f"POST /reset 200", r.status_code == 200)
        state = r.json()
        check("Reset returns vitals", "vitals" in state)
        check("Reset returns step=0", state.get("step") == 0)
        check("Reset returns done=False", state.get("done") == False)
        check("Reset returns alive=True", state.get("alive") == True)

        # State endpoint
        r = requests.get(f"{base_url}/state", timeout=10)
        check("GET /state 200", r.status_code == 200)

        # Step
        r = requests.post(f"{base_url}/step", json={"action": 5}, timeout=10)
        check("POST /step 200", r.status_code == 200)
        result = r.json()
        check("Step returns state", "state" in result)
        check("Step returns reward (float)", isinstance(result.get("reward"), (int, float)))
        check("Step returns done (bool)", isinstance(result.get("done"), bool))
        check("Step returns info (dict)", isinstance(result.get("info"), dict))
        check("Step increments step counter", result["state"]["step"] == 1)

        # Reward range check
        reward = result["reward"]
        check("Reward is finite and in expected range",
              -15.0 <= reward <= 10.0, f"(got {reward})")

        # Run until done (fast — use fixed action)
        done = result["done"]
        for _ in range(max_steps):
            if done:
                break
            r = requests.post(f"{base_url}/step", json={"action": 5}, timeout=10)
            done = r.json()["done"]

        # Grade
        r = requests.get(f"{base_url}/grade", timeout=10)
        check("GET /grade 200 after episode", r.status_code == 200)
        grade = r.json()
        check("Grade has score in [0,1]",
              isinstance(grade.get("score"), (int, float)) and 0.0 <= grade["score"] <= 1.0,
              f"(got {grade.get('score')})")
        check("Grade has reason string", isinstance(grade.get("reason"), str))
        check("Grade has metrics dict", isinstance(grade.get("metrics"), dict))
        check("Grade has passed bool", isinstance(grade.get("passed"), bool))

    except Exception as e:
        check(f"Episode for {task} completed without error", False, str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()
    validate(args.url)
