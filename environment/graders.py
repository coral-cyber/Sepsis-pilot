"""
SepsisPilot — Episode Graders
Each grader evaluates a completed episode and returns a score in [0.0, 1.0].

Grading philosophy:
  - Alive at end:          mandatory (0.0 if dead)
  - Vital improvement:     proportional credit for partial recovery
  - Speed bonus:           faster stabilisation = higher score
  - Treatment quality:     correct drug class selection rewarded
  - Organ protection:      avoiding creatinine rise in hard task
"""

from __future__ import annotations
import math
from typing import List, Optional

from .models import GraderResult, PatientVitals


def _lerp_score(value: float, worst: float, best: float) -> float:
    """Map a value linearly to [0, 1]. Clips to [0, 1]."""
    if best == worst:
        return 0.5
    s = (value - worst) / (best - worst)
    return max(0.0, min(1.0, s))


def grade_mild_sepsis(
    trajectory: List[PatientVitals],
    alive: bool,
    max_steps: int,
    stabilized_at: Optional[int],
) -> GraderResult:
    """
    Easy task grader — mild sepsis.

    Component weights:
      40%  Survival
      25%  Final MAP (goal ≥ 65)
      20%  Final lactate (goal < 2.0)
      10%  Final WBC (goal 4-12)
       5%  Speed bonus
    """
    if not alive or not trajectory:
        return GraderResult(
            score=0.0,
            reason="Patient died — episode score is 0.",
            metrics={"alive": 0.0},
            passed=False,
        )

    final = trajectory[-1]
    metrics: dict = {}

    # Survival (prerequisite)
    metrics["alive"] = 1.0

    # MAP score
    map_score = _lerp_score(final.map_mmhg, 40.0, 85.0)
    metrics["map_score"] = round(map_score, 3)

    # Lactate score (lower is better; reverse scale)
    lactate_score = _lerp_score(final.lactate, 6.0, 1.0)
    metrics["lactate_score"] = round(lactate_score, 3)

    # WBC score
    wbc_target = abs(final.wbc - 8.0)          # distance from midpoint of normal
    wbc_score = _lerp_score(wbc_target, 10.0, 0.0)
    metrics["wbc_score"] = round(wbc_score, 3)

    # Speed bonus
    speed_bonus = 0.0
    if stabilized_at is not None:
        speed_bonus = 1.0 - (stabilized_at / max_steps)
    metrics["speed_bonus"] = round(speed_bonus, 3)

    score = (
        0.40 * 1.0           # survival
        + 0.25 * map_score
        + 0.20 * lactate_score
        + 0.10 * wbc_score
        + 0.05 * speed_bonus
    )
    score = round(max(0.0, min(1.0, score)), 4)

    reason = (
        f"Patient alive. MAP={final.map_mmhg:.1f}, "
        f"Lactate={final.lactate:.2f}, WBC={final.wbc:.1f}. "
        f"Speed bonus={speed_bonus:.2f}"
    )

    return GraderResult(score=score, reason=reason, metrics=metrics, passed=score >= 0.5)


def grade_septic_shock(
    trajectory: List[PatientVitals],
    alive: bool,
    max_steps: int,
    stabilized_at: Optional[int],
    used_narrow_ab: bool,
    used_vasopressor: bool,
) -> GraderResult:
    """
    Medium task grader — septic shock.

    Additional components vs easy:
      15%  Used narrow-spectrum antibiotics (correct for gram-positive)
      15%  Used vasopressor (mandatory for MAP < 65)
      (Vital/speed weights adjusted accordingly)
    """
    if not alive or not trajectory:
        return GraderResult(
            score=0.0,
            reason="Patient died — episode score is 0.",
            metrics={"alive": 0.0},
            passed=False,
        )

    final = trajectory[-1]
    metrics: dict = {}

    metrics["alive"] = 1.0
    map_score      = _lerp_score(final.map_mmhg, 35.0, 80.0)
    lactate_score  = _lerp_score(final.lactate, 10.0, 1.0)
    wbc_score      = _lerp_score(abs(final.wbc - 8.0), 15.0, 0.0)
    creatinine_score = _lerp_score(final.creatinine, 4.0, 0.6)

    narrow_ab_score  = 1.0 if used_narrow_ab else 0.0
    vasopressor_score = 1.0 if used_vasopressor else 0.0

    speed_bonus = 0.0
    if stabilized_at is not None:
        speed_bonus = 1.0 - (stabilized_at / max_steps)

    metrics.update({
        "map_score": round(map_score, 3),
        "lactate_score": round(lactate_score, 3),
        "wbc_score": round(wbc_score, 3),
        "creatinine_score": round(creatinine_score, 3),
        "correct_antibiotic": narrow_ab_score,
        "used_vasopressor": vasopressor_score,
        "speed_bonus": round(speed_bonus, 3),
    })

    score = (
        0.30 * 1.0              # survival
        + 0.20 * map_score
        + 0.15 * lactate_score
        + 0.05 * creatinine_score
        + 0.05 * wbc_score
        + 0.15 * narrow_ab_score
        + 0.05 * vasopressor_score
        + 0.05 * speed_bonus
    )
    score = round(max(0.0, min(1.0, score)), 4)

    reason = (
        f"Patient alive. MAP={final.map_mmhg:.1f}, "
        f"Lactate={final.lactate:.2f}. "
        f"Correct AB={'yes' if used_narrow_ab else 'no'}, "
        f"Vasopressor={'yes' if used_vasopressor else 'no'}."
    )

    return GraderResult(score=score, reason=reason, metrics=metrics, passed=score >= 0.5)


def grade_severe_mods(
    trajectory: List[PatientVitals],
    alive: bool,
    max_steps: int,
    stabilized_at: Optional[int],
    used_broad_first: bool,
    switched_to_narrow: bool,
    peak_resistance: float,
    min_vasopressor_dose: str,   # "none" | "low" | "high"
) -> GraderResult:
    """
    Hard task grader — severe MODS.

    Optimal strategy:
      1. Use broad-spectrum antibiotics first 2 steps
      2. Switch to narrow-spectrum (reduces resistance accumulation)
      3. Maintain MAP with lowest effective vasopressor
      4. Minimise creatinine rise (renal protection)

    Component weights:
      25%  Survival
      20%  Final MAP + lactate combo
      15%  Antibiotic sequencing (broad first → narrow)
      15%  Resistance kept low (< 0.3 = full credit)
      15%  Renal protection (creatinine delta)
      10%  Speed / efficiency
    """
    if not alive or not trajectory:
        return GraderResult(
            score=0.0,
            reason="Patient died — episode score is 0.",
            metrics={"alive": 0.0},
            passed=False,
        )

    final    = trajectory[-1]
    initial  = trajectory[0]
    metrics: dict = {}

    metrics["alive"] = 1.0

    # Vital improvement scores
    map_score     = _lerp_score(final.map_mmhg, 30.0, 75.0)
    lactate_score = _lerp_score(final.lactate, 15.0, 2.0)
    vital_combo   = 0.5 * map_score + 0.5 * lactate_score

    # Antibiotic sequencing
    seq_score = 0.0
    if used_broad_first:
        seq_score += 0.5
    if switched_to_narrow:
        seq_score += 0.5
    metrics["sequencing_score"] = round(seq_score, 3)

    # Resistance management
    resistance_score = _lerp_score(peak_resistance, 1.0, 0.0)
    metrics["resistance_score"] = round(resistance_score, 3)

    # Renal protection: creatinine should not rise too much
    creatinine_rise = final.creatinine - initial.creatinine
    renal_score = _lerp_score(creatinine_rise, 5.0, 0.0)
    metrics["renal_score"] = round(renal_score, 3)

    # Vasopressor efficiency
    vp_efficiency = {"none": 1.0, "low": 0.9, "high": 0.6}.get(min_vasopressor_dose, 0.5)
    metrics["vp_efficiency"] = round(vp_efficiency, 3)

    speed_bonus = 0.0
    if stabilized_at is not None:
        speed_bonus = 1.0 - (stabilized_at / max_steps)

    metrics.update({
        "map_score": round(map_score, 3),
        "lactate_score": round(lactate_score, 3),
        "speed_bonus": round(speed_bonus, 3),
    })

    score = (
        0.25 * 1.0              # survival
        + 0.20 * vital_combo
        + 0.15 * seq_score
        + 0.15 * resistance_score
        + 0.15 * renal_score
        + 0.10 * speed_bonus
    )
    score = round(max(0.0, min(1.0, score)), 4)

    reason = (
        f"Patient alive. MAP={final.map_mmhg:.1f}, "
        f"Lactate={final.lactate:.2f}, Creatinine={final.creatinine:.2f}. "
        f"Sequence={'correct' if seq_score >= 1.0 else 'suboptimal'}, "
        f"Peak resistance={peak_resistance:.2f}."
    )

    return GraderResult(score=score, reason=reason, metrics=metrics, passed=score >= 0.5)
