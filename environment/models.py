"""
SepsisPilot — Typed Models (OpenEnv Spec)
All state, action, step, and grader contracts live here.
"""

from __future__ import annotations
from enum import IntEnum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Action Space  (discrete, 9 actions)
# ──────────────────────────────────────────────

class Action(IntEnum):
    NO_TREATMENT       = 0   # watchful waiting
    BROAD_ANTIBIOTICS  = 1   # e.g. piperacillin-tazobactam (gram-negative coverage)
    NARROW_ANTIBIOTICS = 2   # e.g. vancomycin (gram-positive coverage)
    LOW_VASOPRESSOR    = 3   # norepinephrine 0.1 mcg/kg/min
    HIGH_VASOPRESSOR   = 4   # norepinephrine 0.3 mcg/kg/min
    BROAD_LOW_VASO     = 5   # broad AB + low-dose vasopressor
    BROAD_HIGH_VASO    = 6   # broad AB + high-dose vasopressor
    NARROW_LOW_VASO    = 7   # narrow AB + low-dose vasopressor
    NARROW_HIGH_VASO   = 8   # narrow AB + high-dose vasopressor

ACTION_DESCRIPTIONS: Dict[int, str] = {
    0: "No treatment — watchful waiting",
    1: "Broad-spectrum antibiotics (piperacillin-tazobactam)",
    2: "Narrow-spectrum antibiotics (vancomycin)",
    3: "Low-dose vasopressor (norepinephrine 0.1 mcg/kg/min)",
    4: "High-dose vasopressor (norepinephrine 0.3 mcg/kg/min)",
    5: "Broad-spectrum antibiotics + low-dose vasopressor",
    6: "Broad-spectrum antibiotics + high-dose vasopressor",
    7: "Narrow-spectrum antibiotics + low-dose vasopressor",
    8: "Narrow-spectrum antibiotics + high-dose vasopressor",
}

# ──────────────────────────────────────────────
# Patient State  (observation space, shape=[8])
# ──────────────────────────────────────────────

class PatientVitals(BaseModel):
    """Continuous observation vector. Normal ranges noted inline."""
    map_mmhg:    float = Field(..., description="Mean Arterial Pressure mmHg. Normal 70-100; sepsis goal >65")
    lactate:     float = Field(..., description="Serum lactate mmol/L. Normal 0.5-2.0; crisis >4")
    wbc:         float = Field(..., description="White blood cell count k/uL. Normal 4-11; sepsis >12 or <4")
    temperature: float = Field(..., description="Core temp °C. Normal 36.5-37.5; sepsis >38 or <36")
    heart_rate:  float = Field(..., description="Heart rate bpm. Normal 60-100; sepsis >90")
    creatinine:  float = Field(..., description="Serum creatinine mg/dL. Normal 0.6-1.2; AKI >1.5")
    sofa_score:  float = Field(..., description="SOFA score 0-24. >10 = high mortality")
    resistance:  float = Field(..., description="Antibiotic resistance index 0-1 (hard task only)")

    def to_list(self) -> List[float]:
        return [
            self.map_mmhg, self.lactate, self.wbc, self.temperature,
            self.heart_rate, self.creatinine, self.sofa_score, self.resistance,
        ]

    def is_stable(self) -> bool:
        """All key vitals in target range."""
        return (
            self.map_mmhg >= 65
            and self.lactate <= 2.0
            and 4.0 <= self.wbc <= 12.0
            and 36.0 <= self.temperature <= 38.0
            and self.heart_rate <= 100
        )

    def is_dead(self) -> bool:
        return (
            self.map_mmhg < 35
            or self.lactate > 15
            or self.heart_rate > 165
            or self.heart_rate < 25
        )


class PatientState(BaseModel):
    """Full state exposed to the agent."""
    vitals:         PatientVitals
    step:           int
    max_steps:      int
    done:           bool
    alive:          bool
    task:           str
    stabilized_at:  Optional[int] = None   # step when vitals first became stable
    episode_reward: float = 0.0

    def to_observation(self) -> List[float]:
        """Flat numeric vector for RL agents."""
        return self.vitals.to_list() + [self.step / self.max_steps]


# ──────────────────────────────────────────────
# API Request / Response models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field("mild_sepsis", description="Task name: mild_sepsis | septic_shock | severe_mods")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class ActionRequest(BaseModel):
    action: int = Field(..., ge=0, le=8, description="Action index 0-8")

class StepResult(BaseModel):
    state:  PatientState
    reward: float
    done:   bool
    info:   Dict[str, Any]

class GraderResult(BaseModel):
    score:   float = Field(..., ge=0.0, le=1.0)
    reason:  str
    metrics: Dict[str, float]
    passed:  bool   # score >= 0.5

class TaskInfo(BaseModel):
    name:        str
    difficulty:  str
    description: str
    max_steps:   int
    action_n:    int = 9
    obs_shape:   List[int] = [9]
