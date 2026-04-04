"""
SepsisPilot — OpenEnv Environment
Implements: reset() / step() / state() / grade()
This class is the single source of truth for episode state.
"""

from __future__ import annotations
from typing import Optional, List

from .models import (
    Action, PatientState, PatientVitals, StepResult, GraderResult, TaskInfo, ResetRequest,
)
from .patient_sim import PatientSimulator, TASK_PROFILES
from .graders import grade_mild_sepsis, grade_septic_shock, grade_severe_mods


AVAILABLE_TASKS = list(TASK_PROFILES.keys())


class SepsisPilotEnv:
    """
    OpenEnv-compliant environment for sepsis treatment sequencing.

    Usage:
        env = SepsisPilotEnv()
        state = env.reset("mild_sepsis")
        while not state.done:
            result = env.step(action_int)
            state = result.state
        grade = env.grade()
    """

    def __init__(self):
        self._sim:          Optional[PatientSimulator] = None
        self._task:         Optional[str] = None
        self._step_count:   int = 0
        self._alive:        bool = True
        self._done:         bool = False
        self._episode_reward: float = 0.0
        self._stabilized_at: Optional[int] = None
        self._trajectory:   List[PatientVitals] = []
        self._current_vitals: Optional[PatientVitals] = None

        # Grader tracking metadata
        self._used_narrow_ab:   bool = False
        self._used_vasopressor: bool = False
        self._used_broad_first: bool = False
        self._switched_to_narrow: bool = False
        self._peak_resistance:  float = 0.0
        self._min_vp_dose:      str = "none"
        self._first_ab_step:    Optional[int] = None
        self._narrow_after_broad: bool = False

    # ─────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────

    def reset(self, task: str = "mild_sepsis", seed: Optional[int] = None) -> PatientState:
        """Reset environment to start a new episode."""
        if task not in TASK_PROFILES:
            raise ValueError(f"Unknown task '{task}'. Available: {AVAILABLE_TASKS}")

        profile = TASK_PROFILES[task]
        self._sim = PatientSimulator(profile, seed=seed)
        self._task = task
        self._step_count = 0
        self._alive = True
        self._done = False
        self._episode_reward = 0.0
        self._stabilized_at = None
        self._trajectory = []
        self._current_vitals = self._sim.reset(seed=seed)
        self._trajectory.append(self._current_vitals)

        # Reset grader metadata
        self._used_narrow_ab    = False
        self._used_vasopressor  = False
        self._used_broad_first  = False
        self._switched_to_narrow = False
        self._peak_resistance   = self._current_vitals.resistance
        self._min_vp_dose       = "none"
        self._first_ab_step     = None
        self._narrow_after_broad = False

        return self._make_state()

    def step(self, action: int) -> StepResult:
        """Apply action, advance one timestep, return result."""
        if self._sim is None or self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")
        if not (0 <= action <= 8):
            raise ValueError(f"Invalid action {action}. Must be 0-8.")

        profile = TASK_PROFILES[self._task]
        self._step_count += 1

        # Track grader metadata before sim step
        self._update_grader_metadata(action)

        # Advance simulation
        vitals, reward, sim_done, info = self._sim.step(action)
        self._current_vitals = vitals
        self._trajectory.append(vitals)
        self._episode_reward += reward

        # Determine episode termination
        self._alive = not vitals.is_dead()
        if vitals.is_stable() and self._stabilized_at is None:
            self._stabilized_at = self._step_count

        self._done = (
            sim_done
            or self._step_count >= profile.max_steps
        )

        # Update resistance peak
        self._peak_resistance = max(self._peak_resistance, vitals.resistance)

        state = self._make_state()
        return StepResult(state=state, reward=reward, done=self._done, info=info)

    def state(self) -> PatientState:
        """Return current state without advancing the simulation."""
        if self._sim is None:
            raise RuntimeError("Call reset() first.")
        return self._make_state()

    def grade(self) -> GraderResult:
        """Grade the completed episode. Returns score in [0.0, 1.0]."""
        if not self._done:
            raise RuntimeError("Episode not done yet. Cannot grade.")

        profile = TASK_PROFILES[self._task]

        if self._task == "mild_sepsis":
            return grade_mild_sepsis(
                trajectory=self._trajectory,
                alive=self._alive,
                max_steps=profile.max_steps,
                stabilized_at=self._stabilized_at,
            )
        elif self._task == "septic_shock":
            return grade_septic_shock(
                trajectory=self._trajectory,
                alive=self._alive,
                max_steps=profile.max_steps,
                stabilized_at=self._stabilized_at,
                used_narrow_ab=self._used_narrow_ab,
                used_vasopressor=self._used_vasopressor,
            )
        elif self._task == "severe_mods":
            return grade_severe_mods(
                trajectory=self._trajectory,
                alive=self._alive,
                max_steps=profile.max_steps,
                stabilized_at=self._stabilized_at,
                used_broad_first=self._used_broad_first,
                switched_to_narrow=self._switched_to_narrow,
                peak_resistance=self._peak_resistance,
                min_vasopressor_dose=self._min_vp_dose,
            )
        else:
            raise ValueError(f"No grader for task '{self._task}'")

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _make_state(self) -> PatientState:
        profile = TASK_PROFILES[self._task]
        return PatientState(
            vitals=self._current_vitals,
            step=self._step_count,
            max_steps=profile.max_steps,
            done=self._done,
            alive=self._alive,
            task=self._task,
            stabilized_at=self._stabilized_at,
            episode_reward=round(self._episode_reward, 4),
        )

    def _update_grader_metadata(self, action: int):
        has_broad   = action in (1, 5, 6)
        has_narrow  = action in (2, 7, 8)
        has_low_vp  = action in (3, 5, 7)
        has_high_vp = action in (4, 6, 8)

        if has_narrow:
            self._used_narrow_ab = True
        if has_low_vp or has_high_vp:
            self._used_vasopressor = True

        # Vasopressor dose tracking (prefer lowest dose used)
        if has_low_vp and self._min_vp_dose == "none":
            self._min_vp_dose = "low"
        if has_high_vp:
            self._min_vp_dose = "high" if self._min_vp_dose == "none" else self._min_vp_dose

        # Antibiotic sequencing (broad → narrow is optimal for severe MODS)
        if has_broad and self._first_ab_step is None:
            self._first_ab_step = self._step_count
            self._used_broad_first = True
        if has_narrow and self._used_broad_first and not self._switched_to_narrow:
            self._switched_to_narrow = True

    @staticmethod
    def task_list() -> List[TaskInfo]:
        from .patient_sim import TASK_PROFILES
        return [
            TaskInfo(
                name=p.name,
                difficulty=p.difficulty,
                description=p.description,
                max_steps=p.max_steps,
            )
            for p in TASK_PROFILES.values()
        ]
