"""
SepsisPilot — Unit Tests
Run: pytest tests/ -v
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from environment import SepsisPilotEnv, AVAILABLE_TASKS
from environment.models import PatientVitals


class TestPatientVitals:
    def test_is_stable_when_all_vitals_normal(self):
        v = PatientVitals(
            map_mmhg=72, lactate=1.5, wbc=8.0, temperature=37.0,
            heart_rate=80, creatinine=1.0, sofa_score=0.0, resistance=0.0,
        )
        assert v.is_stable() is True

    def test_is_not_stable_when_map_low(self):
        v = PatientVitals(
            map_mmhg=60, lactate=1.5, wbc=8.0, temperature=37.0,
            heart_rate=80, creatinine=1.0, sofa_score=2.0, resistance=0.0,
        )
        assert v.is_stable() is False

    def test_is_dead_when_map_critical(self):
        v = PatientVitals(
            map_mmhg=30, lactate=3.0, wbc=10.0, temperature=37.0,
            heart_rate=90, creatinine=1.2, sofa_score=10.0, resistance=0.0,
        )
        assert v.is_dead() is True

    def test_is_dead_when_lactate_extreme(self):
        v = PatientVitals(
            map_mmhg=70, lactate=16.0, wbc=10.0, temperature=37.0,
            heart_rate=90, creatinine=1.2, sofa_score=10.0, resistance=0.0,
        )
        assert v.is_dead() is True

    def test_to_list_returns_8_elements(self):
        v = PatientVitals(
            map_mmhg=70, lactate=2.0, wbc=10.0, temperature=37.0,
            heart_rate=80, creatinine=1.0, sofa_score=2.0, resistance=0.0,
        )
        assert len(v.to_list()) == 8


class TestSepsisPilotEnv:
    def setup_method(self):
        self.env = SepsisPilotEnv()

    def test_available_tasks(self):
        assert set(AVAILABLE_TASKS) == {"mild_sepsis", "septic_shock", "severe_mods"}

    def test_reset_returns_valid_state(self):
        state = self.env.reset("mild_sepsis", seed=42)
        assert state.step == 0
        assert state.done is False
        assert state.alive is True
        assert state.task == "mild_sepsis"
        assert state.vitals is not None

    @pytest.mark.parametrize("task", ["mild_sepsis", "septic_shock", "severe_mods"])
    def test_reset_all_tasks(self, task):
        state = self.env.reset(task, seed=42)
        assert state.task == task
        assert state.step == 0

    def test_step_increments_counter(self):
        self.env.reset("mild_sepsis", seed=42)
        result = self.env.step(5)
        assert result.state.step == 1

    def test_step_returns_reward_and_done(self):
        self.env.reset("mild_sepsis", seed=42)
        result = self.env.step(1)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_invalid_action_raises(self):
        self.env.reset("mild_sepsis", seed=42)
        with pytest.raises(ValueError):
            self.env.step(99)

    def test_step_before_reset_raises(self):
        fresh_env = SepsisPilotEnv()
        with pytest.raises(RuntimeError):
            fresh_env.step(0)

    def test_grade_before_done_raises(self):
        self.env.reset("mild_sepsis", seed=42)
        with pytest.raises(RuntimeError):
            self.env.grade()

    def test_full_episode_mild_sepsis(self):
        """Complete a full mild_sepsis episode and verify grade."""
        state = self.env.reset("mild_sepsis", seed=42)
        steps = 0
        while not state.done:
            result = self.env.step(5)  # broad AB + low vaso
            state = result.state
            steps += 1
            assert steps <= 30, "Episode exceeded max_steps guard"

        grade = self.env.grade()
        assert 0.0 <= grade.score <= 1.0
        assert isinstance(grade.reason, str)
        assert isinstance(grade.passed, bool)

    def test_no_treatment_worsens_vitals(self):
        """No treatment should worsen MAP and lactate over time."""
        state = self.env.reset("mild_sepsis", seed=42)
        initial_map = state.vitals.map_mmhg
        initial_lactate = state.vitals.lactate

        for _ in range(5):
            if state.done:
                break
            result = self.env.step(0)  # no treatment
            state = result.state

        # MAP should trend down, lactate should trend up
        assert state.vitals.map_mmhg <= initial_map + 5   # allow noise
        assert state.vitals.lactate >= initial_lactate - 0.5

    def test_reproducibility(self):
        """Same seed produces identical trajectories."""
        def run(seed):
            env = SepsisPilotEnv()
            env.reset("septic_shock", seed=seed)
            rewards = []
            for _ in range(6):
                r = env.step(7)
                rewards.append(r.reward)
                if r.done:
                    break
            return rewards

        assert run(42) == run(42)

    def test_grader_score_varies_with_strategy(self):
        """Good strategy should score higher than bad strategy."""

        def episode(task, actions_cycle, seed=42):
            env = SepsisPilotEnv()
            state = env.reset(task, seed=seed)
            i = 0
            while not state.done:
                action = actions_cycle[i % len(actions_cycle)]
                result = env.step(action)
                state = result.state
                i += 1
            return env.grade().score

        good = episode("mild_sepsis", [5, 1, 1, 5])   # broad AB + low vaso
        bad  = episode("mild_sepsis", [0, 0, 0, 0])   # no treatment

        assert good > bad, f"Expected good > bad, got {good:.3f} <= {bad:.3f}"

    def test_task_list(self):
        tasks = SepsisPilotEnv.task_list()
        assert len(tasks) == 3
        names = {t.name for t in tasks}
        assert names == {"mild_sepsis", "septic_shock", "severe_mods"}
        for t in tasks:
            assert t.difficulty in ("easy", "medium", "hard")
            assert t.max_steps > 0
