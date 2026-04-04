"""
SepsisPilot — Patient Physiology Simulator
Simulates realistic sepsis progression and treatment response.
Each timestep = 1 hour of ICU treatment.

Infection types:
  gram_negative → broad-spectrum antibiotics more effective
  gram_positive → narrow-spectrum antibiotics more effective
  mixed_resistant → requires precise sequencing; wrong ABs build resistance
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .models import Action, PatientVitals


# ──────────────────────────────────────────────
# Task profiles
# ──────────────────────────────────────────────

@dataclass
class TaskProfile:
    name:            str
    difficulty:      str
    infection_type:  str          # gram_negative | gram_positive | mixed_resistant
    max_steps:       int
    noise_scale:     float        # vitals noise multiplier
    decay_rate:      float        # disease progression speed (higher = faster deterioration)
    resistance_mode: bool         # track antibiotic resistance
    description:     str

    # Initial vitals
    init_map:         float = 65.0
    init_lactate:     float = 2.5
    init_wbc:         float = 14.0
    init_temperature: float = 38.2
    init_heart_rate:  float = 95.0
    init_creatinine:  float = 0.9
    init_resistance:  float = 0.0


TASK_PROFILES: Dict[str, TaskProfile] = {
    "mild_sepsis": TaskProfile(
        name="mild_sepsis",
        difficulty="easy",
        infection_type="gram_negative",
        max_steps=24,
        noise_scale=0.3,
        decay_rate=0.6,
        resistance_mode=False,
        description=(
            "Mild sepsis secondary to gram-negative UTI. Patient is haemodynamically "
            "marginal but responsive. Broad-spectrum antibiotics are the cornerstone "
            "therapy; vasopressors needed only if MAP falls below 65."
        ),
        init_map=65.0, init_lactate=2.5, init_wbc=14.0,
        init_temperature=38.2, init_heart_rate=95.0, init_creatinine=0.9,
    ),
    "septic_shock": TaskProfile(
        name="septic_shock",
        difficulty="medium",
        infection_type="gram_positive",
        max_steps=48,
        noise_scale=0.5,
        decay_rate=1.2,
        resistance_mode=False,
        description=(
            "Septic shock secondary to gram-positive bacteraemia (MRSA suspected). "
            "MAP critically low; vasopressors mandatory immediately. Narrow-spectrum "
            "antibiotics (vancomycin) are optimal. Broad-spectrum provides partial "
            "coverage but is suboptimal. Time is critical — each hour of delayed "
            "source control increases mortality."
        ),
        init_map=52.0, init_lactate=4.2, init_wbc=18.0,
        init_temperature=38.9, init_heart_rate=115.0, init_creatinine=1.4,
    ),
    "severe_mods": TaskProfile(
        name="severe_mods",
        difficulty="hard",
        infection_type="mixed_resistant",
        max_steps=72,
        noise_scale=0.8,
        decay_rate=2.0,
        resistance_mode=True,
        description=(
            "Severe sepsis with Multi-Organ Dysfunction Syndrome (MODS). Mixed "
            "drug-resistant infection — early broad-spectrum coverage is needed for "
            "gram-negative load, but repeated use accelerates resistance. Narrow-spectrum "
            "must follow for gram-positive coverage. High-dose vasopressors risk acute "
            "kidney injury (creatinine rises). Optimal sequencing: broad AB first "
            "2 steps → narrow AB → maintain MAP > 65 with lowest effective vasopressor."
        ),
        init_map=42.0, init_lactate=7.0, init_wbc=22.0,
        init_temperature=39.6, init_heart_rate=128.0, init_creatinine=2.2,
        init_resistance=0.1,
    ),
}


# ──────────────────────────────────────────────
# Physiology engine
# ──────────────────────────────────────────────

class PatientSimulator:
    """
    Discrete-time physiology model. Call step(action) each hour.
    All effects are additive deltas applied in order:
      1. Natural disease progression  (always worsens without treatment)
      2. Treatment effects            (improve vitals if correct)
      3. Side effects                 (high-dose vasopressor → kidney stress)
      4. Gaussian noise               (stochastic variation)
      5. Physiological coupling       (HR tracks MAP)
    """

    def __init__(self, profile: TaskProfile, seed: Optional[int] = None):
        self.profile = profile
        self.rng = random.Random(seed)
        self._reset_vitals()

    # ── public ──────────────────────────────

    def reset(self, seed: Optional[int] = None) -> PatientVitals:
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_vitals()
        return self._snapshot()

    def step(self, action: int) -> Tuple[PatientVitals, float, bool, dict]:
        """
        Apply one hour of treatment.
        Returns (next_vitals, reward, done, info).
        """
        prev = self._snapshot()

        self._apply_disease_progression()
        self._apply_treatment(action)
        self._apply_physiological_coupling()
        self._apply_noise()
        self._clamp_vitals()

        curr = self._snapshot()
        done = curr.is_dead() or curr.is_stable()
        reward = self._compute_reward(prev, curr, action, done)
        info = self._build_info(action, prev, curr)

        return curr, reward, done, info

    # ── private — vitals state ──────────────

    def _reset_vitals(self):
        p = self.profile
        self.map        = p.init_map
        self.lactate    = p.init_lactate
        self.wbc        = p.init_wbc
        self.temperature = p.init_temperature
        self.heart_rate = p.init_heart_rate
        self.creatinine = p.init_creatinine
        self.resistance = p.init_resistance
        self._consecutive_low_map = 0
        self._ab_history: list[int] = []  # last 4 steps antibiotic actions

    def _snapshot(self) -> PatientVitals:
        return PatientVitals(
            map_mmhg=round(self.map, 2),
            lactate=round(self.lactate, 3),
            wbc=round(self.wbc, 2),
            temperature=round(self.temperature, 2),
            heart_rate=round(self.heart_rate, 1),
            creatinine=round(self.creatinine, 3),
            sofa_score=round(self._compute_sofa(), 2),
            resistance=round(self.resistance, 3),
        )

    # ── private — physiology ──────────────

    def _apply_disease_progression(self):
        """Natural deterioration each hour."""
        d = self.profile.decay_rate
        self.map        -= 0.8 * d
        self.lactate    += 0.12 * d
        self.wbc        += 0.4 * d
        self.temperature += 0.04 * d
        self.heart_rate += 1.2 * d
        self.creatinine += 0.015 * d

    def _apply_treatment(self, action: int):
        """Treatment effects depend on infection type matching."""
        infection = self.profile.infection_type
        has_broad   = action in (1, 5, 6)
        has_narrow  = action in (2, 7, 8)
        has_low_vp  = action in (3, 5, 7)
        has_high_vp = action in (4, 6, 8)

        # ── Antibiotic effect ──────────────────
        # gram_negative → broad is optimal (efficiency=1.0), narrow suboptimal (0.3)
        # gram_positive → narrow is optimal (1.0), broad suboptimal (0.3)
        # mixed_resistant → both needed; wrong/repeated use builds resistance

        ab_efficiency = 0.0
        if has_broad or has_narrow:
            if infection == "gram_negative":
                ab_efficiency = 1.0 if has_broad else 0.3
            elif infection == "gram_positive":
                ab_efficiency = 1.0 if has_narrow else 0.3
            else:  # mixed_resistant
                # Track resistance: repeated same antibiotic is less effective
                last = self._last_ab_type()
                if has_broad and last != "broad":
                    ab_efficiency = 0.8
                    self._ab_history.append(1)
                elif has_narrow and last != "narrow":
                    ab_efficiency = 0.7
                    self._ab_history.append(2)
                elif has_broad:   # repeated broad
                    ab_efficiency = max(0.1, 0.8 - self.resistance)
                    self.resistance = min(1.0, self.resistance + 0.08)
                    self._ab_history.append(1)
                elif has_narrow:  # repeated narrow
                    ab_efficiency = max(0.1, 0.7 - self.resistance * 0.5)
                    self._ab_history.append(2)
                if len(self._ab_history) > 6:
                    self._ab_history.pop(0)

            eff = ab_efficiency
            self.wbc         -= 0.6 * eff
            self.temperature -= 0.07 * eff
            # Lactate improves only when MAP adequate + antibiotics active
            if self.map >= 60:
                self.lactate -= 0.15 * eff

        # ── Vasopressor effect ──────────────────
        if has_low_vp:
            self.map        += 5.0
            self.heart_rate -= 3.0
        if has_high_vp:
            self.map        += 9.0
            self.heart_rate -= 5.0
            self.creatinine += 0.04   # renal vasoconstriction side effect

        # Lactate also falls as perfusion improves with vasopressors
        if has_low_vp or has_high_vp:
            if self.map >= 65:
                self.lactate -= 0.08

    def _apply_physiological_coupling(self):
        """HR and lactate track MAP (compensatory responses)."""
        # Compensatory tachycardia when MAP drops
        if self.map < 65:
            deficit = 65 - self.map
            self.heart_rate += 0.4 * deficit * 0.1
        # Lactate worsens severely when MAP very low (anaerobic metabolism)
        if self.map < 50:
            self.lactate += 0.25
        # Creatinine rises with prolonged low MAP (AKI)
        if self.map < 55:
            self.creatinine += 0.02

    def _apply_noise(self):
        """Gaussian noise to mimic measurement + biological variability."""
        s = self.profile.noise_scale
        n = self.rng.gauss
        self.map         += n(0, 1.5 * s)
        self.lactate     += n(0, 0.05 * s)
        self.wbc         += n(0, 0.2 * s)
        self.temperature += n(0, 0.05 * s)
        self.heart_rate  += n(0, 1.5 * s)
        self.creatinine  += n(0, 0.02 * s)

    def _clamp_vitals(self):
        """Hard physiological limits."""
        self.map         = max(20.0,  min(160.0, self.map))
        self.lactate     = max(0.1,   min(20.0,  self.lactate))
        self.wbc         = max(0.5,   min(40.0,  self.wbc))
        self.temperature = max(33.0,  min(42.0,  self.temperature))
        self.heart_rate  = max(20.0,  min(170.0, self.heart_rate))
        self.creatinine  = max(0.3,   min(12.0,  self.creatinine))
        self.resistance  = max(0.0,   min(1.0,   self.resistance))

    def _compute_sofa(self) -> float:
        """Simplified SOFA score (0-24). Captures multi-organ function."""
        score = 0.0
        # Cardiovascular (MAP)
        if self.map < 70:   score += 1
        if self.map < 65:   score += 1
        if self.map < 55:   score += 2
        # Renal (creatinine)
        if self.creatinine > 1.2:  score += 1
        if self.creatinine > 2.0:  score += 2
        if self.creatinine > 3.5:  score += 3
        # Metabolic (lactate proxy)
        if self.lactate > 2.0:  score += 1
        if self.lactate > 4.0:  score += 2
        if self.lactate > 8.0:  score += 3
        return min(24.0, score)

    # ── private — reward ──────────────────

    def _compute_reward(
        self,
        prev: PatientVitals,
        curr: PatientVitals,
        action: int,
        done: bool,
    ) -> float:
        """
        Dense reward shaping:
        - Per-vital improvement signals throughout trajectory
        - Stabilization bonus for achieving all-normal state
        - Death penalty
        - Small time penalty to encourage efficiency
        """
        r = 0.0

        # MAP improvement (most critical)
        if curr.map_mmhg >= 65:
            r += 0.35
        elif curr.map_mmhg > prev.map_mmhg:
            r += 0.15
        else:
            r -= 0.20

        # Lactate clearance (mortality signal)
        if curr.lactate <= 2.0:
            r += 0.30
        elif curr.lactate < prev.lactate:
            r += 0.12
        else:
            r -= 0.18

        # WBC normalisation (infection control)
        if 4.0 <= curr.wbc <= 12.0:
            r += 0.10
        elif curr.wbc < prev.wbc:
            r += 0.04
        else:
            r -= 0.05

        # Temperature control
        if 36.0 <= curr.temperature <= 38.0:
            r += 0.08
        elif curr.temperature < prev.temperature:
            r += 0.03

        # Creatinine protection (renal function)
        if curr.creatinine < prev.creatinine:
            r += 0.05
        elif curr.creatinine > prev.creatinine + 0.1:
            r -= 0.08   # rapid kidney deterioration

        # Resistance penalty (hard task)
        if self.profile.resistance_mode and curr.resistance > prev.resistance:
            r -= 0.15 * (curr.resistance - prev.resistance) * 10

        # Per-step cost (encourages speed)
        r -= 0.025

        # Terminal rewards
        if done:
            if curr.is_dead():
                r -= 8.0
            elif curr.is_stable():
                r += 5.0

        return round(r, 4)

    def _build_info(self, action: int, prev: PatientVitals, curr: PatientVitals) -> dict:
        from .models import ACTION_DESCRIPTIONS
        return {
            "action_taken": ACTION_DESCRIPTIONS.get(action, "unknown"),
            "infection_type": self.profile.infection_type,
            "map_delta": round(curr.map_mmhg - prev.map_mmhg, 2),
            "lactate_delta": round(curr.lactate - prev.lactate, 3),
            "wbc_delta": round(curr.wbc - prev.wbc, 2),
            "sofa_score": curr.sofa_score,
            "resistance": curr.resistance,
            "stable": curr.is_stable(),
            "alive": not curr.is_dead(),
        }

    def _last_ab_type(self) -> Optional[str]:
        if not self._ab_history:
            return None
        return "broad" if self._ab_history[-1] == 1 else "narrow"
