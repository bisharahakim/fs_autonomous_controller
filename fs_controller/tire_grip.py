from __future__ import annotations

from math import sqrt

import numpy as np


N_TABLE = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], dtype=float)
F_TABLE = np.array([700, 1300, 1900, 2300, 2700, 3000, 3200, 3300], dtype=float)
_POLY_COEFFS = np.polyfit(N_TABLE, F_TABLE, 2)


def max_tire_force(normal_load_n: float) -> float:
    """Maximum combined tire force available at normal load N."""

    if normal_load_n <= 0.0:
        return 0.0
    load = min(normal_load_n, float(N_TABLE[-1]))
    return max(0.0, float(np.polyval(_POLY_COEFFS, load)))


def total_grip_force_equal_load(mass_kg: float, g_mps2: float = 9.81) -> float:
    normal_load_per_tire = mass_kg * g_mps2 / 4.0
    return 4.0 * max_tire_force(normal_load_per_tire)


def max_lateral_accel_equal_load(mass_kg: float, g_mps2: float = 9.81) -> float:
    return total_grip_force_equal_load(mass_kg, g_mps2) / mass_kg


def apply_friction_circle(
    longitudinal_force_n: float,
    lateral_force_n: float,
    mass_kg: float,
    g_mps2: float = 9.81,
) -> tuple[float, float, float]:
    total_grip = total_grip_force_equal_load(mass_kg, g_mps2)
    longitudinal = max(-total_grip, min(total_grip, longitudinal_force_n))
    lateral_budget = sqrt(max(total_grip * total_grip - longitudinal * longitudinal, 0.0))
    lateral = max(-lateral_budget, min(lateral_budget, lateral_force_n))
    return longitudinal, lateral, total_grip


def tire_fit_summary(loads_n: tuple[float, ...] = (500.0, 1500.0, 2500.0, 4000.0)) -> list[str]:
    rows: list[str] = []
    for load in loads_n:
        force = max_tire_force(load)
        effective_mu = force / load if load > 0.0 else 0.0
        rows.append(f"N={load:.0f} N -> f={force:.1f} N, effective mu={effective_mu:.2f}")
    return rows
