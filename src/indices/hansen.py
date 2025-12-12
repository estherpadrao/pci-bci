from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Fallback if income missing/invalid; you can set this to a city-specific value later
DEFAULT_MEDIAN_HOURLY_WAGE = 25.0

def income_cost_penalty_minutes(
    annual_income: float,
    mode_cost: float,
    fallback_hourly_wage: float = DEFAULT_MEDIAN_HOURLY_WAGE
) -> float:
    """
    Convert a monetary out-of-pocket cost into equivalent minutes using wage.
      hourly_wage = annual_income / 2080
      penalty_minutes = (mode_cost / hourly_wage) * 60
    """
    if mode_cost <= 0:
        return 0.0

    if annual_income is None or (not np.isfinite(annual_income)) or annual_income <= 0:
        hourly_wage = fallback_hourly_wage
    else:
        hourly_wage = max(float(annual_income) / 2080.0, 7.25)

    return (float(mode_cost) / float(hourly_wage)) * 60.0

def hansen_accessibility(
    travel_times: Dict,        # travel_times[origin_hex][dest_hex] = minutes
    dest_mass: pd.Series,      # Mass_j at destination (hex-indexed)
    beta: float,
    income: Optional[pd.Series] = None,
    mode_cost: float = 0.0,
    fallback_hourly_wage: float = DEFAULT_MEDIAN_HOURLY_WAGE,
) -> pd.Series:
    """
    Hansen accessibility:

      A_i = Σ_j [ Mass_j * exp( -beta * (t_ij + penalty_i) ) ]

    Where penalty_i is optional (income-generalized cost):
      penalty_i = (mode_cost / hourly_wage_i) * 60
    """
    hex_ids = dest_mass.index
    dest_mass = dest_mass.reindex(hex_ids).fillna(0.0).astype(float)

    out = pd.Series(0.0, index=hex_ids, dtype=float)

    for origin_hex in hex_ids:
        tt = travel_times.get(origin_hex, None)
        if not tt:
            continue

        penalty = 0.0
        if income is not None and mode_cost > 0.0:
            annual = float(income.get(origin_hex, np.nan))
            penalty = income_cost_penalty_minutes(
                annual_income=annual,
                mode_cost=mode_cost,
                fallback_hourly_wage=fallback_hourly_wage
            )

        total = 0.0
        for dest_hex, t in tt.items():
            m = float(dest_mass.get(dest_hex, 0.0))
            if m <= 0.0:
                continue
            t_eff = float(t) + penalty
            total += m * np.exp(-float(beta) * t_eff)

        out[origin_hex] = total

    return out

def normalize_minmax(s: pd.Series, scale: float = 100.0) -> pd.Series:
    s = s.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx > mn:
        return (s - mn) / (mx - mn) * float(scale)
    return pd.Series(float(scale) / 2.0, index=s.index, dtype=float)

def normalize_div_max(s: pd.Series) -> pd.Series:
    """
    Used for your weight-free BCI option:
      A_i / max(A)
    """
    s = s.astype(float)
    mx = float(s.max())
    if mx > 0:
        return s / mx
    return pd.Series(0.0, index=s.index, dtype=float)
