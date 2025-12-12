from __future__ import annotations
from typing import Dict, Optional, Literal
import pandas as pd

from src.indices.hansen import hansen_accessibility, normalize_minmax, normalize_div_max

def compute_bci_masses(
    population: pd.Series,
    income: pd.Series,
    labour: pd.Series,
    hex_ids: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """
    BCI mass definitions (hex-indexed):
      Market mass = P * Y
      Labour mass = L
    """
    P = population.reindex(hex_ids).fillna(0.0).astype(float)
    Y = income.reindex(hex_ids).fillna(0.0).astype(float)
    L = labour.reindex(hex_ids).fillna(0.0).astype(float)
    return (P * Y).fillna(0.0), L.fillna(0.0)

def compute_bci(
    travel_times: dict,
    population: pd.Series,
    income: pd.Series,
    labour: pd.Series,
    betas: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    combine: Literal["weight_free", "weighted_sum"] = "weight_free",
    interface: Optional[pd.Series] = None,
    interface_lambda: float = 0.0,
) -> dict:
    """
    Compute BCI using Hansen structure (Harris/Hanson market potential downscaled):

      A_i^market = Σ_j ( (P_j*Y_j) * exp(-beta_market * t_ij) )
      A_i^labour = Σ_j ( (L_j)     * exp(-beta_labour * t_ij) )

    Combine:
      - weight_free:
          BCI_i = A_i^market / max(A^market) + A_i^labour / max(A^labour)
      - weighted_sum:
          BCI_i = w_market*A_i^market + w_labour*A_i^labour   (then normalized)

    Optional:
      BCI_i *= (1 + interface_lambda * rank_pct(interface_i))

    Returns dict with components:
      {"A_market", "A_labour", "BCI"}
    """
    hex_ids = population.index

    market_mass, labour_mass = compute_bci_masses(population, income, labour, hex_ids)

    beta_market = float(betas.get("market", 0.08))
    beta_labour = float(betas.get("labour", 0.08))

    A_market = hansen_accessibility(travel_times, market_mass, beta=beta_market)
    A_labour = hansen_accessibility(travel_times, labour_mass, beta=beta_labour)

    if combine == "weight_free":
        bci_raw = normalize_div_max(A_market) + normalize_div_max(A_labour)

    elif combine == "weighted_sum":
        w = weights or {}
        w_m = float(w.get("market", 1.0))
        w_l = float(w.get("labour", 1.0))
        bci_raw = (w_m * A_market) + (w_l * A_labour)

    else:
        raise ValueError(f"Unknown combine='{combine}'")

    # Optional interface multiplier
    if interface is not None and interface_lambda != 0.0:
        iface = interface.reindex(hex_ids).fillna(0.0)
        iface01 = iface.rank(pct=True)
        bci_raw = bci_raw * (1.0 + float(interface_lambda) * iface01)

    # Scale to 0–100 for presentation
    BCI = normalize_minmax(bci_raw, 100.0)

    return {"A_market": A_market, "A_labour": A_labour, "BCI": BCI}
