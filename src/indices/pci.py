from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.mass.layers import MassLayer
from src.indices.hansen import hansen_accessibility, normalize_minmax

def compute_active_street_score(
    hex_ids: pd.Index,
    layers: Dict[str, MassLayer],
    active_layers: tuple[str, ...] = ("food_retail", "community"),
) -> pd.Series:
    """
    ActiveStreetScore_i = percentile rank of (sum of selected layers) over hexes.
    Default: food_retail + community (you can change later).
    Output is in [0,1].
    """
    active = pd.Series(0.0, index=hex_ids, dtype=float)
    for k in active_layers:
        if k in layers:
            active = active.add(layers[k].raw.reindex(hex_ids).fillna(0.0), fill_value=0.0)
    return active.rank(pct=True)

def compute_park_coverage(
    hex_area_m2: pd.Series,
    parks_area_m2: pd.Series,
    hex_ids: pd.Index
) -> pd.Series:
    """
    Coverage_i = park_area_i / hex_area_i
    """
    a_hex = hex_area_m2.reindex(hex_ids).astype(float)
    a_park = parks_area_m2.reindex(hex_ids).fillna(0.0).astype(float)
    cov = a_park / a_hex.replace(0, np.nan)
    return cov.fillna(0.0)

def compute_pci(
    travel_times: dict,
    topo_mass: pd.Series,                    # PCI destination mass (hex-indexed)
    layers: Dict[str, MassLayer],            # hex-clipped layers (for ActiveStreetScore + optional parks)
    betas: Dict[str, float],
    income: Optional[pd.Series] = None,      # hex-indexed annual income
    mode_cost: float = 0.0,
    active_lambda: float = 0.30,
    park_mask_enabled: bool = True,
    park_mask_threshold: float = 0.90,
    hex_area_m2: Optional[pd.Series] = None  # required if park_mask_enabled and parks layer is area
) -> pd.Series:
    """
    PCI workflow (correct order, no network recompute):
      1) Hansen accessibility on topo_mass with income penalty option
      2) Normalize A_i to 0–100
      3) ActiveStreetScore_i from layers -> multiplier (1 + λ * score)
      4) Normalize again to 0–100
      5) Optional park mask

    Returns:
      pci: pd.Series indexed by hex_id (0–100, NaN for masked if enabled)
    """
    hex_ids = topo_mass.index

    beta = float(betas.get("pci", 0.08))

    # 1) Hansen
    A = hansen_accessibility(
        travel_times=travel_times,
        dest_mass=topo_mass,
        beta=beta,
        income=income,
        mode_cost=mode_cost,
    )

    # 2) Normalize accessibility to 0–100
    A_n = normalize_minmax(A, scale=100.0)

    # 3) Active street multiplier
    active_score = compute_active_street_score(hex_ids, layers)
    pci_raw = A_n * (1.0 + float(active_lambda) * active_score)

    # 4) Normalize to 0–100
    pci = normalize_minmax(pci_raw, scale=100.0)

    # 5) Park mask (optional)
    if park_mask_enabled:
        if "parks" in layers:
            parks = layers["parks"].raw.reindex(hex_ids).fillna(0.0).astype(float)

            # If parks layer is already a fraction/coverage, you can pass hex_area_m2=None
            # We detect by magnitude: if max <= 1.5 assume it's already a coverage ratio.
            if float(parks.max()) <= 1.5:
                coverage = parks
            else:
                if hex_area_m2 is None:
                    raise ValueError("hex_area_m2 must be provided when parks layer is area (m^2).")
                coverage = compute_park_coverage(hex_area_m2=hex_area_m2, parks_area_m2=parks, hex_ids=hex_ids)

            mask = coverage > float(park_mask_threshold)
            pci = pci.where(~mask, np.nan)

    return pci
