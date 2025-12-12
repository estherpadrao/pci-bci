from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from .layers import MassLayer

def normalize_01(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.0, index=s.index)

def normalize_zscore(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    mu, sd = float(s.mean()), float(s.std(ddof=0))
    if sd > 0:
        return (s - mu) / sd
    return pd.Series(0.0, index=s.index)

@dataclass
class TopographyResult:
    """
    Holds the composite topography and optional diagnostics.
    """
    topo: pd.Series                      # hex_id -> composite value
    components: Dict[str, pd.Series]     # layer -> weighted contribution
    meta: Dict[str, Any]                 # settings used

def build_composite_topography(
    layers: Dict[str, MassLayer],
    weights: Dict[str, float],
    hex_ids: pd.Index,
    topo_rules: Dict[str, Any] | None = None
) -> TopographyResult:
    """
    Composite topography:
        topo = Σ_k w_k * layer_k

    topo_rules (optional):
      - normalize: "none" | "01" | "zscore"
      - clip_min: float
      - clip_max: float
      - log1p: bool  (apply log(1+x) to each layer before weighting)
    """
    topo_rules = topo_rules or {}
    normalize_mode = topo_rules.get("normalize", "none")
    clip_min = topo_rules.get("clip_min", None)
    clip_max = topo_rules.get("clip_max", None)
    log1p = bool(topo_rules.get("log1p", False))

    topo = pd.Series(0.0, index=hex_ids, dtype=float)
    comps: Dict[str, pd.Series] = {}

    for name, layer in layers.items():
        w = float(weights.get(name, 1.0))
        s = layer.reindex_to_hexes(hex_ids).raw.astype(float)

        if log1p:
            s = np.log1p(s.clip(lower=0.0))

        contrib = w * s
        comps[name] = contrib
        topo = topo.add(contrib, fill_value=0.0)

    if clip_min is not None:
        topo = topo.clip(lower=float(clip_min))
    if clip_max is not None:
        topo = topo.clip(upper=float(clip_max))

    if normalize_mode == "01":
        topo = normalize_01(topo)
    elif normalize_mode == "zscore":
        topo = normalize_zscore(topo)

    meta = {
        "weights": dict(weights),
        "topo_rules": dict(topo_rules),
        "normalize": normalize_mode,
        "log1p": log1p,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    return TopographyResult(topo=topo, components=comps, meta=meta)

def build_bci_masses(
    population: pd.Series,
    income: pd.Series,
    labour: pd.Series,
    hex_ids: pd.Index
) -> tuple[pd.Series, pd.Series]:
    """
    BCI mass definitions (hex-clipped / hex-indexed):
      Market mass = P * Y
      Labour mass = L

    Inputs must already be hex-indexed Series (or will be reindexed).
    """
    P = population.reindex(hex_ids).fillna(0.0).astype(float)
    Y = income.reindex(hex_ids).fillna(0.0).astype(float)
    L = labour.reindex(hex_ids).fillna(0.0).astype(float)

    market = (P * Y).fillna(0.0)
    labour_mass = L.fillna(0.0)
    return market, labour_mass
