from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd

from src.mass.layers import MassLayer
from src.scenarios.scenario import (
    MassEdit,
    RedistributeLayerToHexes,
    RemoveLayerFromComposite,
)

def _redistribute_series_to_targets(
    s: pd.Series,
    hex_ids: pd.Index,
    target_hex_ids: list[str],
    method: str = "equal",
) -> pd.Series:
    """
    Redistribute all mass in s into target_hex_ids, preserving total mass.
    """
    s = s.reindex(hex_ids).fillna(0.0).astype(float)
    total = float(s.sum())

    if total == 0.0:
        return s * 0.0

    targets = [h for h in target_hex_ids if h in set(hex_ids)]
    if len(targets) == 0:
        raise ValueError("No target_hex_ids are in the provided hex_ids index.")

    out = pd.Series(0.0, index=hex_ids, dtype=float)

    if method == "equal":
        share = total / len(targets)
        out.loc[targets] = share
        return out

    if method == "proportional_to_existing":
        base = s.loc[targets].clip(lower=0.0)
        denom = float(base.sum())
        if denom <= 0.0:
            # fall back to equal if all target masses are zero
            share = total / len(targets)
            out.loc[targets] = share
            return out
        out.loc[targets] = (base / denom) * total
        return out

    raise ValueError(f"Unknown redistribution method: {method}")

def apply_mass_edits(
    layers: Dict[str, MassLayer],
    weights: Dict[str, float],
    hex_ids: pd.Index,
    mass_edits: list[MassEdit],
) -> Tuple[Dict[str, MassLayer], Dict[str, float]]:
    """
    Applies scenario mass edits to hex-clipped MassLayers.

    Returns:
      (new_layers, new_weights)

    Notes:
      - RedistributeLayerToHexes modifies the MassLayer.raw for that layer.
      - RemoveLayerFromComposite modifies weights by forcing that layer's weight to 0.
    """
    new_layers = {k: v.copy().reindex_to_hexes(hex_ids) for k, v in layers.items()}
    new_weights = dict(weights)

    for edit in mass_edits:
        if isinstance(edit, RedistributeLayerToHexes):
            if edit.layer not in new_layers:
                raise KeyError(f"Layer '{edit.layer}' not found in layers. Available: {list(new_layers.keys())}")

            layer = new_layers[edit.layer]
            redistributed = _redistribute_series_to_targets(
                layer.raw,
                hex_ids=hex_ids,
                target_hex_ids=edit.target_hex_ids,
                method=edit.method,
            )
            new_layers[edit.layer] = MassLayer(name=layer.name, raw=redistributed, meta=dict(layer.meta))
            continue

        if isinstance(edit, RemoveLayerFromComposite):
            # Keep the layer around (for debugging/visualization), but remove it from the composite
            new_weights[edit.layer] = 0.0
            continue

        raise ValueError(f"Unsupported MassEdit type: {type(edit)}")

    return new_layers, new_weights
