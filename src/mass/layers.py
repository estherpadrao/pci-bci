from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd

@dataclass
class MassLayer:
    """
    A hex-indexed mass layer.

    raw: pd.Series indexed by hex_id, values are mass/count/area/etc.
    meta: any metadata (source, units, description, etc.)
    """
    name: str
    raw: pd.Series
    meta: Dict[str, Any] = field(default_factory=dict)

    def reindex_to_hexes(self, hex_ids: pd.Index, fill: float = 0.0) -> "MassLayer":
        """Ensure this layer covers all hexes, filling missing with `fill` (default 0)."""
        r = self.raw.reindex(hex_ids).fillna(fill).astype(float)
        return MassLayer(name=self.name, raw=r, meta=dict(self.meta))

    def copy(self) -> "MassLayer":
        return MassLayer(name=self.name, raw=self.raw.copy(), meta=dict(self.meta))

def layers_to_dict(layers: list[MassLayer], hex_ids: Optional[pd.Index] = None) -> Dict[str, MassLayer]:
    """Convert list[MassLayer] -> dict[name] = layer, optionally reindexed to hex_ids."""
    out: Dict[str, MassLayer] = {}
    for layer in layers:
        out[layer.name] = layer.reindex_to_hexes(hex_ids) if hex_ids is not None else layer
    return out
