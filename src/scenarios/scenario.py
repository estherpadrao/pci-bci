from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Tuple

# -----------------------------
# Mass edits (hex-clipped layers)
# -----------------------------

@dataclass(frozen=True)
class MassEdit:
    """Base class for edits that change hex-level masses."""
    kind: str

@dataclass(frozen=True)
class RedistributeLayerToHexes(MassEdit):
    """
    Take all mass in a layer and redistribute into target hexes.

    Example use: "cluster all businesses into these hexes".
    """
    layer: str
    target_hex_ids: List[str]
    method: Literal["equal", "proportional_to_existing"] = "equal"

    def __init__(self, layer: str, target_hex_ids: List[str], method: str = "equal"):
        object.__setattr__(self, "kind", "redistribute_layer_to_hexes")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "target_hex_ids", target_hex_ids)
        object.__setattr__(self, "method", method)

@dataclass(frozen=True)
class RemoveLayerFromComposite(MassEdit):
    """
    Remove a layer from the PCI composite by forcing its weight to 0 at build time.
    Example: "remove parks contribution".
    """
    layer: str

    def __init__(self, layer: str):
        object.__setattr__(self, "kind", "remove_layer_from_composite")
        object.__setattr__(self, "layer", layer)

# -----------------------------
# Network edits (edge changes)
# -----------------------------

@dataclass(frozen=True)
class NetworkEdit:
    """Base class for edits that change the network graph."""
    kind: str

@dataclass(frozen=True)
class CloseEdges(NetworkEdit):
    """
    Close specific edges by node ids.
    Accepts (u,v) or (u,v,key) tuples.
    """
    edges: List[Tuple]

    def __init__(self, edges: List[Tuple]):
        object.__setattr__(self, "kind", "close_edges")
        object.__setattr__(self, "edges", edges)

@dataclass(frozen=True)
class SpeedFactorByAttribute(NetworkEdit):
    """
    Multiply time on edges that match attribute filters.
    Example: add bike lanes -> reduce bike edge time by factor 0.8.
    """
    where: Dict[str, Any]      # e.g. {"mode":"bike", "name":["Valencia St","Market St"]}
    factor: float

    def __init__(self, where: Dict[str, Any], factor: float):
        object.__setattr__(self, "kind", "speed_factor_by_attribute")
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "factor", float(factor))

@dataclass(frozen=True)
class RemoveTransitLine(NetworkEdit):
    """
    Remove (or effectively disable) a transit line by route id/name.
    Requires transit edges to carry attributes like mode='transit' and route_id.
    """
    route_id: str

    def __init__(self, route_id: str):
        object.__setattr__(self, "kind", "remove_transit_line")
        object.__setattr__(self, "route_id", route_id)

# -----------------------------
# Scenario (what you asked for)
# -----------------------------

@dataclass(frozen=True)
class Scenario:
    """
    Scenario = baseline + a set of edits + parameter changes.

    This stays stable over time even as you add more edit types.
    """
    name: str = "baseline"
    description: str = ""

    # Edits
    mass_edits: List[MassEdit] = field(default_factory=list)
    network_edits: List[NetworkEdit] = field(default_factory=list)

    # PCI/BCI parameters (these absolutely should vary by scenario)
    weights: Dict[str, float] = field(default_factory=dict)   # layer weights
    betas: Dict[str, float] = field(default_factory=dict)     # {"pci":..,"market":..,"labour":..} or per-layer later
    gamma: float = 0.5                                        # topo penalty strength

    active_lambda: float = 0.30
    mode_cost: float = 0.0

    park_mask_enabled: bool = True
    park_mask_threshold: float = 0.90

    interface_lambda: float = 0.0
