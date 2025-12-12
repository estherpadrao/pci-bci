from __future__ import annotations
from typing import Dict, Any, Tuple
import networkx as nx

from src.scenarios.scenario import (
    NetworkEdit,
    CloseEdges,
    SpeedFactorByAttribute,
    RemoveTransitLine,
)

def _edge_matches_where(data: dict, where: Dict[str, Any]) -> bool:
    """
    Simple attribute filter:
      where={"mode":"bike", "name":["Market St","Valencia St"]}
    matches if:
      data["mode"] == "bike" and data["name"] in that list
    """
    for k, v in where.items():
        if isinstance(v, (list, tuple, set)):
            if data.get(k) not in v:
                return False
        else:
            if data.get(k) != v:
                return False
    return True

def apply_network_edits(
    G_base: nx.MultiDiGraph,
    network_edits: list[NetworkEdit],
    close_time: float = 1e9
) -> nx.MultiDiGraph:
    """
    Apply scenario network edits to a *copy* of the base graph.

    Rules:
      - We do NOT topo-weight here. This is purely structural edits / base-time edits.
      - Closing an edge sets time_min to a huge value (or you can remove edges later).
      - SpeedFactorByAttribute multiplies time_min by factor for matching edges.
      - RemoveTransitLine disables transit edges matching route_id (expects attrs mode='transit', route_id).
    """
    G = G_base.copy()

    for edit in network_edits:
        # 1) Close specific edges
        if isinstance(edit, CloseEdges):
            for e in edit.edges:
                if len(e) == 2:
                    u, v = e
                    # affect all keys between u and v
                    if G.has_edge(u, v):
                        for k in list(G[u][v].keys()):
                            G[u][v][k]["time_min"] = close_time
                elif len(e) == 3:
                    u, v, k = e
                    if G.has_edge(u, v, k):
                        G[u][v][k]["time_min"] = close_time
                else:
                    raise ValueError(f"CloseEdges edge tuple must be (u,v) or (u,v,k). Got: {e}")
            continue

        # 2) Multiply times for edges matching attributes
        if isinstance(edit, SpeedFactorByAttribute):
            factor = float(edit.factor)
            for u, v, k, data in G.edges(keys=True, data=True):
                if _edge_matches_where(data, edit.where):
                    if "time_min" in data and data["time_min"] is not None:
                        data["time_min"] = float(data["time_min"]) * factor
            continue

        # 3) Remove a transit line by route_id
        if isinstance(edit, RemoveTransitLine):
            rid = str(edit.route_id)
            for u, v, k, data in G.edges(keys=True, data=True):
                if data.get("mode") == "transit" and str(data.get("route_id")) == rid:
                    data["time_min"] = close_time
            continue

        raise ValueError(f"Unsupported NetworkEdit type: {type(edit)}")

    return G
