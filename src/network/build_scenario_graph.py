from __future__ import annotations
import networkx as nx
import pandas as pd

from src.scenarios.scenario import Scenario
from src.scenarios.apply_network_edits import apply_network_edits
from src.network.topo_weighted import build_topo_weighted_graph

def build_scenario_graph(
    G_base: nx.MultiDiGraph,
    topo_hex: pd.Series,     # hex_id -> topo (already computed + clipped)
    node_to_hex: dict,       # node -> hex_id
    scenario: Scenario,
    close_time: float = 1e9,
) -> nx.MultiDiGraph:
    """
    Build the final scenario-specific graph used for Dijkstra:
      base graph -> apply network edits -> apply topo penalty

    This is the ONLY graph that should be used for travel times.
    """
    # 1) apply network edits (closures/speed factors/transit removals)
    G_edit = apply_network_edits(G_base, scenario.network_edits, close_time=close_time)

    # 2) apply topographic penalty to time_min (gamma)
    G_topo = build_topo_weighted_graph(
        G_base=G_edit,
        topo_hex=topo_hex,
        node_to_hex=node_to_hex,
        gamma=scenario.gamma,
        edge_overrides=None,  # already applied via apply_network_edits
    )

    return G_topo
