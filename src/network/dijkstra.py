from __future__ import annotations
import networkx as nx
from typing import Dict

def compute_hex_travel_times(
    G: nx.MultiDiGraph,
    hex_to_node: Dict,
    max_time: float = 60.0,
    weight: str = "time_min",
    progress_every: int = 50,
) -> Dict:
    """
    Compute travel times between hexes via Dijkstra on graph nodes.

    Args:
      G: scenario graph with edge weight attr `weight` (default 'time_min')
      hex_to_node: mapping hex_id -> assigned nearest graph node
      max_time: cutoff in minutes
      weight: edge attribute to use for shortest path
      progress_every: print progress every N sources

    Returns:
      travel_times[origin_hex][dest_hex] = shortest travel time in minutes (<= max_time)
    """
    # group hexes by their assigned node
    node_to_hexes = {}
    for h, n in hex_to_node.items():
        node_to_hexes.setdefault(n, []).append(h)

    unique_nodes = list(node_to_hexes.keys())
    travel_times = {}

    for i, source_node in enumerate(unique_nodes):
        if progress_every and (i % progress_every == 0):
            print(f"Dijkstra source {i+1}/{len(unique_nodes)}")

        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, source_node, cutoff=max_time, weight=weight
            )
        except Exception:
            continue

        # write lengths for each hex mapped to this source node
        for origin_hex in node_to_hexes[source_node]:
            od = {}
            for dest_hex, dest_node in hex_to_node.items():
                t = lengths.get(dest_node, None)
                if t is not None:
                    od[dest_hex] = float(t)
            travel_times[origin_hex] = od

    return travel_times
