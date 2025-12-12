from __future__ import annotations
import networkx as nx
import pandas as pd

def normalize_01(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.0, index=s.index)

def build_topo_weighted_graph(
    G_base: nx.MultiDiGraph,
    topo_hex: pd.Series,       # hex_id -> topo (already clipped to hexes)
    node_to_hex: dict,         # node -> hex_id
    gamma: float,
    edge_overrides: dict | None = None,
) -> nx.MultiDiGraph:
    """
    Copy G_base and update edge 'time_min' to include a topo penalty:

      time_topo = time_base * (1 + gamma * avg(topo01(hu), topo01(hv)))

    Notes:
      - topo_hex is hex-indexed.
      - node_to_hex maps graph nodes -> hex_id.
      - This modifies time_min for *all* edges unless you later add mode filters.
      - edge_overrides is optional; we’re not using it right now because we already
        apply scenario network edits in apply_network_edits.py.
    """
    G = G_base.copy()
    topo01 = normalize_01(topo_hex)

    for u, v, k, data in G.edges(keys=True, data=True):
        base_t = data.get("time_min", None)
        if base_t is None:
            continue

        hu = node_to_hex.get(u, None)
        hv = node_to_hex.get(v, None)

        mu = float(topo01.get(hu, 0.0)) if hu is not None else 0.0
        mv = float(topo01.get(hv, 0.0)) if hv is not None else 0.0

        penalty = 1.0 + float(gamma) * 0.5 * (mu + mv)
        data["time_min"] = float(base_t) * penalty

    return G
