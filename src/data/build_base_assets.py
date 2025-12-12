from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import networkx as nx

from src.mass.layers import MassLayer

def build_base_assets(
    *,
    hex_gdf,                     # GeoDataFrame with geometry + hex_id
    layers: Dict[str, MassLayer],
    weights: Dict[str, float],
    G_base: nx.MultiDiGraph,
    node_to_hex: Dict,
    hex_to_node: Dict,
    population: pd.Series,
    income: pd.Series,
    labour: pd.Series,
    hex_area_m2: pd.Series | None = None,
    urban_interface: pd.Series | None = None,
) -> Dict[str, Any]:
    """
    Build the base_assets dictionary required by ScenarioRunner.

    This function does NOT:
      - fetch data
      - edit scenarios
      - run networks
      - compute indices

    It ONLY:
      - validates inputs
      - enforces hex indexing
      - returns a clean, canonical base_assets object
    """

    # -----------------------------
    # Validate hex ids
    # -----------------------------
    if "hex_id" not in hex_gdf.columns:
        raise ValueError("hex_gdf must have a 'hex_id' column")

    hex_ids = pd.Index(hex_gdf["hex_id"].astype(str))

    # -----------------------------
    # Reindex all hex-based inputs
    # -----------------------------
    layers_out = {
        name: layer.reindex_to_hexes(hex_ids)
        for name, layer in layers.items()
    }

    population = population.reindex(hex_ids).fillna(0.0)
    income = income.reindex(hex_ids).fillna(0.0)
    labour = labour.reindex(hex_ids).fillna(0.0)

    if hex_area_m2 is not None:
        hex_area_m2 = hex_area_m2.reindex(hex_ids).fillna(0.0)

    if urban_interface is not None:
        urban_interface = urban_interface.reindex(hex_ids).fillna(0.0)

    # -----------------------------
    # Final object
    # -----------------------------
    base_assets = {
        "hex_gdf": hex_gdf,
        "hex_ids": hex_ids,
        "layers": layers_out,
        "weights": dict(weights),
        "G_base": G_base,
        "node_to_hex": dict(node_to_hex),
        "hex_to_node": dict(hex_to_node),
        "population": population,
        "income": income,
        "labour": labour,
        "hex_area_m2": hex_area_m2,
        "urban_interface": urban_interface,
    }

    return base_assets
