"""
Scenario Playground (Baseline + One Scenario)

This script is meant to be executed in Colab as a reference.
You can copy/paste the sections into notebook cells if you prefer.

Order enforced:
  1) Load hexes (must exist)
  2) Load Census data (pop, income, labour) -> hex-indexed
  3) Build base network (placeholder until you plug your multimodal unified_graph)
  4) Build node->hex + hex->node mappings
  5) Build base_assets
  6) Run ScenarioRunner baseline
  7) Run one scenario
  8) Visualize maps (mass/topo/network/PCI/BCI)
"""

import os
import pandas as pd
import geopandas as gpd

from src.scenarios.artifacts import ArtifactStore
from src.scenarios.scenario import Scenario, RedistributeLayerToHexes, CloseEdges
from src.scenarios.runner import ScenarioRunner

from src.data.census_income import load_hex_income_from_acs
from src.data.census_labour import load_hex_labour_from_acs
from src.data.census_population import load_hex_population_from_acs

from src.network.base_network import build_osm_base_graph
from src.network.node_hex_map import assign_nodes_to_hexes, map_hex_centroids_to_nearest_nodes

from src.data.build_base_assets import build_base_assets
from src.mass.layers import MassLayer

from src.viz.folium_maps import map_hex_series, map_hex_layers, map_network_edges


def main():
    # -----------------------------
    # USER INPUTS (edit these)
    # -----------------------------
    HEX_PATH = os.environ.get("HEX_PATH", "")  # e.g. "/content/drive/MyDrive/sf_hexes.geojson"
    HEX_LAYER = os.environ.get("HEX_LAYER", "")  # if using a GeoPackage; else leave ""
    ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")  # set to Drive in Colab

    ACS_YEAR = int(os.environ.get("ACS_YEAR", "2022"))
    STATE_FIPS = os.environ.get("STATE_FIPS", "06")
    COUNTY_FIPS = os.environ.get("COUNTY_FIPS", "075")
    CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", None)

    PLACE_NAME = os.environ.get("PLACE_NAME", "San Francisco, California, USA")

    # -----------------------------
    # 1) Load hexes
    # -----------------------------
    if not HEX_PATH:
        raise ValueError("Set HEX_PATH to your hex grid file (GeoJSON, GeoPackage, etc.).")

    if HEX_LAYER:
        hex_gdf = gpd.read_file(HEX_PATH, layer=HEX_LAYER)
    else:
        hex_gdf = gpd.read_file(HEX_PATH)

    if "hex_id" not in hex_gdf.columns:
        raise ValueError("hex_gdf must contain a 'hex_id' column.")

    # Ensure hex_id is string
    hex_gdf["hex_id"] = hex_gdf["hex_id"].astype(str)

    # If CRS missing, you must set it
    if hex_gdf.crs is None:
        raise ValueError("hex_gdf has no CRS. Set it before proceeding.")

    hex_ids = pd.Index(hex_gdf["hex_id"].astype(str))

    # -----------------------------
    # 2) Load ACS data (tract -> hex)
    # -----------------------------
    income = load_hex_income_from_acs(
        hex_gdf=hex_gdf,
        year=ACS_YEAR,
        state_fips=STATE_FIPS,
        county_fips=COUNTY_FIPS,
        api_key=CENSUS_API_KEY,
        assign_method="largest_area",
    )

    population = load_hex_population_from_acs(
        hex_gdf=hex_gdf,
        year=ACS_YEAR,
        state_fips=STATE_FIPS,
        county_fips=COUNTY_FIPS,
        api_key=CENSUS_API_KEY,
        assign_method="area_weighted",
    )

    labour = load_hex_labour_from_acs(
        hex_gdf=hex_gdf,
        year=ACS_YEAR,
        state_fips=STATE_FIPS,
        county_fips=COUNTY_FIPS,
        api_key=CENSUS_API_KEY,
        assign_method="area_weighted",
    )

    # -----------------------------
    # 3) Build BASE NETWORK (placeholder)
    # Replace this with your true multimodal unified_graph later
    # -----------------------------
    G_base = build_osm_base_graph(PLACE_NAME, network_type="walk")

    # -----------------------------
    # 4) Build mappings: node->hex and hex->node
    # -----------------------------
    node_to_hex = assign_nodes_to_hexes(G_base, hex_gdf)
    hex_to_node = map_hex_centroids_to_nearest_nodes(G_base, hex_gdf)

    # -----------------------------
    # 5) Baseline amenity layers (placeholder)
    # Replace these with your real amenity layers clipped to hexes later
    # -----------------------------
    # For now we create empty layers so the pipeline runs end-to-end.
    layers = {
        "food_retail": MassLayer("food_retail", pd.Series(0.0, index=hex_ids)),
        "community": MassLayer("community", pd.Series(0.0, index=hex_ids)),
        "parks": MassLayer("parks", pd.Series(0.0, index=hex_ids)),  # can be area or coverage
    }
    weights = {"food_retail": 1.0, "community": 1.0, "parks": 0.0}

    # Optional: hex area (m^2) if you later use parks area for masking
    # If your hex CRS is geographic, this area is not meaningful—project first.
    hex_area_m2 = pd.Series(hex_gdf.to_crs(3857).geometry.area.values, index=hex_ids)

    # -----------------------------
    # 6) Build base_assets
    # -----------------------------
    base_assets = build_base_assets(
        hex_gdf=hex_gdf,
        layers=layers,
        weights=weights,
        G_base=G_base,
        node_to_hex=node_to_hex,
        hex_to_node=hex_to_node,
        population=population,
        income=income,
        labour=labour,
        hex_area_m2=hex_area_m2,
        urban_interface=None,
    )

    # -----------------------------
    # 7) Run baseline + scenario
    # -----------------------------
    store = ArtifactStore(root=ARTIFACT_DIR)
    runner = ScenarioRunner(store=store, base_assets=base_assets)

    baseline = Scenario(
        name="baseline",
        betas={"pci": 0.08, "market": 0.08, "labour": 0.08},
        gamma=0.5,
        active_lambda=0.30,
        mode_cost=0.0,
    )

    out0 = runner.run(baseline, max_time=60.0)

    # Example scenario: cluster businesses (placeholder) + close an edge (placeholder)
    # You will update target hexes and edges after you have real layers/network.
    scenario1 = Scenario(
        name="scenario1_cluster_and_close",
        mass_edits=[
            RedistributeLayerToHexes(layer="food_retail", target_hex_ids=list(hex_ids[:10]), method="equal"),
        ],
        network_edits=[
            # CloseEdges([(u, v)])  # fill with real node ids later
        ],
        betas={"pci": 0.08, "market": 0.08, "labour": 0.08},
        gamma=0.5,
        active_lambda=0.30,
        mode_cost=0.0,
    )
    out1 = runner.run(scenario1, max_time=60.0)

    # -----------------------------
    # 8) Visualization hooks (return maps)
    # -----------------------------
    m_topo = map_hex_series(hex_gdf, out0["topo_pci"], "PCI Topography (Baseline)")
    m_pci  = map_hex_series(hex_gdf, out0["pci"], "PCI (Baseline)")
    m_bci  = map_hex_series(hex_gdf, out0["bci"], "BCI (Baseline)")

    m_net  = map_network_edges(out0["G_s"], sample_n=6000)

    return {
        "out0": out0,
        "out1": out1,
        "maps": {"topo": m_topo, "pci": m_pci, "bci": m_bci, "net": m_net}
    }


if __name__ == "__main__":
    main()
