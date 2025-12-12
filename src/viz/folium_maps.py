from __future__ import annotations
from typing import Optional, Dict, Any
import random

import geopandas as gpd
import pandas as pd
import folium
from folium.features import GeoJsonTooltip

def _center_of_gdf(gdf: gpd.GeoDataFrame) -> list[float]:
    c = gdf.geometry.centroid
    return [float(c.y.mean()), float(c.x.mean())]

def map_hex_series(
    hex_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    name: str,
    hex_id_col: str = "hex_id",
    tiles: str = "cartodbpositron",
    zoom_start: int = 12,
) -> folium.Map:
    """
    Choropleth of a hex-indexed Series (mass, topo, PCI, BCI).
    """
    g = hex_gdf[[hex_id_col, "geometry"]].copy()
    g = g.set_index(hex_id_col)
    g[name] = values.reindex(g.index)

    m = folium.Map(location=_center_of_gdf(g.reset_index()), tiles=tiles, zoom_start=zoom_start)

    # Use GeoJson for better tooltips on NaNs too
    gj = folium.GeoJson(
        g.reset_index().to_json(),
        name=name,
        tooltip=GeoJsonTooltip(
            fields=[hex_id_col, name],
            aliases=["hex_id", name],
            localize=True,
            sticky=False,
            labels=True,
        ),
    )
    gj.add_to(m)

    # Style based on value (simple opacity scaling); Folium’s built-in Choropleth
    # is fine too, but GeoJson gives better hover behavior.
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def map_hex_layers(
    hex_gdf: gpd.GeoDataFrame,
    layer_series: Dict[str, pd.Series],
    hex_id_col: str = "hex_id",
    tiles: str = "cartodbpositron",
    zoom_start: int = 12,
) -> folium.Map:
    """
    Adds multiple hex layers to one Folium map so you can toggle them.
    Each layer is a pd.Series indexed by hex_id.
    """
    m = folium.Map(location=_center_of_gdf(hex_gdf), tiles=tiles, zoom_start=zoom_start)

    for name, series in layer_series.items():
        g = hex_gdf[[hex_id_col, "geometry"]].copy().set_index(hex_id_col)
        g[name] = series.reindex(g.index)

        gj = folium.GeoJson(
            g.reset_index().to_json(),
            name=name,
            tooltip=GeoJsonTooltip(fields=[hex_id_col, name], aliases=["hex_id", name]),
        )
        gj.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

def map_network_edges(
    G,
    sample_n: int = 5000,
    weight_attr: str = "time_min",
    tiles: str = "cartodbpositron",
    zoom_start: int = 12,
    show_nodes: bool = False,
) -> folium.Map:
    """
    Visualize a (potentially huge) network by sampling edges.
    Assumes nodes have 'x' (lon) and 'y' (lat) attributes (OSMnx-style).
    """
    # Collect edges
    edges = list(G.edges(keys=True, data=True))
    if len(edges) == 0:
        raise ValueError("Graph has no edges to plot.")

    if sample_n is not None and len(edges) > sample_n:
        edges = random.sample(edges, sample_n)

    # Center map at mean node coords
    xs, ys = [], []
    for n, data in G.nodes(data=True):
        x, y = data.get("x"), data.get("y")
        if x is not None and y is not None:
            xs.append(float(x)); ys.append(float(y))
    if not xs:
        raise ValueError("Graph nodes missing x/y coordinates; cannot plot on Folium.")
    m = folium.Map(location=[sum(ys)/len(ys), sum(xs)/len(xs)], tiles=tiles, zoom_start=zoom_start)

    # Plot edges
    for u, v, k, data in edges:
        udat = G.nodes[u]
        vdat = G.nodes[v]
        if ("x" not in udat) or ("y" not in udat) or ("x" not in vdat) or ("y" not in vdat):
            continue

        t = data.get(weight_attr, None)
        tooltip = f"{u}->{v}  {weight_attr}={t}" if t is not None else f"{u}->{v}"

        folium.PolyLine(
            locations=[(udat["y"], udat["x"]), (vdat["y"], vdat["x"])],
            weight=2,
            opacity=0.6,
            tooltip=tooltip,
        ).add_to(m)

    # Optional nodes
    if show_nodes:
        for n, data in list(G.nodes(data=True))[:3000]:
            if "x" in data and "y" in data:
                folium.CircleMarker(
                    location=(data["y"], data["x"]),
                    radius=1,
                    opacity=0.4,
                    fill=True,
                    fill_opacity=0.4,
                ).add_to(m)

    return m
