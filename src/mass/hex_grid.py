from __future__ import annotations

import math
import geopandas as gpd
from shapely.geometry import Polygon

def _flat_top_hexagon(cx: float, cy: float, r: float) -> Polygon:
    """
    Flat-topped hexagon centered at (cx, cy) with circumradius r.
    Coordinates are assumed in a projected CRS (meters).
    """
    pts = []
    for k in range(6):
        ang = math.radians(60 * k)
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append((x, y))
    return Polygon(pts)

def generate_hex_grid_clipped(
    boundary_gdf: gpd.GeoDataFrame,
    hex_radius_m: float = 250.0,
    target_crs: str = "EPSG:3857",
    hex_id_prefix: str = "hex",
) -> gpd.GeoDataFrame:
    """
    Generate a flat-topped hex grid clipped to the boundary geometry.

    Args:
      boundary_gdf: GeoDataFrame containing a Polygon/MultiPolygon boundary.
      hex_radius_m: hex circumradius in meters (center->vertex). Controls resolution.
      target_crs: projected CRS for grid generation. EPSG:3857 is acceptable for city scale.
      hex_id_prefix: prefix for hex_id.

    Returns:
      GeoDataFrame with columns: ["hex_id", "geometry"], CRS = target_crs.
    """
    if boundary_gdf.crs is None:
        raise ValueError("boundary_gdf must have a CRS set.")

    # Project to working CRS (meters)
    b = boundary_gdf.to_crs(target_crs).copy()
    boundary = b.unary_union

    minx, miny, maxx, maxy = boundary.bounds

    r = float(hex_radius_m)
    dx = 1.5 * r
    dy = math.sqrt(3) * r

    # Build candidate hexes covering the bounding box
    hexes = []
    row = 0
    y = miny - dy
    while y <= maxy + dy:
        x_offset = 0.75 * r if (row % 2 == 1) else 0.0
        x = minx - dx + x_offset
        while x <= maxx + dx:
            h = _flat_top_hexagon(x, y, r)
            if h.intersects(boundary):
                hexes.append(h)
            x += dx
        y += dy
        row += 1

    gdf = gpd.GeoDataFrame({"geometry": hexes}, crs=target_crs)

    # Clip precisely to boundary
    gdf = gpd.clip(gdf, b)
    gdf = gdf[gdf.geometry.area > 0].copy()
    gdf = gdf.reset_index(drop=True)

    # Assign stable hex IDs
    gdf["hex_id"] = [f"{hex_id_prefix}_{i:06d}" for i in range(len(gdf))]

    return gdf[["hex_id", "geometry"]]
