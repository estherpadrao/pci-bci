from __future__ import annotations

from typing import Optional, Literal
import io
import zipfile
import requests
import pandas as pd
import geopandas as gpd

ACS_DATASET = "acs/acs5"

def fetch_acs(
    *,
    year: int,
    variables: list[str],
    for_clause: str,
    in_clause: Optional[str] = None,
    api_key: Optional[str] = None,
    dataset: str = ACS_DATASET
) -> pd.DataFrame:
    """
    Fetch ACS variables from the Census API.

    Example:
      variables=["B19013_001E"]
      for_clause="tract:*"
      in_clause="state:06 county:075"

    Returns a DataFrame with columns: variables + geographic id parts.
    """
    base = f"https://api.census.gov/data/{year}/{dataset}"
    get_vars = ["NAME"] + variables
    params = {
        "get": ",".join(get_vars),
        "for": for_clause,
    }
    if in_clause:
        params["in"] = in_clause
    if api_key:
        params["key"] = api_key

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    return df

def tiger_tracts(
    *,
    year: int,
    state_fips: str,
    county_fips: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Download TIGER/Line tract geometries.

    - If county_fips is provided, uses county-specific tract shapefile.
    - Otherwise downloads statewide tracts (bigger).

    Returns GeoDataFrame with GEOID and geometry (EPSG:4269 typically).
    """
    state_fips = str(state_fips).zfill(2)
    if county_fips is not None:
        county_fips = str(county_fips).zfill(3)
        # County tract shapefile
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}{county_fips}_tract.zip"
    else:
        # Statewide tract shapefile
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    # geopandas can read from an extracted temp directory; simplest: extract in-memory to /tmp via bytes
    # We'll extract to a temporary folder inside the current runtime.
    import tempfile, os
    tmpdir = tempfile.mkdtemp(prefix="tiger_tracts_")
    z.extractall(tmpdir)

    # find .shp
    shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
    if not shp_files:
        raise RuntimeError("No .shp found in TIGER zip.")
    gdf = gpd.read_file(shp_files[0])
    return gdf

def assign_tract_values_to_hexes(
    *,
    hex_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    value_col: str,
    hex_id_col: str = "hex_id",
    method: Literal["largest_area", "area_weighted"] = "largest_area"
) -> pd.Series:
    """
    Assign tract-level values (e.g., median income) to hexes.

    Methods:
      - largest_area (recommended for medians): pick the tract that overlaps each hex the most
      - area_weighted: area-weighted average (not ideal for medians, but available)

    Returns: Series indexed by hex_id (string).
    """
    if hex_gdf.crs is None or tracts_gdf.crs is None:
        raise ValueError("Both hex_gdf and tracts_gdf must have a CRS.")

    # project to a metric CRS for area calculations if needed
    # If hex CRS is geographic, project both to EPSG:3857 as a safe default.
    if hex_gdf.crs.is_geographic:
        hex_proj = hex_gdf.to_crs(3857)
        tracts_proj = tracts_gdf.to_crs(3857)
    else:
        hex_proj = hex_gdf
        tracts_proj = tracts_gdf.to_crs(hex_gdf.crs)

    # ensure ids are strings
    hex_proj = hex_proj.copy()
    hex_proj[hex_id_col] = hex_proj[hex_id_col].astype(str)

    # intersection overlay
    inter = gpd.overlay(
        hex_proj[[hex_id_col, "geometry"]],
        tracts_proj[[value_col, "geometry"]],
        how="intersection"
    )
    if inter.empty:
        return pd.Series(dtype=float)

    inter["area"] = inter.geometry.area
    inter[value_col] = pd.to_numeric(inter[value_col], errors="coerce")

    if method == "largest_area":
        inter = inter.dropna(subset=[value_col])
        inter = inter.sort_values("area", ascending=False)
        best = inter.drop_duplicates(subset=[hex_id_col], keep="first")
        return best.set_index(hex_id_col)[value_col].astype(float)

    if method == "area_weighted":
        inter = inter.dropna(subset=[value_col])
        # weighted mean by overlap area
        grouped = inter.groupby(hex_id_col).apply(
            lambda d: float((d[value_col] * d["area"]).sum() / d["area"].sum()) if d["area"].sum() > 0 else float("nan")
        )
        return grouped.astype(float)

    raise ValueError(f"Unknown method='{method}'")

def load_hex_income_from_acs(
    *,
    hex_gdf: gpd.GeoDataFrame,
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: Optional[str] = None,
    variable: str = "B19013_001E",
    hex_id_col: str = "hex_id",
    assign_method: Literal["largest_area", "area_weighted"] = "largest_area"
) -> pd.Series:
    """
    End-to-end:
      1) Fetch ACS median household income by tract
      2) Download tract geometries
      3) Assign tract income to hexes

    Returns: income Series indexed by hex_id (annual dollars).
    """
    # 1) ACS values
    df = fetch_acs(
        year=year,
        variables=[variable],
        for_clause="tract:*",
        in_clause=f"state:{str(state_fips).zfill(2)} county:{str(county_fips).zfill(3)}",
        api_key=api_key
    )

    # Build GEOID for tract
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df[variable] = pd.to_numeric(df[variable], errors="coerce")

    # 2) tract geometries
    tracts = tiger_tracts(year=year, state_fips=state_fips, county_fips=county_fips)
    if "GEOID" not in tracts.columns:
        raise ValueError("TIGER tracts missing GEOID column.")
    tracts["GEOID"] = tracts["GEOID"].astype(str)

    # 3) join values to geometries
    tracts = tracts.merge(df[["GEOID", variable]], on="GEOID", how="left")

    # 4) assign to hexes
    income_hex = assign_tract_values_to_hexes(
        hex_gdf=hex_gdf,
        tracts_gdf=tracts,
        value_col=variable,
        hex_id_col=hex_id_col,
        method=assign_method
    )

    # return fully reindexed series (fill missing with NaN, not 0, because income shouldn't become zero silently)
    out = income_hex.reindex(hex_gdf[hex_id_col].astype(str)).astype(float)
    out.index = hex_gdf[hex_id_col].astype(str)
    return out
