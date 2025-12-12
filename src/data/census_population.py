from __future__ import annotations

from typing import Optional, Literal
import pandas as pd
import geopandas as gpd

from src.data.census_income import fetch_acs, tiger_tracts, assign_tract_values_to_hexes

def load_hex_population_from_acs(
    *,
    hex_gdf: gpd.GeoDataFrame,
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: Optional[str] = None,
    variable: str = "B01003_001E",
    hex_id_col: str = "hex_id",
    assign_method: Literal["largest_area", "area_weighted"] = "area_weighted",
) -> pd.Series:
    """
    End-to-end:
      1) Fetch ACS population by tract
      2) Download tract geometries
      3) Assign tract population to hexes

    Defaults:
      variable = B01003_001E (Total population)

    assign_method:
      - area_weighted (default, appropriate for counts)
      - largest_area (available but less accurate for counts)
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
    pop_hex = assign_tract_values_to_hexes(
        hex_gdf=hex_gdf,
        tracts_gdf=tracts,
        value_col=variable,
        hex_id_col=hex_id_col,
        method=assign_method
    )

    # Reindex to all hexes (missing stays NaN, not forced to zero)
    out = pop_hex.reindex(hex_gdf[hex_id_col].astype(str)).astype(float)
    out.index = hex_gdf[hex_id_col].astype(str)
    return out
