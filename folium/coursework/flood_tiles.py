from pathlib import Path
import zipfile

import geopandas as gpd
import pandas as pd
import requests


FIPS = "48"
BASE_URL = "https://hazards.fema.gov/nfhlv2/output/State/"
DOWNLOAD_FOLDER = "FEMA24"
OUTPUT_FILE_NAME = "fema_flood_hazard.gpkg"
OUTPUT_LAYER_NAME = "Flood_Hazard_Layers"


def _read_gdb_layer(gdb_path: Path, layer_name: str) -> gpd.GeoDataFrame | None:
    """Read a named layer from a FileGDB using pyogrio. Returns None if not found."""
    try:
        gdf = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio")
        if not gdf.empty:
            print(f"  Loaded {layer_name} from {gdb_path.name}")
            return gdf
    except Exception:
        pass
    return None


def download_fema_data(download_dir: Path) -> None:
    """Download and extract the Texas NFHL ZIP from FEMA. Skips if already extracted."""
    if list(download_dir.rglob("*.gdb")):
        print("Using previously extracted FEMA data.")
        return

    print("Requesting FEMA product path for Texas...")
    response = requests.get(
        (
            "https://msc.fema.gov/portal/advanceSearch?affiliate=fema&query"
            f"&selstate={FIPS}&selcounty={FIPS}001&selcommunity={FIPS}001C"
            f"&searchedCid={FIPS}001C&method=search"
        ),
        timeout=120,
    )
    response.raise_for_status()

    try:
        product_id = response.json()["EFFECTIVE"]["NFHL_STATE_DATA"][0]["product_FILE_PATH"]
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError("Could not retrieve valid FEMA product data for Texas.") from exc

    url = BASE_URL + product_id
    zip_file_path = download_dir / Path(product_id).name

    print("Downloading Texas FEMA data...")
    download_response = requests.get(url, timeout=300)
    download_response.raise_for_status()
    zip_file_path.write_bytes(download_response.content)
    print("Download complete.")

    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_file_path) as zip_ref:
        zip_ref.extractall(download_dir)
    zip_file_path.unlink()
    print("Extraction complete.")


def load_flood_hazard(download_dir: Path) -> gpd.GeoDataFrame:
    """Load and merge all S_Fld_Haz_Ar layers from extracted geodatabases."""
    print("Searching for S_Fld_Haz_Ar feature classes...")
    frames = [
        gdf for gdb in sorted(download_dir.rglob("*.gdb"))
        if (gdf := _read_gdb_layer(gdb, "S_Fld_Haz_Ar")) is not None
    ]
    if not frames:
        raise RuntimeError("No S_Fld_Haz_Ar feature class found in extracted geodatabases.")
    print(f"  Merging {len(frames)} feature class(es)...")
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True)).set_geometry("geometry")
    if frames[0].crs:
        merged = merged.set_crs(frames[0].crs)
    return merged


def deduplicate_flood_hazard(flood_hazard: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove duplicate flood hazard records by primary key FLD_AR_ID."""
    if "FLD_AR_ID" not in flood_hazard.columns:
        print("  Warning: FLD_AR_ID not present, skipping deduplication.")
        return flood_hazard
    before = len(flood_hazard)
    flood_hazard = flood_hazard.drop_duplicates(subset=["FLD_AR_ID"]).copy()
    print(f"  Deduplication: {before:,} -> {len(flood_hazard):,} rows")
    return flood_hazard


def load_firm_panels(download_dir: Path) -> gpd.GeoDataFrame:
    """Load the S_FIRM_Pan layer containing map effective dates."""
    print("Searching for S_FIRM_Pan layer...")
    for gdb in sorted(download_dir.rglob("*.gdb")):
        gdf = _read_gdb_layer(gdb, "S_FIRM_Pan")
        if gdf is not None:
            return gdf
    raise RuntimeError("No S_FIRM_Pan feature class found.")


def add_effective_date(
    flood_hazard: gpd.GeoDataFrame,
    firm_panels: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatially join EFF_DATE from FIRM panels onto each flood hazard polygon."""
    if "EFF_DATE" not in firm_panels.columns:
        raise RuntimeError("S_FIRM_Pan does not contain EFF_DATE.")

    panels = firm_panels[["EFF_DATE", "geometry"]].copy()
    if flood_hazard.crs and panels.crs and flood_hazard.crs != panels.crs:
        panels = panels.to_crs(flood_hazard.crs)

    flood_hazard = flood_hazard.reset_index(drop=True)

    print("Joining effective date from S_FIRM_Pan...")
    joined = gpd.sjoin(flood_hazard, panels, how="left", predicate="intersects")

    # Handle the EFF_DATE_right suffix that sjoin adds on column name collision.
    eff_col = "EFF_DATE_right" if "EFF_DATE_right" in joined.columns else "EFF_DATE"
    eff_series = joined[eff_col].groupby(joined.index).first()

    flood_hazard = flood_hazard.copy()
    flood_hazard["EFF_DATE"] = eff_series
    return flood_hazard


def main() -> None:
    working_dir = Path.cwd()
    download_dir = working_dir / DOWNLOAD_FOLDER
    output_gpkg = working_dir / OUTPUT_FILE_NAME

    download_dir.mkdir(parents=True, exist_ok=True)

    download_fema_data(download_dir)
    flood_hazard = load_flood_hazard(download_dir)
    flood_hazard = deduplicate_flood_hazard(flood_hazard)
    firm_panels = load_firm_panels(download_dir)
    flood_hazard = add_effective_date(flood_hazard, firm_panels)

    is_excluded_zone = flood_hazard["FLD_ZONE"].isin(["AREA NOT INCLUDED", "OPEN WATER"])
    #is_minimal_x = (flood_hazard["FLD_ZONE"] == "X") & (flood_hazard["ZONE_SUBTY"] == "AREA OF MINIMAL FLOOD HAZARD")
    flood_hazard = flood_hazard[~is_excluded_zone].copy()
    
    x_masked = flood_hazard["FLD_ZONE"] == "X"
    minimal_mask = flood_hazard["ZONE_SUBTY"] == "AREA OF MINIMAL FLOOD HAZARD"
    
    flood_hazard.loc[x_masked & minimal_mask, "FLD_ZONE"] = "X - Unshaded"
    flood_hazard.loc[x_masked & ~minimal_mask, "FLD_ZONE"] = "X - Shaded"

    if output_gpkg.exists():
        output_gpkg.unlink()

    flood_hazard.to_file(output_gpkg, layer=OUTPUT_LAYER_NAME, driver="GPKG")
    print(f"Saved to: {output_gpkg}")


if __name__ == "__main__":
    main()
