# step 1

# Expect columns: lat, lon, TIV, CoverageA (names can be adapted)
gdf = gpd.GeoDataFrame(
    port_df,
    geometry=gpd.points_from_xy(port_df["Lon"], port_df["Lat"]),
    crs="EPSG:4326"
)


# -------------------------
# 2) Project to a meter-based CRS
# -------------------------
# Easy, robust option for a single state: UTM zone based on data centroid
centroid = gdf.geometry.union_all().centroid
lon0, lat0 = centroid.x, centroid.y
utm_zone = int((lon0 + 180) // 6) + 1
epsg_utm = 32600 + utm_zone if lat0 >= 0 else 32700 + utm_zone  # WGS84 / UTM

# use 5070 for US
gdf_m = gdf.to_crs(5070)


# -------------------------
# 3) Build a HEX grid covering the points
# -------------------------

# H3 resolution (0=coarsest, 15=finest)
# Resolution 7 ≈ 5km², Resolution 6 ≈ 36km², Resolution 5 ≈ 252km²
h3_resolution = 6

# Convert points to H3 hex IDs (back in WGS84 for h3)
gdf_wgs = gdf.copy()  # original WGS84 gdf from step 1
gdf_wgs["h3_id"] = gdf_wgs.apply(
    lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, h3_resolution), axis=1
)

# Build hex polygons only for cells that contain points
hex_ids = gdf_wgs["h3_id"].unique()
hex_polys = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)]) for h in hex_ids]


grid = gpd.GeoDataFrame({"h3_id": hex_ids}, geometry=hex_polys, crs="EPSG:4326")

# Project to meters for area calculations in later steps
grid_m = grid.to_crs(5070)

# Sanity check
print(f"Total hex cells: {len(grid)}")
print(grid.head())


# -------------------------
# 4) Join points -> hex cells
# -------------------------
joined = gdf_wgs.copy()  # already has h3_id from step 3


# -------------------------
# 5) Aggregate by hex cell
# -------------------------
agg = (
    joined.groupby(["h3_id","Port"], dropna=True)
    .agg(
        risk_count=("geometry", "size"),
        tiv_sum=("TIV", "sum"),
    )
    .reset_index()
)

grid_out = grid.merge(agg, on="h3_id", how="left").fillna(
    {"risk_count": 0, "tiv_sum": 0.0}
)


# -------------------------
# 6) Compute density metrics
# -------------------------
# H3 hex area varies slightly by location — use average area for the resolution
cell_area_m2 = h3.average_hexagon_area(h3_resolution, unit='m^2')

grid_out["tiv_per_sqkm"] = grid_out["tiv_sum"] / (cell_area_m2 / 1_000_000)
grid_out["risks_per_sqkm"] = grid_out["risk_count"] / (cell_area_m2 / 1_000_000)

SQM_PER_SQMI = 2_589_988.110336
grid_out["tiv_per_sqmi"] = grid_out["tiv_sum"] / (cell_area_m2 / SQM_PER_SQMI)
grid_out["risks_per_sqmi"] = grid_out["risk_count"] / (cell_area_m2 / SQM_PER_SQMI)

# Sanity check
print(f"H3 resolution {h3_resolution} avg hex area: {cell_area_m2/1_000_000:.2f} km²")



# -------------------------
# 7) Export
# -------------------------
# Use H3 to get accurate centroids (lat/lon) directly from hex ID
grid_out[["centroid_y", "centroid_x"]] = grid_out["h3_id"].apply(
    lambda h: pd.Series(h3.cell_to_latlng(h))  # returns (lat, lon)
)

# this is a geopandas dataframe
grid_out


# save the grid_out
app_dir = r'C:\Users\BryanButler\OneDrive - Hadron Specialty Insurance EU\Documents\PythonFiles\USPortfolioMaps'

grid_out.to_file(os.path.join(app_dir, "grid_data.geojson"), driver="GeoJSON")





