grid_wgs = gpd.read_file(os.path.join(app_dir, "grid_data.geojson"))

print(grid_wgs.columns.tolist())
print(grid_wgs["Port"].value_counts())
print(f"Total rows: {len(grid_wgs)}")
# print(f"port_list: {port_list}")



# ── Output folder — define once at the top ───────────────────────────────────────
folder = r"C:\Users\BryanButler\OneDrive - Hadron Specialty Insurance EU\Documents\PythonFiles\USPortfolioMaps"



# ── Data folder ───────────────────────────────────────────────────────────────────
data_folder = r"C:\Users\BryanButler\OneDrive - Hadron Specialty Insurance EU\Documents\PythonFiles\USPortfolioMaps\SCS"
os.makedirs(data_folder, exist_ok=True)

# ── Tornado tracks ────────────────────────────────────────────────────────────────
torn_save = os.path.join(data_folder, "torn_recent.parquet")

headers = {"User-Agent": "Mozilla/5.0"}

if os.path.exists(torn_save):
    torn_recent = gpd.read_parquet(torn_save)
    print(f"Tornado tracks loaded from cache — {len(torn_recent)} tracks")
else:
    print("Downloading tornado tracks...")
    torn_extract = os.path.join(data_folder, "torn_shp")
    os.makedirs(torn_extract, exist_ok=True)
    r = requests.get( "https://www.spc.noaa.gov/gis/svrgis/zipped/1950-2025-torn-aspath.zip",
                      timeout=60,
                      headers=headers)
    print(f"Status code: {r.status_code}")
    print(f"Content length: {len(r.content)} bytes")

    
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(torn_extract)
    torn_sub = os.path.join(torn_extract, "1950-2025-torn-aspath")
    shp_path = os.path.join(torn_sub,
                            [f for f in os.listdir(torn_sub) if f.endswith(".shp")][0])
    
    torn_gdf    = gpd.read_file(shp_path).to_crs("EPSG:4326")
    torn_recent = torn_gdf[torn_gdf["yr"].between(2015, 2025)].copy()
    torn_recent.to_parquet(torn_save, index=False)
    print(f"Tornado tracks saved — {len(torn_recent)} tracks (2015-2025)")

# ── Hail reports ──────────────────────────────────────────────────────────────────
hail_save = os.path.join(data_folder, "hail_recent.parquet")

if os.path.exists(hail_save):
    hail_recent = gpd.read_parquet(hail_save)
    print(f"Hail reports loaded from cache — {len(hail_recent)} reports")
else:
    print("Downloading hail reports...")
    hail_extract = os.path.join(data_folder, "hail_shp")
    os.makedirs(hail_extract, exist_ok=True)
    r = requests.get("https://www.spc.noaa.gov/gis/svrgis/zipped/1955-2025-hail-aspath.zip", timeout=60, headers=headers)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(hail_extract)
    hail_sub = os.path.join(hail_extract, "1955-2025-hail-aspath")
    shp_path = os.path.join(hail_sub,
                            [f for f in os.listdir(hail_sub) if f.endswith(".shp")][0])
    hail_gdf    = gpd.read_file(shp_path).to_crs("EPSG:4326")
    hail_recent = hail_gdf[
        hail_gdf["yr"].between(2015, 2025) &
        (hail_gdf["mag"] >= 1.0)
    ].copy()
    hail_recent.to_parquet(hail_save, index=False)
    print(f"Hail reports saved — {len(hail_recent)} reports (2015-2025, >= 1in)")

# ── Wind reports ──────────────────────────────────────────────────────────────────
wind_save = os.path.join(data_folder, "wind_recent.parquet")

if os.path.exists(wind_save):
    wind_recent = gpd.read_parquet(wind_save)
    print(f"Wind reports loaded from cache — {len(wind_recent)} reports")
else:
    print("Downloading wind reports...")
    wind_extract = os.path.join(data_folder, "wind_shp")
    os.makedirs(wind_extract, exist_ok=True)
    r = requests.get("https://www.spc.noaa.gov/gis/svrgis/zipped/1955-2025-wind-aspath.zip", timeout=60, headers=headers)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(wind_extract)
    wind_sub = os.path.join(wind_extract, "1955-2025-wind-aspath")
    shp_path = os.path.join(wind_sub,
                            [f for f in os.listdir(wind_sub) if f.endswith(".shp")][0])
    wind_gdf    = gpd.read_file(shp_path).to_crs("EPSG:4326")
    wind_recent = wind_gdf[wind_gdf["yr"].between(2015, 2025)].copy()
    wind_recent.to_parquet(wind_save, index=False)
    print(f"Wind reports saved — {len(wind_recent)} reports (2015-2025)")

print("\nAll SCS data ready.")



# ── Build SCS JSON by year ────────────────────────────────────────────────────────
import json
def build_scs_by_year(torn_gdf, hail_gdf, wind_gdf):
    scs_by_year = {}
    
    years = sorted(set(
        list(torn_gdf["yr"].unique()) +
        list(hail_gdf["yr"].unique()) +
        list(wind_gdf["yr"].unique())
    ))
    
    for year in years:
        torn_yr = torn_gdf[torn_gdf["yr"] == year]
        hail_yr = hail_gdf[hail_gdf["yr"] == year]
        wind_yr = wind_gdf[wind_gdf["yr"] == year]
        
        # Tornado tracks
        tornadoes = []
        for _, row in torn_yr.iterrows():
            try:
                tornadoes.append({
                    "lat1": row["slat"], "lon1": row["slon"],
                    "lat2": row["elat"], "lon2": row["elon"],
                    "mag":  int(row["mag"]) if row["mag"] >= 0 else -1,
                    "st":   row["st"],
                    "mo":   int(row["mo"]),
                    "dy":   int(row["dy"]),
                    "len":  round(float(row["len"]), 1),
                    "wid":  int(row["wid"]),
                })
            except Exception:
                continue
        # Hail reports
        hail_pts = []
        for _, row in hail_yr.iterrows():
            try:
                hail_pts.append({
                    "lat": row["slat"], "lon": row["slon"],
                    "mag": round(float(row["mag"]), 2),
                    "st":  row["st"],
                    "mo":  int(row["mo"]),
                    "dy":  int(row["dy"]),
                })
            except Exception:
                continue
        # Wind reports — filtered to >= 65 mph
        wind_pts = []
        for _, row in wind_yr.iterrows():
            try:
                mag = int(row["mag"]) if row["mag"] >= 0 else 0
                if mag < 65:
                    continue
                wind_pts.append({
                    "lat": row["slat"], "lon": row["slon"],
                    "mag": mag,
                    "st":  row["st"],
                    "mo":  int(row["mo"]),
                    "dy":  int(row["dy"]),
                })
            except Exception:
                continue
        scs_by_year[str(year)] = {
            "tornadoes": tornadoes,
            "hail":      hail_pts,
            "wind":      wind_pts,
        }
    
    return scs_by_year
scs_by_year  = build_scs_by_year(torn_recent, hail_recent, wind_recent)
scs_json_str = json.dumps(scs_by_year)
# Save to disk
scs_json_path = os.path.join(folder, "scs_data.json")
with open(scs_json_path, "w", encoding="utf-8") as f:
    f.write(scs_json_str)
print(f"scs_data.json: {os.path.getsize(scs_json_path)/1024/1024:.2f} MB")
print(f"Years: {sorted(scs_by_year.keys())}")


# import the grid


# ── Grid prep ────────────────────────────────────────────────────────────────────

print(grid_wgs.crs)
print(f"Null tiv_sum:  {grid_wgs['tiv_sum'].isna().sum()}")
print(f"Null geometry: {grid_wgs.geometry.isna().sum()}")
print(f"Rows:          {len(grid_wgs)}")

# grid prep
grid_wgs = grid_out.copy()
grid_wgs["tiv_sum"] = grid_wgs["tiv_sum"].fillna(0).round(0).astype(int)
value_col  = "tiv_sum"
# Global scale across ALL portfolios — compute before any per-port loop
vmin_round = int(np.floor(grid_wgs[value_col].min()))
vmax_round = int(np.ceil(grid_wgs[value_col].max()))
port_list  = sorted(grid_wgs["Port"].dropna().unique().tolist())

print(f"port_list at grid prep: {port_list}")  # ← add this

# ── Formatted tooltip columns ────────────────────────────────────────────────────
def fmt_int_commas(x):
    try:    return f"{int(round(float(x))):,}"
    except: return ""

def fmt_dollars_k(x):
    try:    x = float(x)
    except: return ""
    xk = x / 1_000.0
    return f"${xk:,.0f}K" if abs(xk) >= 100 else f"${xk:,.1f}K"

def fmt_num(x, decimals=2):
    try:    return f"{float(x):,.{decimals}f}"
    except: return ""


# ── Shared colormap ───────────────────────────────────────────────────────────────
colormap = cm.LinearColormap(
    colors=['#0d0887','#4c02a1','#7e03a8','#aa2395','#cc4778',
            '#e66c5c','#f89540','#fdc527','#f0f921'],
    vmin=vmin_round, vmax=vmax_round
)

cmap_mpl  = plt.get_cmap("plasma")
n_stops   = 9
hex_colors = [mcolors.to_hex(cmap_mpl(i / (n_stops - 1))) for i in range(n_stops)]
gradient_stops = ", ".join(
    f"{hex_colors[i]} {int(100 * i / (n_stops - 1))}%" for i in range(n_stops)
)
tick_values = np.linspace(vmin_round, vmax_round, 4)
tick_labels = [fmt_dollars_k(x) for x in tick_values]


# ── Add this right after grid prep ───────────────────────────────
def tiv_to_color(val):
    try:
        return colormap(float(val))
    except:
        return "#0d0887"

grid_wgs["hex_color"] = grid_wgs["tiv_sum"].apply(tiv_to_color)

# ── Formatted tooltip columns ────────────────────────────────────────────────────
def fmt_int_commas(x):
    try:    return f"{int(round(float(x))):,}"
    except: return ""

def fmt_dollars_k(x):
    try:    x = float(x)
    except: return ""
    xk = x / 1_000.0
    return f"${xk:,.0f}K" if abs(xk) >= 100 else f"${xk:,.1f}K"

def fmt_num(x, decimals=2):
    try:    return f"{float(x):,.{decimals}f}"
    except: return ""

grid_wgs["h3_id_fmt"]          = grid_wgs["h3_id"].astype(str)
grid_wgs["risk_count_fmt"]     = grid_wgs["risk_count"].apply(fmt_int_commas)
grid_wgs["tiv_sum_fmt"]        = grid_wgs["tiv_sum"].apply(fmt_dollars_k)
grid_wgs["tiv_per_sqmi_fmt"]   = grid_wgs["tiv_per_sqmi"].apply(fmt_dollars_k)
grid_wgs["risks_per_sqmi_fmt"] = grid_wgs["risks_per_sqmi"].apply(lambda x: fmt_num(x, 3))



# ── Add this right after the colormap definition ──────────────────
def port_style_fn(x):
    val = x["properties"].get(value_col, 0) or 0
    return {
        "fillColor": colormap(val),
        "color": "white",
        "weight": 0.2,
        "fillOpacity": 0.5
    }

# ── TOOLTIP FOR EXPOSURE ONLY - HIST HAS IT'S OWN TOOLTIP ────────────────────────────────────────────────────────
def style_fn(feature):
    val = feature["properties"].get(value_col, 0) or 0
    return {"fillColor": colormap(val), "color": "white",
            "weight": 0.2, "fillOpacity": 0.5}


# ── Shared legend HTML ────────────────────────────────────────────────────────────
legend_html = f"""
<div id='continuous-legend' style='position:fixed;
     bottom:20px; right:20px; z-index:9999;'>
  <div style="background:white; padding:8px 10px; border-radius:6px;
       box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial,Helvetica,sans-serif;
       font-size:12px; color:#333; min-width:600px;">
    <div style="font-weight:600; margin-bottom:6px;">$TIV (Thousands)</div>
    <div style="height:14px; border-radius:4px;
         background:linear-gradient(to right, {gradient_stops});
         border:1px solid rgba(0,0,0,0.15); margin-bottom:6px;"></div>
    <div style="display:flex; justify-content:space-between; margin-top:2px;">
      {''.join(f"<span style='white-space:nowrap'>{lbl}</span>" for lbl in tick_labels)}
    </div>
  </div>
</div>
"""

weather_legend_html = """
<div style="position:fixed; bottom:160px; right:10px; z-index:1000;
     background:rgba(255,255,255,0.92); border:1px solid #ccc; border-radius:6px;
     padding:10px 14px; font-family:Arial,sans-serif; font-size:12px;
     min-width:170px; box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
  <div style="font-weight:bold; margin-bottom:6px;">📡 NEXRAD Reflectivity</div>
  <div style="display:flex; align-items:center; margin-bottom:3px;">
    <div style="width:100%; height:12px; border-radius:3px;
      background:linear-gradient(to right,
        #00ecec,#019ff4,#0300f4,#02fd02,#01c501,#008e00,
        #fdf802,#e5bc00,#fd9500,#fd0000,#d40000,#bc0000,#f800fd,#9854c6);
    "></div>
  </div>
  <div style="display:flex; justify-content:space-between; color:#555; margin-bottom:10px;">
    <span>Light</span><span>Moderate</span><span>Heavy</span><span>Extreme</span>
  </div>
  <div style="font-weight:bold; margin-bottom:6px;">🌀 Hurricane Category</div>
  <table style="border-collapse:collapse; width:100%;">
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#5ebaff;margin-right:5px;"></span>TD</td><td style="color:#555;">≤ 38 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#00faf4;margin-right:5px;"></span>TS</td><td style="color:#555;">39–73 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffffcc;margin-right:5px;"></span>Cat 1</td><td style="color:#555;">74–95 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffe775;margin-right:5px;"></span>Cat 2</td><td style="color:#555;">96–110 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffc140;margin-right:5px;"></span>Cat 3</td><td style="color:#555;">111–129 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff8f20;margin-right:5px;"></span>Cat 4</td><td style="color:#555;">130–156 mph</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff6060;margin-right:5px;"></span>Cat 5</td><td style="color:#555;">≥ 157 mph</td></tr>
  </table>
</div>
"""


flood_legend_html = """
<div id="fema-legend-wrapper" style="position:fixed; bottom:40px; left:740px;
     z-index:1000; font-family:'Aptos Display',Arial,sans-serif; font-size:12px;">
  <!-- Collapse toggle bar -->
  <div id="fema-legend-toggle"
       onclick="
         var body = document.getElementById('fema-legend-body');
         var arrow = document.getElementById('fema-legend-arrow');
         var expanded = body.style.display !== 'none';
         body.style.display = expanded ? 'none' : 'block';
         arrow.textContent = expanded ? '▶' : '▼';
       "
       style="background:#02473b; color:#fff; border:2px solid #1a9f9c;
              border-radius:6px 6px 0 0; padding:6px 12px; cursor:pointer;
              display:flex; align-items:center; gap:8px; font-weight:700;
              font-size:12px; user-select:none;
              box-shadow:0 2px 6px rgba(0,0,0,0.35);">
    🌊 FEMA Flood Zones
    <span id="fema-legend-arrow" style="margin-left:auto; font-size:10px;">▶</span>
  </div>
  <!-- Legend body -->
  <div id="fema-legend-body"
       style="display:none; background:rgba(255,255,255,0.95); border:2px solid #1a9f9c;
              border-top:none; border-radius:0 0 6px 6px; padding:8px 12px;
              box-shadow:0 2px 6px rgba(0,0,0,0.25); min-width:185px;">
    <table style="border-collapse:collapse; width:100%;">
      <tr><td style="padding:2px 0;">
        <span style="display:inline-block;width:12px;height:12px;
             border-radius:2px;background:#313695;margin-right:6px;
             vertical-align:middle;"></span>
        <b>VE/V</b></td>
        <td style="color:#555; padding-left:8px;">Coastal High Velocity</td></tr>
      <tr><td style="padding:2px 0;">
        <span style="display:inline-block;width:12px;height:12px;
             border-radius:2px;background:#4575b4;margin-right:6px;
             vertical-align:middle;"></span>
        <b>AE/A</b></td>
        <td style="color:#555; padding-left:8px;">100yr Floodplain</td></tr>
      <tr><td style="padding:2px 0;">
        <span style="display:inline-block;width:12px;height:12px;
             border-radius:2px;background:#abd9e9;margin-right:6px;
             vertical-align:middle;"></span>
        <b>AH/AO</b></td>
        <td style="color:#555; padding-left:8px;">Shallow Flooding</td></tr>
    </table>
  </div>
</div>
"""

# ── SCS Legend ──────────────────────────────────────────────────────────
scs_legend_html = """
<div style="position:fixed; bottom:400px; left:20px; z-index:1000;
     background:rgba(255,255,255,0.92); border:1px solid #ccc; border-radius:6px;
     padding:10px 14px; font-family:Arial,sans-serif; font-size:12px;
     min-width:160px; box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
  <div style="font-weight:bold; margin-bottom:6px;">🌪 Tornado EF Scale</div>
  <table style="border-collapse:collapse; width:100%; margin-bottom:8px;">
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#ffffb2;margin-right:5px;"></span>EF0</td><td style="color:#555;">65–85 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#fecc5c;margin-right:5px;"></span>EF1</td><td style="color:#555;">86–110 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#fd8d3c;margin-right:5px;"></span>EF2</td><td style="color:#555;">111–135 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#f03b20;margin-right:5px;"></span>EF3</td><td style="color:#555;">136–165 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#bd0026;margin-right:5px;"></span>EF4</td><td style="color:#555;">166–200 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#6a0005;margin-right:5px;"></span>EF5</td><td style="color:#555;">> 200 mph</td></tr>
  </table>
  <div style="font-weight:bold; margin-bottom:6px;">🌨 Hail Size</div>
  <table style="border-collapse:collapse; width:100%; margin-bottom:8px;">
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffeda0;margin-right:5px;"></span>1.0"</td><td style="color:#555;">Quarter</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#feb24c;margin-right:5px;"></span>1.75"</td><td style="color:#555;">Golf ball</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f03b20;margin-right:5px;"></span>2.0"</td><td style="color:#555;">Tennis ball</td></tr>
    <tr><td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#bd0026;margin-right:5px;"></span>4.0"+</td><td style="color:#555;">Baseball+</td></tr>
  </table>
  <div style="font-weight:bold; margin-bottom:6px;">💨 Wind Speed</div>
  <table style="border-collapse:collapse; width:100%;">
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#4575b4;margin-right:5px;"></span>65–74 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#08306b;margin-right:5px;"></span>75–99 mph</td></tr>
    <tr><td><span style="display:inline-block;width:30px;height:4px;background:#c51b8a;margin-right:5px;"></span>100+ mph</td></tr>
  </table>
</div>
"""

# ── ruler ──────────────────────────────────────────────────────────
class RulerControl(MacroElement):
    def __init__(self, position="topright", unit="mi"):
        super().__init__()
        self._name = "RulerControl"
        self.position = position
        self.unit = unit
        self._template = Template("""
        {% macro header(this, kwargs) %}
        <link rel="stylesheet" href="https://unpkg.com/leaflet-ruler@1.0.0/src/leaflet-ruler.css"/>
        <script src="https://unpkg.com/leaflet-ruler@1.0.0/src/leaflet-ruler.js"></script>
        {% endmacro %}

        {% macro script(this, kwargs) %}
        L.control.ruler({
            position: '{{ this.position }}',
            lengthUnit: {
                factor: 0.621371,   // meters -> miles
                display: 'mi',
                decimal: 2,
                label: 'Distance:'
            },
            angleUnit: {
                display: '&deg;',
                decimal: 2,
                factor: null,
                label: 'Bearing:'
            }
        }).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)

# ── Live HU Legend ──────────────────────────────────────────────────────────
class NHCLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var nhcDiv = document.createElement('div');
            nhcDiv.className = 'nhc-legend';
            nhcDiv.style.position = 'absolute';
            nhcDiv.style.bottom = '350px';
            nhcDiv.style.left = '10px';
            nhcDiv.style.zIndex = '9999';
            nhcDiv.style.background = 'white';
            nhcDiv.style.padding = '8px 10px';
            nhcDiv.style.borderRadius = '6px';
            nhcDiv.style.boxShadow = '0 1px 4px rgba(0,0,0,0.4)';
            nhcDiv.style.fontFamily = 'Arial, sans-serif';
            nhcDiv.style.fontSize = '12px';
            nhcDiv.innerHTML = `
                <div id="nhc-legend-header" style="cursor:pointer;font-weight:bold;color:#02473b;">
                    <span id="nhc-legend-arrow">&#9654;</span> NHC Storm Layers
                </div>
                <div id="nhc-legend-body" style="display:none; margin-top:6px;">
                    <div style="margin-bottom:4px;"><b>Cone / Track</b></div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#ff6600;opacity:0.4;border:1px solid #cc3300;margin-right:6px;"></span>Forecast cone</div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#ff6600;border-radius:50%;border:1px solid #cc3300;margin-right:6px;"></span>Track point</div>
                    <div style="margin:6px 0 4px;"><b>Watches / Warnings</b></div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#ff0000;margin-right:6px;"></span>Hurricane Warning</div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#ffb3d9;margin-right:6px;"></span>Hurricane Watch</div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#0000ff;margin-right:6px;"></span>Tropical Storm Warning</div>
                    <div><span style="display:inline-block;width:12px;height:12px;background:#ffff00;margin-right:6px;"></span>Tropical Storm Watch</div>
                </div>
            `;
            document.querySelector('.leaflet-container').appendChild(nhcDiv);
            var header = nhcDiv.querySelector('#nhc-legend-header');
            var body = nhcDiv.querySelector('#nhc-legend-body');
            var arrow = nhcDiv.querySelector('#nhc-legend-arrow');
            header.onclick = function() {
                if (body.style.display === 'none') {
                    body.style.display = 'block';
                    arrow.innerHTML = '&#9660;';
                } else {
                    body.style.display = 'none';
                    arrow.innerHTML = '&#9654;';
                }
            };
        })();
        {% endmacro %}
        """)


        
# ── Cone Impact ──────────────────────────────────────────────────────────
class ConeImpactPanel(MacroElement):
    def __init__(self, impacts):
        super().__init__()
        self._name = "ConeImpactPanel"
        self.impacts = impacts
        rows = ""
        for imp in impacts:
            states_str = ", ".join(imp["states"]) if imp["states"] else "none"
            rows += f"""
                <div style="margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid #eee;">
                    <div style="font-weight:bold; color:#02473b;">{imp['storm_name']}</div>
                    <div>States: {states_str}</div>
                    <div><a href="{imp['csv_file']}" download style="color:#1a9f9c; text-decoration:underline;">
                        Download {imp['storm_name']} Cone (CSV)
                    </a></div>
                </div>
            """
        self.rows_html = rows
        self._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var coneDiv = document.createElement('div');
            coneDiv.className = 'cone-impact-panel';
            coneDiv.style.position = 'absolute';
            coneDiv.style.top = '10px';
            coneDiv.style.left = '60px';
            coneDiv.style.zIndex = '9999';
            coneDiv.style.background = 'white';
            coneDiv.style.padding = '8px 10px';
            coneDiv.style.borderRadius = '6px';
            coneDiv.style.boxShadow = '0 1px 4px rgba(0,0,0,0.4)';
            coneDiv.style.fontFamily = 'Arial, sans-serif';
            coneDiv.style.fontSize = '12px';
            coneDiv.style.maxWidth = '220px';
            coneDiv.innerHTML = `
                <div id="cone-impact-header" style="cursor:pointer;font-weight:bold;color:#02473b;">
                    <span id="cone-impact-arrow">&#9654;</span> Cone impact (states/cities)
                </div>
                <div id="cone-impact-body" style="display:none; margin-top:6px;">
                    {{ this.rows_html }}
                </div>
            `;
            document.querySelector('.leaflet-container').appendChild(coneDiv);
            var header = coneDiv.querySelector('#cone-impact-header');
            var body = coneDiv.querySelector('#cone-impact-body');
            var arrow = coneDiv.querySelector('#cone-impact-arrow');
            header.onclick = function() {
                if (body.style.display === 'none') {
                    body.style.display = 'block';
                    arrow.innerHTML = '&#9660;';
                } else {
                    body.style.display = 'none';
                    arrow.innerHTML = '&#9654;';
                }
            };
        })();
        {% endmacro %}
        """)



# ── MacroElement classes ──────────────────────────────────────────────────────────
class PortfolioLayers(MacroElement):
    def __init__(self):
        super().__init__()
        self._name = "PortfolioLayers"
        self._template = Template("""
        {% macro html(this, kwargs) %}
        <script>
        (function() {
            var base = window.location.href.replace(/\/[^\/]*$/, "/");
            var portColors = {};  // cache hex_color per feature

            function tooltipContent(p) {
                return '<div style="font-family:arial;font-size:12px;padding:10px;background:white;color:black;">'
                    + '<b>Hex ID:</b> '      + (p.h3_id_fmt          || '') + '<br>'
                    + '<b>Risks:</b> '       + (p.risk_count_fmt     || '') + '<br>'
                    + '<b>TIV:</b> '         + (p.tiv_sum_fmt        || '') + '<br>'
                    + '<b>TIV / sq mi:</b> ' + (p.tiv_per_sqmi_fmt   || '') + '<br>'
                    + '<b>Risks / sq mi:</b>'+ (p.risks_per_sqmi_fmt || '') + '</div>';
            }

            var attempts = 0;
            var poller = setInterval(function() {
                attempts++;
                var maps = Object.values(window).filter(function(v) {
                    return v && typeof v === "object" && v._container && typeof v.on === "function";
                });
                if (maps.length > 0) {
                    clearInterval(poller);
                    var leafMap = maps[0];
                    var layerControl = null;

                    // Find existing LayerControl
                    leafMap.eachLayer(function(l) {});
                    Object.values(leafMap._controlCorners || {}).forEach(function(corner) {
                        // LayerControl attaches to map controls not layers
                    });

                    // Build a fresh LayerControl
                    layerControl = L.control.layers(null, null, {collapsed: false}).addTo(leafMap);

                    fetch(base + "ports_data.geojson")
                        .then(function(r) { return r.json(); })
                        .then(function(data) {
                            Object.keys(data).forEach(function(port) {
                                var fc = data[port];
                                var layer = L.geoJSON(fc, {
                                    style: function(feature) {
                                        return {
                                            fillColor:   feature.properties.hex_color || "#0d0887",
                                            color:       "white",
                                            weight:      0.2,
                                            fillOpacity: 0.5
                                        };
                                    },
                                    onEachFeature: function(feature, layer) {
                                        var p = feature.properties;
                                        layer.bindTooltip(tooltipContent(p),
                                            {sticky: true, opacity: 1.0});
                                    }
                                });
                                layerControl.addOverlay(layer, "Port: " + port);
                                layer.addTo(leafMap);
                            });
                        })
                        .catch(function(err) {
                            console.error("Failed to load ports_data.geojson:", err);
                        });
                }
                if (attempts > 40) clearInterval(poller);
            }, 250);
        })();
        </script>
        {% endmacro %}
        """)

class LayerControlStyle(MacroElement):
    def __init__(self):
        super().__init__()
        self._name = "LayerControlStyle"
        self._template = Template("""
        {% macro html(this, kwargs) %}
        <style>
            .leaflet-control-layers-toggle {
                display: none !important;
            }
            .leaflet-control-layers {
                padding: 6px 10px !important;
                border-radius: 6px !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
            }
            .leaflet-control-layers-expanded {
                padding: 6px 10px !important;
            }

            /* Hide layer reorder arrows */
            .leaflet-control-layers-overlays .leaflet-control-layers-separator,
            .leaflet-control-layers-overlays button {
                display: none !important;
            }
        </style>
        {% endmacro %}
        """)



class StateBoundaries(MacroElement):
    def __init__(self):
        super().__init__()
        self._name = "StateBoundaries"
        self._template = Template("""
        {% macro html(this, kwargs) %}
        <script>
        (function() {
            var base = window.location.href.replace(/\/[^\/]*$/, "/");
            var attempts = 0;
            var poller = setInterval(function() {
                attempts++;
                var maps = Object.values(window).filter(function(v) {
                    return v && typeof v === "object" && v._container && typeof v.on === "function";
                });
                if (maps.length > 0) {
                    clearInterval(poller);
                    var leafMap = maps[0];
                    fetch(base + "states.geojson")
                        .then(function(r) { return r.json(); })
                        .then(function(data) {
                            L.geoJSON(data, {
                                style: function() {
                                    return {
                                        fillColor: "none",
                                        color: "#888888",
                                        weight: 1,
                                        fillOpacity: 0
                                    };
                                }
                            }).addTo(leafMap);
                        })
                        .catch(function(err) {
                            console.error("Failed to load states.geojson:", err);
                        });
                }
                if (attempts > 40) clearInterval(poller);
            }, 250);
        })();
        </script>
        {% endmacro %}
        """)



class FloatLegend(MacroElement):
    def __init__(self, html):
        super().__init__()
        self._name = "FloatLegend"
        self.html  = html
        self._template = Template("""
            {% macro html(this, kwargs) %}{{ this.html | safe }}{% endmacro %}
        """)

class RadarRefresh(MacroElement):
    def __init__(self, interval_ms=300000):
        super().__init__()
        self._name        = "RadarRefresh"
        self.interval_ms  = interval_ms
        self._template    = Template("""
        {% macro html(this, kwargs) %}
        <script>
        (function() {
            setInterval(function() {
                Object.values(window).forEach(function(obj) {
                    if (obj && obj._url &&
                        obj._url.includes("mesonet.agron.iastate.edu")) {
                        obj.setUrl(obj._url.split("&_ts=")[0] + "&_ts=" + Date.now());
                    }
                });
            }, {{ this.interval_ms }});
        })();
        </script>
        {% endmacro %}
        """)

class SelectionSummary(MacroElement):
    def __init__(self, geojson_data):
        super().__init__()
        self._name        = "SelectionSummary"
        self.geojson_data = geojson_data
        self._template    = Template("""
        {% macro html(this, kwargs) %}
        <div id="selection-box" style="
            position:fixed; top:10px; left:50%; transform:translateX(-50%);
            z-index:9999; background:rgba(255,255,255,0.96); border:1px solid #ccc;
            border-radius:6px; padding:10px 18px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);
            display:none; min-width:320px; text-align:center;">
            <b>📐 Selection Summary</b>
            <span style="float:right;cursor:pointer;color:#999;"
                  onclick="document.getElementById('selection-box').style.display='none'">✕</span>
            <hr style="margin:6px 0;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr><td style="text-align:left;color:#555;">Hexagons selected</td>
                    <td id="sel-count" style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="text-align:left;color:#555;">Total Risks</td>
                    <td id="sel-risks" style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="text-align:left;color:#555;">Total TIV</td>
                    <td id="sel-tiv"   style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="text-align:left;color:#555;">Avg TIV / Hex</td>
                    <td id="sel-avg"   style="text-align:right;font-weight:bold;">—</td></tr>
            </table>
        </div>
        <script>
        (function() {
            var hexData = {{ this.geojson_data }};
            function ptInPoly(pt, poly) {
                var x=pt[0],y=pt[1],inside=false;
                for(var k=0;k<poly.length;k++){
                    var ring=poly[k];
                    for(var i=0,j=ring.length-1;i<ring.length;j=i++){
                        var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];
                        if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))inside=!inside;
                    }
                }
                return inside;
            }
            function centroid(coords){
                var ring=coords[0],x=0,y=0;
                for(var i=0;i<ring.length;i++){x+=ring[i][0];y+=ring[i][1];}
                return[x/ring.length,y/ring.length];
            }
            function fmtInt(n){return n.toLocaleString();}
            function fmtTIV(n){
                var k=n/1000;
                if(k>=1e6) return"$"+(k/1e6).toFixed(2)+"B";
                else if(k>=1000) return"$"+(k/1000).toFixed(1)+"M";
                else return"$"+k.toFixed(1)+"K";
            }
            function onDrawn(e){
                var geo=e.layer.toGeoJSON().geometry;
                var sel=geo.coordinates;
                if(geo.type==="MultiPolygon")sel=sel[0];
                var hc=0,rs=0,tv=0;
                hexData.features.forEach(function(f){
                    var g=f.geometry,p=f.properties,ctr;
                    if(g.type==="Polygon")ctr=centroid(g.coordinates);
                    else if(g.type==="MultiPolygon")ctr=centroid(g.coordinates[0]);
                    else return;
                    if(ptInPoly(ctr,sel)){
                        hc++;rs+=parseFloat(p.risk_count)||0;tv+=parseFloat(p.tiv_sum)||0;
                    }
                });
                document.getElementById("sel-count").innerText=fmtInt(hc);
                document.getElementById("sel-risks").innerText=fmtInt(rs);
                document.getElementById("sel-tiv").innerText=fmtTIV(tv);
                document.getElementById("sel-avg").innerText=hc>0?fmtTIV(tv/hc):"—";
                document.getElementById("selection-box").style.display="block";
            }
            var attempts=0;
            var poller=setInterval(function(){
                attempts++;
                var maps=Object.values(window).filter(function(v){
                    return v&&typeof v==="object"&&v._container&&typeof v.on==="function";
                });
                if(maps.length>0){
                    clearInterval(poller);
                    maps.forEach(function(mp){
                        mp.on("draw:created",onDrawn);
                        mp.on("draw:deleted",function(){
                            document.getElementById("selection-box").style.display="none";
                        });
                    });
                }
                if(attempts>40)clearInterval(poller);
            },250);
        })();
        </script>
        {% endmacro %}
        """)

class FloodRegionSelector(MacroElement):
    def __init__(self):
        super().__init__()
        self._name     = "FloodRegionSelector"
        self._template = Template("""
        {% macro html(this, kwargs) %}

        <div id="flood-ctrl" style="
            position:fixed; bottom:10px; left:490px; z-index:9999;
            background:rgba(255,255,255,0.96); border:1px solid #ccc;
            border-radius:6px; padding:10px 14px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);
            min-width:200px; max-width:220px;">

            <div style="display:flex;align-items:center;justify-content:space-between;">
                <b>🌊 FEMA Flood Zones</b>
                <span id="flood-toggle-btn" onclick="floodTogglePanel()"
                style="cursor:pointer;font-size:11px;color:#0066cc;">▶ show</span>
            </div>

            <div id="flood-panel" style="display:none;">
                <hr style="margin:6px 0;">

                <!-- Zone type filter -->
                <div style="font-size:11px;color:#555;margin-bottom:4px;">
                    Show zones:
                </div>
                <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;">
                    <label style="font-size:11px;display:flex;align-items:center;gap:2px;cursor:pointer;">
                        <input type="checkbox" class="flood-zone-chk" value="AE,A,A99" checked>
                        <span style="display:inline-block;width:10px;height:10px;
                              background:#4575b4;border-radius:2px;"></span>100yr
                    </label>
                    <label style="font-size:11px;display:flex;align-items:center;gap:2px;cursor:pointer;">
                        <input type="checkbox" class="flood-zone-chk" value="VE,V" checked>
                        <span style="display:inline-block;width:10px;height:10px;
                              background:#313695;border-radius:2px;"></span>Coastal
                    </label>
                    <label style="font-size:11px;display:flex;align-items:center;gap:2px;cursor:pointer;">
                        <input type="checkbox" class="flood-zone-chk" value="AH,AO" checked>
                        <span style="display:inline-block;width:10px;height:10px;
                              background:#abd9e9;border-radius:2px;"></span>Shallow
                    </label>
                </div>

                <!-- Region checkboxes -->
                <div style="font-size:11px;color:#555;margin-bottom:4px;">
                    Load regions:
                </div>
                <div id="flood-region-list" style="max-height:200px;overflow-y:auto;">
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Northeast">Northeast
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Mid_Atlantic">Mid-Atlantic
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Southeast">Southeast
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="FL_North">Florida North
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="FL_South">Florida South
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Gulf">Gulf
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Upper_Midwest">Upper Midwest
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Central_Midwest">Central Midwest
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Lower_Midwest">Lower Midwest
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Mountain">Mountain
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Southwest">Southwest
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="California">California
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;
                           padding:2px 0;font-size:12px;cursor:pointer;">
                        <input type="checkbox" class="flood-rgn-chk"
                               data-region="Northwest">Northwest
                    </label>
                </div>

                <div style="margin-top:8px;display:flex;gap:6px;">
                    <button onclick="floodSelectAll(true)" style="
                        flex:1;padding:3px;font-size:11px;
                        border:1px solid #ccc;border-radius:4px;
                        cursor:pointer;background:#f5f5f5;">All</button>
                    <button onclick="floodSelectAll(false)" style="
                        flex:1;padding:3px;font-size:11px;
                        border:1px solid #ccc;border-radius:4px;
                        cursor:pointer;background:#f5f5f5;">None</button>
                </div>

                <div id="flood-status" style="margin-top:6px;font-size:11px;
                     color:#555;border-top:1px solid #eee;padding-top:6px;"></div>
            </div>
        </div>

        <script>
        (function() {
            var SERVER   = "http://localhost:8000";
            var FLOOD_TILES = SERVER + "/Flood/flood_tiles/";
            var base     = window.location.href.replace(/\/[^\/]*$/, "/");
            var leafMap  = null;
            var loaded   = {};   // region -> L.geoJSON layer
            var active   = {};   // region -> bool

            // ── Active zone filter ────────────────────────────────────────────────
            function getActiveZones() {
                var zones = [];
                document.querySelectorAll(".flood-zone-chk:checked").forEach(function(chk) {
                    chk.value.split(",").forEach(function(z) { zones.push(z.trim()); });
                });
                return zones;
            }

            // ── Style function ────────────────────────────────────────────────────
            function floodStyle(feature) {
                return {
                    fillColor:   feature.properties.color || "#4575b4",
                    color:       "#2c5f8a",
                    weight:      0.3,
                    fillOpacity: 0.45,
                };
            }

            // ── Filter visible features by zone ──────────────────────────────────
            function refreshRegion(region) {
                if (!loaded[region] || !leafMap) return;
                var zones = getActiveZones();
                loaded[region].clearLayers();
                loaded[region].options.filter = function(feature) {
                    return zones.indexOf(feature.properties.FLD_ZONE) !== -1;
                };
                // Reload features with new filter
                var data = loaded[region]._data;
                if (data) {
                    loaded[region].addData(data);
                }
            }

            // ── Load a region ─────────────────────────────────────────────────────
            function loadRegion(region) {
                if (loaded[region]) {
                    loaded[region].addTo(leafMap);
                    active[region] = true;
                    updateStatus();
                    return;
                }

                updateStatus("Loading " + region + "...");

                fetch(FLOOD_TILES + "flood_" + region + ".json")
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        var zones = getActiveZones();
                        var layer = L.geoJSON(data, {
                            style: floodStyle,
                            filter: function(feature) {
                                return zones.indexOf(feature.properties.FLD_ZONE) !== -1;
                            },
                            onEachFeature: function(feature, lyr) {
                                var p = feature.properties;
                                lyr.bindTooltip(
                                    "<b>" + (p.zone_lbl || p.FLD_ZONE) + "</b>",
                                    {sticky:true,
                                     style:"background:white;font-family:arial;"
                                           +"font-size:12px;padding:6px;"}
                                );
                            }
                        });
                        layer._data = data;   // cache raw data for zone filtering
                        loaded[region] = layer;
                        active[region] = true;
                        layer.addTo(leafMap);
                        updateStatus();
                    })
                    .catch(function(err) {
                        console.error("Failed to load flood_" + region + ":", err);
                        // Uncheck the box if load failed
                        document.querySelectorAll(".flood-rgn-chk").forEach(function(chk) {
                            if(chk.dataset.region === region) chk.checked = false;
                        });
                        active[region] = false;
                        updateStatus("Error loading " + region);
                    });
}

            // ── Remove a region ───────────────────────────────────────────────────
            function removeRegion(region) {
                if (loaded[region]) {
                    loaded[region].remove();
                }
                active[region] = false;
                updateStatus();
            }

            // ── Select all / none ─────────────────────────────────────────────────
            window.floodSelectAll = function(state) {
                document.querySelectorAll(".flood-rgn-chk").forEach(function(chk) {
                    chk.checked = state;
                    var region = chk.dataset.region;
                    if (state) loadRegion(region);
                    else removeRegion(region);
                });
            };

            // ── Collapse panel ────────────────────────────────────────────────────
            window.floodTogglePanel = function() {
                var panel = document.getElementById("flood-panel");
                var btn   = document.getElementById("flood-toggle-btn");
                if (panel.style.display === "none") {
                    panel.style.display = "block";
                    btn.innerText = "▼ hide";
                } else {
                    panel.style.display = "none";
                    btn.innerText = "▶ show";
                }
            };

            // ── Status line ───────────────────────────────────────────────────────
            function updateStatus(msg) {
                var n       = Object.values(active).filter(Boolean).length;
                var loading = Object.keys(active).filter(function(r) {
                    return active[r] && !loaded[r];
                }).length;
                
                var el = document.getElementById("flood-status");
                if (msg) {
                    el.innerHTML = msg;
                } else if (n === 0) {
                    el.innerHTML = "No regions loaded — check a region above";
                } else {
                    el.innerHTML = "<b>" + n + "</b> region" + (n!==1?"s":"") + " loaded";
                }
            }

            // ── Region checkbox handler ───────────────────────────────────────────
            document.querySelectorAll(".flood-rgn-chk").forEach(function(chk) {
                chk.addEventListener("change", function(e) {
                    if (e.target.checked) loadRegion(e.target.dataset.region);
                    else removeRegion(e.target.dataset.region);
                });
            });

            // ── Zone filter handler ───────────────────────────────────────────────
            document.querySelectorAll(".flood-zone-chk").forEach(function(chk) {
                chk.addEventListener("change", function() {
                    Object.keys(loaded).forEach(function(region) {
                        if (active[region]) refreshRegion(region);
                    });
                });
            });

            // ── Wait for Leaflet ──────────────────────────────────────────────────
            var attempts=0, poller=setInterval(function() {
                attempts++;
                var maps=Object.values(window).filter(function(v) {
                    return v&&typeof v==="object"&&v._container&&typeof v.on==="function";
                });
                if(maps.length>0){
                    clearInterval(poller);
                    leafMap=maps[0];
                    updateStatus();    // ← initialize status once map is ready
                }
                if(attempts>40)clearInterval(poller);
            },250);

        })();
        </script>

        {% endmacro %}
        """)


class SCSYearSelector(MacroElement):
    def __init__(self, year_min, year_max):
        super().__init__()
        self._name    = "SCSYearSelector"
        self.year_min = year_min
        self.year_max = year_max
        self._template = Template("""
        {% macro html(this, kwargs) %}

        <div id="scs-ctrl" style="
            position:fixed; bottom:10px; left:280px; z-index:9999;
            background:rgba(255,255,255,0.96); border:1px solid #ccc;
            border-radius:6px; padding:10px 14px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);
            min-width:200px; max-width:220px;">
            <b>⛈ SCS Events</b>
            <hr style="margin:6px 0;">
            <label style="font-size:12px;color:#555;">Select Year</label><br>
            <select id="scs-year" style="width:100%;margin:4px 0 8px 0;
                padding:4px;font-size:13px;border:1px solid #ccc;border-radius:4px;">
                <option value="">— choose year —</option>
                {% for yr in range(this.year_max, this.year_min - 1, -1) %}
                <option value="{{ yr }}">{{ yr }}</option>
                {% endfor %}
            </select>

            <!-- Toggles -->
            <div style="font-size:12px;">
                <label style="display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;">
                    <input type="checkbox" id="scs-torn-chk" checked>
                    <span style="display:inline-block;width:20px;height:3px;
                          background:#f03b20;margin-right:2px;"></span>
                    Tornadoes
                    <span id="scs-torn-count" style="color:#888;font-size:11px;margin-left:auto;">—</span>
                </label>
                <label style="display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;">
                    <input type="checkbox" id="scs-hail-chk" checked>
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                          background:#bd0026;margin-right:2px;"></span>
                    Hail ≥ 1"
                    <span id="scs-hail-count" style="color:#888;font-size:11px;margin-left:auto;">—</span>
                </label>
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                    <input type="checkbox" id="scs-wind-chk" checked>
                    <span style="display:inline-block;width:20px;height:3px;
                          background:#08306b;margin-right:2px;"></span>
                    Wind ≥ 65 mph
                    <span id="scs-wind-count" style="color:#888;font-size:11px;margin-left:auto;">—</span>
                </label>
            </div>

            <div style="margin-top:8px;display:flex;gap:6px;">
                <button onclick="scsFitBounds()" style="
                    flex:1;padding:3px;font-size:11px;
                    border:1px solid #ccc;border-radius:4px;
                    cursor:pointer;background:#e8f4fd;color:#0066cc;">Fit</button>
                <button onclick="scsClear()" style="
                    flex:1;padding:3px;font-size:11px;
                    border:1px solid #ccc;border-radius:4px;
                    cursor:pointer;background:#f5f5f5;">Clear</button>
            </div>

            <div id="scs-summary" style="margin-top:8px;font-size:11px;
                color:#555;border-top:1px solid #eee;padding-top:6px;"></div>
        </div>

        <script>
        (function() {
            var base    = window.location.href.replace(/\/[^\/]*$/, "/");
            var SCS     = null;
            var leafMap = null;
            var tornLayer = null, hailLayer = null, windLayer = null;

            // ── Color helpers ─────────────────────────────────────────────────────
            function efColor(mag) {
                return {0:"#ffffb2",1:"#fecc5c",2:"#fd8d3c",
                        3:"#f03b20",4:"#bd0026",5:"#6a0005"}[mag] || "#aaaaaa";
            }
            // Yellow-to-red, 4-class — matches legend breaks (1.0" / 1.75" / 2.0" / 4.0"+)
            function hailColor(s) {
                if(s>=4.0)return"#bd0026";if(s>=2.0)return"#f03b20";
                if(s>=1.75)return"#feb24c";return"#ffeda0";
            }
            // Blue -> dark blue -> magenta, 3-class — data is pre-filtered to >= 65 mph
            function windColor(m) {
                if(m>=100)return"#c51b8a";if(m>=75)return"#08306b";
                return"#4575b4";
            }

            // ── Clear all layers ──────────────────────────────────────────────────
            window.scsClear = function() {
                if(tornLayer){tornLayer.remove();tornLayer=null;}
                if(hailLayer){hailLayer.remove();hailLayer=null;}
                if(windLayer){windLayer.remove();windLayer=null;}
                document.getElementById("scs-summary").innerHTML = "";
                document.getElementById("scs-torn-count").innerText = "—";
                document.getElementById("scs-hail-count").innerText = "—";
                document.getElementById("scs-wind-count").innerText = "—";
            };

            // ── Fit bounds ────────────────────────────────────────────────────────
            window.scsFitBounds = function() {
                var bounds = [];
                [tornLayer, hailLayer, windLayer].forEach(function(l) {
                    if(l) {
                        try { bounds.push(l.getBounds()); } catch(e) {}
                    }
                });
                if(bounds.length > 0) {
                    var combined = bounds[0];
                    bounds.forEach(function(b) { combined.extend(b); });
                    leafMap.fitBounds(combined.pad(0.05));
                }
            };

            // ── Draw year ─────────────────────────────────────────────────────────
            function drawYear(year) {
                scsClear();
                var d = SCS[year];
                if(!d) return;

                var showTorn = document.getElementById("scs-torn-chk").checked;
                var showHail = document.getElementById("scs-hail-chk").checked;
                var showWind = document.getElementById("scs-wind-chk").checked;

                // Tornadoes
                if(showTorn && d.tornadoes.length) {
                    tornLayer = L.layerGroup();
                    d.tornadoes.forEach(function(t) {
                        if(!t.lat1||!t.lon1) return;
                        var color = efColor(t.mag);
                        var coords = (t.lat2 && t.lon2 && (t.lat2!==0||t.lon2!==0))
                            ? [[t.lat1,t.lon1],[t.lat2,t.lon2]]
                            : [[t.lat1,t.lon1],[t.lat1+0.01,t.lon1+0.01]];
                        L.polyline(coords, {color:color, weight:2, opacity:0.8})
                        .bindTooltip(
                            "<b>EF"+(t.mag>=0?t.mag:"?")+"</b> — "+t.st+"<br>"+
                            year+"-"+String(t.mo).padStart(2,"0")+"-"+String(t.dy).padStart(2,"0")+"<br>"+
                            "Length: "+t.len+" mi &nbsp; Width: "+t.wid+" yds",
                            {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:6px;"}
                        ).addTo(tornLayer);
                    });
                    tornLayer.addTo(leafMap);
                }

                // Hail
                if(showHail && d.hail.length) {
                    hailLayer = L.layerGroup();
                    d.hail.forEach(function(h) {
                        if(!h.lat||!h.lon) return;
                        L.circleMarker([h.lat,h.lon], {
                            radius:4, fillColor:hailColor(h.mag),
                            color:hailColor(h.mag), weight:0.5,
                            fillOpacity:0.7
                        }).bindTooltip(
                            "<b>Hail: "+h.mag+'"</b><br>'+
                            year+"-"+String(h.mo).padStart(2,"0")+"-"+String(h.dy).padStart(2,"0")+"<br>"+
                            "State: "+h.st,
                            {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:6px;"}
                        ).addTo(hailLayer);
                    });
                    hailLayer.addTo(leafMap);
                }

                // Wind
                if(showWind && d.wind.length) {
                    windLayer = L.layerGroup();
                    d.wind.forEach(function(w) {
                        if(!w.lat||!w.lon) return;
                        L.circleMarker([w.lat,w.lon], {
                            radius:3, fillColor:windColor(w.mag),
                            color:windColor(w.mag), weight:0.5,
                            fillOpacity:0.6
                        }).bindTooltip(
                            "<b>Wind: "+w.mag+" mph</b><br>"+
                            year+"-"+String(w.mo).padStart(2,"0")+"-"+String(w.dy).padStart(2,"0")+"<br>"+
                            "State: "+w.st,
                            {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:6px;"}
                        ).addTo(windLayer);
                    });
                    windLayer.addTo(leafMap);
                }

                // Update counts
                document.getElementById("scs-torn-count").innerText =
                    d.tornadoes.length.toLocaleString();
                document.getElementById("scs-hail-count").innerText =
                    d.hail.length.toLocaleString();
                document.getElementById("scs-wind-count").innerText =
                    d.wind.length.toLocaleString();

                document.getElementById("scs-summary").innerHTML =
                    "<b>"+(d.tornadoes.length+d.hail.length+d.wind.length).toLocaleString()
                    +"</b> total events";
            }

            // ── Checkbox toggles ──────────────────────────────────────────────────
            ["torn","hail","wind"].forEach(function(t) {
                document.getElementById("scs-"+t+"-chk")
                    .addEventListener("change", function() {
                        var yr = document.getElementById("scs-year").value;
                        if(yr) drawYear(yr);
                    });
            });

            // ── Year dropdown ─────────────────────────────────────────────────────
            document.getElementById("scs-year")
                .addEventListener("change", function(e) {
                    if(e.target.value) drawYear(e.target.value);
                    else scsClear();
                });

            // ── Fetch SCS data ────────────────────────────────────────────────────
            fetch(base + "scs_data.json")
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    SCS = data;
                    console.log("SCS data loaded — years:", Object.keys(SCS).length);
                })
                .catch(function(err) {
                    console.error("Failed to load scs_data.json:", err);
                });

            // ── Wait for Leaflet ──────────────────────────────────────────────────
            var attempts=0, poller=setInterval(function() {
                attempts++;
                var maps=Object.values(window).filter(function(v) {
                    return v&&typeof v==="object"&&v._container&&typeof v.on==="function";
                });
                if(maps.length>0){clearInterval(poller);leafMap=maps[0];}
                if(attempts>40)clearInterval(poller);
            },250);

        })();
        </script>
        {% endmacro %}
        """)






class HurdatYearSelector(MacroElement):
    def __init__(self, storms_json, year_min, year_max):
        super().__init__()
        self._name       = "HurdatYearSelector"
        self.storms_json = storms_json
        self.year_min    = year_min
        self.year_max    = year_max
        self._template   = Template("""
        {% macro html(this, kwargs) %}
        <div id="hurdat-ctrl" style="
            position:fixed; bottom:10px; left:10px; z-index:9999;
            background:rgba(255,255,255,0.95); border:1px solid #ccc;
            border-radius:6px; padding:10px 14px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2); min-width:220px;">
            <b>🌀 Atlantic Storm Season</b>
            <hr style="margin:6px 0;">
            <label style="font-size:12px;color:#555;">Select Year</label><br>
            <select id="hurdat-year" style="width:100%;margin:4px 0 8px 0;
                padding:4px;font-size:13px;border:1px solid #ccc;border-radius:4px;">
                <option value="">— choose year —</option>
                {% for yr in range(this.year_max, this.year_min - 1, -1) %}
                <option value="{{ yr }}">{{ yr }}</option>
                {% endfor %}
            </select>
            <div id="hurdat-storm-list" style="max-height:220px;overflow-y:auto;font-size:12px;"></div>
            <div style="margin-top:8px;display:flex;gap:6px;">
                <button onclick="hurdatSelectAll(true)"  style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">All</button>
                <button onclick="hurdatSelectAll(false)" style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">None</button>
                <button onclick="hurdatFitBounds()"      style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#e8f4fd;color:#0066cc;">Fit</button>
            </div>
            <div id="hurdat-summary" style="margin-top:8px;font-size:11px;color:#555;border-top:1px solid #eee;padding-top:6px;"></div>
        </div>
        <script>
        (function() {
            var STORMS={{ this.storms_json }},layers={},visible={},leafMap=null,curYear=null;
            function borderColor(c){return{"#5ebaff":"#2a7ab5","#00faf4":"#00a8a4","#ffffcc":"#b8b800","#ffe775":"#b89a00","#ffc140":"#b87800","#ff8f20":"#b85000","#ff6060":"#b82020"}[c]||"#333";}
            function drawStorm(storm){
                var sid=storm.storm_id,pts=storm.points;
                if(!pts.length)return;
                var coords=pts.map(function(p){return[p.lat,p.lon];});
                var line=L.polyline(coords,{color:"#333333",weight:4,opacity:0.8,dashArray:"4 4"}).addTo(leafMap);
                var circles=pts.map(function(p){
                    return L.circleMarker([p.lat,p.lon],{radius:5,fillColor:p.color,color:borderColor(p.color),weight:0.8,fillOpacity:0.9})
                    .bindTooltip("<b>"+storm.name+"</b><br>"+p.time+"<br><b>Wind:</b> "+p.wind_kt+" kt ("+Math.round(p.wind_kt*1.15082)+" mph)<br>"+(p.pres_mb?"<b>Pressure:</b> "+p.pres_mb+" mb<br>":"")+"<b>Status:</b> "+p.status,
                    {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:8px;"}).addTo(leafMap);
                });
                layers[sid]={line:line,circles:circles};visible[sid]=true;
            }
            function clearAllStorms(){
                Object.keys(layers).forEach(function(sid){layers[sid].line.remove();layers[sid].circles.forEach(function(c){c.remove();});});
                layers={};visible={};
            }
            function toggleStorm(sid,show){
                if(!layers[sid])return;
                if(show){layers[sid].line.addTo(leafMap);layers[sid].circles.forEach(function(c){c.addTo(leafMap);});}
                else{layers[sid].line.remove();layers[sid].circles.forEach(function(c){c.remove();});}
                visible[sid]=show;updateSummary();
            }
            window.hurdatSelectAll=function(state){document.querySelectorAll(".hurdat-chk").forEach(function(chk){chk.checked=state;toggleStorm(chk.dataset.sid,state);});};
            window.hurdatFitBounds=function(){
                var pts=[];
                Object.keys(layers).forEach(function(sid){if(visible[sid])layers[sid].line.getLatLngs().forEach(function(ll){pts.push(ll);});});
                if(pts.length)leafMap.fitBounds(L.latLngBounds(pts).pad(0.1));
            };
            function updateSummary(){
                if(!curYear)return;
                var total=STORMS[curYear]?STORMS[curYear].length:0;
                var shown=Object.values(visible).filter(Boolean).length;
                var hu=(STORMS[curYear]||[]).filter(function(s){return s.max_wind>=64;}).length;
                document.getElementById("hurdat-summary").innerHTML="<b>"+total+"</b> named &nbsp;|&nbsp; <b>"+hu+"</b> hurricanes &nbsp;|&nbsp; <b>"+shown+"</b> shown";
            }
            function loadYear(year){
                curYear=year;clearAllStorms();
                var list=document.getElementById("hurdat-storm-list");
                list.innerHTML="";
                var yearStorms=(STORMS[year]||[]).sort(function(a,b){return b.max_wind-a.max_wind;});
                yearStorms.forEach(function(storm){
                    drawStorm(storm);
                    var peakPt=storm.points.reduce(function(a,b){return a.wind_kt>b.wind_kt?a:b;},storm.points[0]);
                    var peakColor=peakPt?peakPt.color:"#ccc";
                    var row=document.createElement("div");
                    row.style.cssText="display:flex;align-items:center;padding:3px 0;border-bottom:1px solid #f0f0f0;";
                    row.innerHTML='<input type="checkbox" class="hurdat-chk" data-sid="'+storm.storm_id+'" checked style="margin-right:6px;">'
                        +'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+peakColor+';margin-right:5px;flex-shrink:0;"></span>'
                        +'<span style="flex:1;">'+storm.name+'</span>'
                        +'<span style="color:#888;font-size:11px;">'+storm.cat+' '+storm.max_wind+'kt</span>';
                    row.querySelector(".hurdat-chk").addEventListener("change",function(e){toggleStorm(e.target.dataset.sid,e.target.checked);updateSummary();});
                    list.appendChild(row);
                });
                updateSummary();
            }
            document.getElementById("hurdat-year").addEventListener("change",function(e){if(e.target.value)loadYear(e.target.value);else clearAllStorms();});
            var attempts=0,poller=setInterval(function(){
                attempts++;
                var maps=Object.values(window).filter(function(v){return v&&typeof v==="object"&&v._container&&typeof v.on==="function";});
                if(maps.length>0){clearInterval(poller);leafMap=maps[0];}
                if(attempts>40)clearInterval(poller);
            },250);
        })();
        </script>
        {% endmacro %}
        """)

        
class HistoricalExposure(MacroElement):
    def __init__(self, year_min, year_max):   # no more storms_json or grid_json params
        super().__init__()
        self._name    = "HistoricalExposure"
        self.year_min = year_min
        self.year_max = year_max
        self._template = Template("""
        {% macro html(this, kwargs) %}
        <!-- panel HTML unchanged -->
        <div id="hist-ctrl" style="
            position:fixed; bottom:10px; left:10px; z-index:9999;
            background:rgba(255,255,255,0.96); border:1px solid #ccc;
            border-radius:6px; padding:10px 14px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);
            min-width:230px; max-width:260px;">
            <b>🌀 Atlantic Storm Season</b>
            <hr style="margin:6px 0;">
            <label style="font-size:12px;color:#555;">Select Year</label><br>
            <select id="hist-year" style="width:100%;margin:4px 0 8px 0;
                padding:4px;font-size:13px;border:1px solid #ccc;border-radius:4px;">
                <option value="">— choose year —</option>
                {% for yr in range(this.year_max, this.year_min - 1, -1) %}
                <option value="{{ yr }}">{{ yr }}</option>
                {% endfor %}
            </select>
            <div id="hist-storm-list" style="max-height:200px;overflow-y:auto;font-size:12px;"></div>
            <div style="margin-top:8px;display:flex;gap:6px;">
                <button onclick="histSelectAll(true)"  style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">All</button>
                <button onclick="histSelectAll(false)" style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">None</button>
                <button onclick="histFitBounds()"      style="flex:1;padding:3px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#e8f4fd;color:#0066cc;">Fit</button>
            </div>
            <div style="margin-top:8px;border-top:1px solid #eee;padding-top:8px;">
                <label style="display:flex;align-items:center;cursor:pointer;gap:6px;">
                    <input type="checkbox" id="exposure-toggle" checked>
                    <span style="font-size:12px;">Highlight exposed hexes</span>
                </label>
                <div style="margin-top:4px;">
                    <label style="font-size:11px;color:#555;">Buffer radius (km):</label>
                    <input type="range" id="buffer-slider" min="10" max="250" value="50" step="10" style="width:100%;margin:2px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:10px;color:#888;">
                        <span>10</span>
                        <span id="buffer-val" style="font-weight:bold;color:#333;">50 km</span>
                        <span>250</span>
                    </div>
                </div>
            </div>
            <div id="hist-summary" style="margin-top:8px;font-size:11px;color:#555;
                border-top:1px solid #eee;padding-top:6px;"></div>
        </div>

        <div id="exposure-box" style="
            position:fixed; bottom:425px; right:10px; z-index:9999;
            background:rgba(255,255,255,0.96); border:1px solid #ccc;
            border-radius:6px; padding:10px 14px; font-family:Arial,sans-serif;
            font-size:13px; box-shadow:2px 2px 8px rgba(0,0,0,0.2);
            min-width:210px; display:none;">
            <b>📊 Exposed Portfolio</b>
            <hr style="margin:6px 0;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <tr><td style="color:#555;">Hexes in corridor</td><td id="exp-hexes" style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="color:#555;">Total Risks</td>      <td id="exp-risks" style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="color:#555;">Total TIV</td>        <td id="exp-tiv"   style="text-align:right;font-weight:bold;color:#c0392b;">—</td></tr>
                <tr><td style="color:#555;">Avg TIV / Hex</td>    <td id="exp-avg"   style="text-align:right;font-weight:bold;">—</td></tr>
                <tr><td style="color:#555;">Peak TIV hex</td>     <td id="exp-peak"  style="text-align:right;font-weight:bold;">—</td></tr>
            </table>
        </div>

        <!-- Loading indicator -->
        <div id="hist-loading" style="
            position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
            z-index:99999; background:rgba(255,255,255,0.95);
            border:1px solid #ccc; border-radius:8px;
            padding:20px 30px; font-family:Arial,sans-serif;
            font-size:14px; box-shadow:2px 2px 12px rgba(0,0,0,0.3);
            display:block; text-align:center;">
            ⏳ Loading storm data...
        </div>

        <script>
        (function() {
            var STORMS = null;
            var GRID   = null;
            var layers={},visible={},expLayer=null,leafMap=null,
                curYear=null,bufferKm=50,showExp=true;

            // ── Fetch both data files in parallel ────────────────────────────────
            // Build absolute URLs relative to wherever this file is being served from
            var base = window.location.href.replace(/\/[^\/]*$/, "/");

            console.log("Base URL:", base);
            console.log("Fetching:", base + "storms_data.json");
            console.log("Fetching:", base + "grid_data.json");
            
            Promise.all([
            fetch(base + "storms_data.json").then(function(r) { return r.json(); }),
            fetch(base + "grid_data.json").then(function(r)   { return r.json(); })
            
            
            ]).then(function(results) {
                STORMS = results[0];
                GRID   = results[1];
                document.getElementById("hist-loading").style.display = "none";
                console.log("Storm data loaded — years available:",
                            Object.keys(STORMS).length);
            }).catch(function(err) {
                document.getElementById("hist-loading").innerText =
                    "❌ Failed to load storm data. " +
                    "Ensure storms_data.json and grid_data.json " +
                    "are in the same folder and you are running " +
                    "via a local server (not file://).";
                console.error("Data load error:", err);
            });

            // ── All your existing helper functions unchanged ──────────────────────
            function fmtInt(n){return Math.round(n).toLocaleString();}
            function fmtTIV(n){var k=n/1000;if(k>=1e6)return"$"+(k/1e6).toFixed(2)+"B";else if(k>=1000)return"$"+(k/1000).toFixed(1)+"M";else return"$"+k.toFixed(1)+"K";}
            function borderColor(c){return{"#5ebaff":"#2a7ab5","#00faf4":"#00a8a4","#ffffcc":"#b8b800","#ffe775":"#b89a00","#ffc140":"#b87800","#ff8f20":"#b85000","#ff6060":"#b82020"}[c]||"#333";}
            function haversine(lat1,lon1,lat2,lon2){var R=6371,dL=(lat2-lat1)*Math.PI/180,dl=(lon2-lon1)*Math.PI/180;var a=Math.sin(dL/2)*Math.sin(dL/2)+Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dl/2)*Math.sin(dl/2);return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));}
            function ptSegDist(plat,plon,alat,alon,blat,blon){var dx=blon-alon,dy=blat-alat,len2=dx*dx+dy*dy;if(len2===0)return haversine(plat,plon,alat,alon);var t=Math.max(0,Math.min(1,((plon-alon)*dx+(plat-alat)*dy)/len2));return haversine(plat,plon,alat+t*dy,alon+t*dx);}
            function hexInCorridor(hex,stormPoints){for(var i=0;i<stormPoints.length-1;i++){if(ptSegDist(hex.cy,hex.cx,stormPoints[i].lat,stormPoints[i].lon,stormPoints[i+1].lat,stormPoints[i+1].lon)<=bufferKm)return true;}return false;}

            function drawStorm(storm){
                var sid=storm.storm_id,pts=storm.points;
                if(!pts.length)return;
                var line=L.polyline(pts.map(function(p){return[p.lat,p.lon];}),{color:"#333333",weight:2.5,opacity:0.6,dashArray:"4 4"}).addTo(leafMap);
                var circles=pts.map(function(p){
                    return L.circleMarker([p.lat,p.lon],{radius:5,fillColor:p.color,color:borderColor(p.color),weight:0.8,fillOpacity:0.9})
                    .bindTooltip("<b>"+storm.name+"</b><br>"+p.time+"<br><b>Wind:</b> "+p.wind_kt+" kt ("+Math.round(p.wind_kt*1.15082)+" mph)<br>"+(p.pres_mb?"<b>Pressure:</b> "+p.pres_mb+" mb<br>":"")+"<b>Status:</b> "+p.status,
                    {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:8px;"}).addTo(leafMap);
                });
                layers[sid]={line:line,circles:circles,points:pts};visible[sid]=true;
            }
            function clearAllStorms(){Object.keys(layers).forEach(function(sid){layers[sid].line.remove();layers[sid].circles.forEach(function(c){c.remove();});});layers={};visible={};clearExposure();}
            function toggleStorm(sid,show){
                if(!layers[sid])return;
                if(show){layers[sid].line.addTo(leafMap);layers[sid].circles.forEach(function(c){c.addTo(leafMap);});}
                else{layers[sid].line.remove();layers[sid].circles.forEach(function(c){c.remove();});}
                visible[sid]=show;runExposure();updateSummary();
            }
            window.histSelectAll=function(state){document.querySelectorAll(".hist-chk").forEach(function(chk){chk.checked=state;toggleStorm(chk.dataset.sid,state);});};
            window.histFitBounds=function(){var pts=[];Object.keys(layers).forEach(function(sid){if(visible[sid])layers[sid].line.getLatLngs().forEach(function(ll){pts.push(ll);});});if(pts.length)leafMap.fitBounds(L.latLngBounds(pts).pad(0.1));};

            function clearExposure(){if(expLayer){expLayer.remove();expLayer=null;}document.getElementById("exposure-box").style.display="none";}
            function runExposure(){
                clearExposure();if(!showExp)return;
                var activePts=[];
                Object.keys(layers).forEach(function(sid){if(visible[sid])activePts=activePts.concat(layers[sid].points);});
                if(!activePts.length)return;
                var expHexes=[],riskSum=0,tivSum=0,peakTIV=0;
                GRID.forEach(function(hex){
                    if(hexInCorridor(hex,activePts)){
                        expHexes.push(hex);riskSum+=hex.risk_count||0;tivSum+=hex.tiv_sum||0;
                        if(hex.tiv_sum>peakTIV)peakTIV=hex.tiv_sum;
                    }
                });
                expLayer=L.layerGroup();
                expHexes.forEach(function(hex){
                    L.circleMarker([hex.cy,hex.cx],{radius:8,fillColor:"#ff0066",color:"#cc0044",weight:1,fillOpacity:0.55})
                    .bindTooltip("<b>Exposed Hex</b><br>TIV: "+fmtTIV(hex.tiv_sum)+"<br>Risks: "+fmtInt(hex.risk_count),
                    {sticky:true,style:"background:white;font-family:arial;font-size:12px;padding:6px;"}).addTo(expLayer);
                });
                expLayer.addTo(leafMap);
                var n=expHexes.length;
                document.getElementById("exp-hexes").innerText=fmtInt(n);
                document.getElementById("exp-risks").innerText=fmtInt(riskSum);
                document.getElementById("exp-tiv").innerText=fmtTIV(tivSum);
                document.getElementById("exp-avg").innerText=n>0?fmtTIV(tivSum/n):"—";
                document.getElementById("exp-peak").innerText=fmtTIV(peakTIV);
                document.getElementById("exposure-box").style.display="block";
            }

            function loadYear(year){
                if(!STORMS){console.warn("Storm data not yet loaded");return;}
                curYear=year;clearAllStorms();
                var list=document.getElementById("hist-storm-list");list.innerHTML="";
                var yearStorms=(STORMS[year]||[]).sort(function(a,b){return b.max_wind-a.max_wind;});
                yearStorms.forEach(function(storm){
                    drawStorm(storm);
                    var peakPt=storm.points.reduce(function(a,b){return a.wind_kt>b.wind_kt?a:b;},storm.points[0]);
                    var row=document.createElement("div");
                    row.style.cssText="display:flex;align-items:center;padding:3px 0;border-bottom:1px solid #f0f0f0;";
                    row.innerHTML='<input type="checkbox" class="hist-chk" data-sid="'+storm.storm_id+'" checked style="margin-right:6px;">'
                        +'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+(peakPt?peakPt.color:"#ccc")+';margin-right:5px;flex-shrink:0;"></span>'
                        +'<span style="flex:1;">'+storm.name+'</span>'
                        +'<span style="color:#888;font-size:11px;">'+storm.cat+' '+storm.max_wind+'kt</span>';
                    row.querySelector(".hist-chk").addEventListener("change",function(e){toggleStorm(e.target.dataset.sid,e.target.checked);});
                    list.appendChild(row);
                });
                runExposure();updateSummary();
            }
            function updateSummary(){
                if(!curYear)return;
                var total=STORMS[curYear]?STORMS[curYear].length:0;
                var hu=(STORMS[curYear]||[]).filter(function(s){return s.max_wind>=64;}).length;
                var shown=Object.values(visible).filter(Boolean).length;
                document.getElementById("hist-summary").innerHTML="<b>"+total+"</b> named &nbsp;|&nbsp; <b>"+hu+"</b> hurricanes &nbsp;|&nbsp; <b>"+shown+"</b> shown";
            }

            document.getElementById("buffer-slider").addEventListener("input",function(e){
                bufferKm=parseInt(e.target.value);
                document.getElementById("buffer-val").innerText=bufferKm+" km";
                if(curYear)runExposure();
            });
            document.getElementById("exposure-toggle").addEventListener("change",function(e){
                showExp=e.target.checked;if(showExp)runExposure();else clearExposure();
            });
            document.getElementById("hist-year").addEventListener("change",function(e){
                if(e.target.value)loadYear(e.target.value);else clearAllStorms();
            });

            var attempts=0,poller=setInterval(function(){
                attempts++;
                var maps=Object.values(window).filter(function(v){return v&&typeof v==="object"&&v._container&&typeof v.on==="function";});
                if(maps.length>0){clearInterval(poller);leafMap=maps[0];}
                if(attempts>40)clearInterval(poller);
            },250);
        })();
        </script>
        {% endmacro %}
        """)
 

# ── HURDAT2 parse ─────────────────────────────────────────────────────────────────
# url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
url = 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt'
raw = requests.get(url).text
storms, current = [], {}
for line in raw.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3 and parts[0].startswith("AL"):
        current = {"storm_id": parts[0], "storm_name": parts[1],
                   "n_records": int(parts[2])}
    elif len(parts) >= 8:
        storms.append({**current, "date": parts[0], "time": parts[1],
                       "status": parts[3],
                       "lat":  float(parts[4].replace("N","").replace("S","")),
                       "lon": -float(parts[5].replace("W","").replace("E","")),
                       "wind_kt": int(parts[6]),
                       "pres_mb": int(parts[7]) if parts[7].strip() != "-999" else None
                      })

hurdat = pd.DataFrame(storms)
hurdat["datetime"] = pd.to_datetime(hurdat["date"] + " " + hurdat["time"],
                                    format="%Y%m%d %H%M")
hurdat["year"] = hurdat["datetime"].dt.year

def ss_color(wind_kt):
    if wind_kt < 34:  return "#5ebaff"
    if wind_kt < 64:  return "#00faf4"
    if wind_kt < 83:  return "#ffffcc"
    if wind_kt < 96:  return "#ffe775"
    if wind_kt < 113: return "#ffc140"
    if wind_kt < 137: return "#ff8f20"
    return "#ff6060"

year_min, year_max = 1880, 2025
storms_by_year = {}
for year, ydf in hurdat[hurdat["year"].between(year_min, year_max)].groupby("year"):
    year_storms = []
    for storm_id, sdf in ydf.groupby("storm_id"):
        sdf = sdf.sort_values("datetime")
        max_wind = int(sdf["wind_kt"].max())
        if max_wind < 34:    cat = "TD"
        elif max_wind < 64:  cat = "TS"
        elif max_wind < 83:  cat = "Cat 1"
        elif max_wind < 96:  cat = "Cat 2"
        elif max_wind < 113: cat = "Cat 3"
        elif max_wind < 137: cat = "Cat 4"
        else:                cat = "Cat 5"

        points = []
        for _, row in sdf.iterrows():
            points.append({
                "lat":     row["lat"],
                "lon":     row["lon"],
                "wind_kt": int(row["wind_kt"]),
                "pres_mb": None if pd.isna(row["pres_mb"]) else int(row["pres_mb"]),
                "status":  row["status"],
                "time":    row["datetime"].strftime("%Y-%m-%d %H:%MZ"),
                "color":   ss_color(row["wind_kt"]),
            })

        year_storms.append({            # inside storm loop
            "storm_id": storm_id,
            "name":     sdf["storm_name"].iloc[0],
            "cat":      cat,
            "max_wind": max_wind,
            "points":   points,
        })

    storms_by_year[str(year)] = year_storms   # inside year loop, outside storm loop

storms_json = json.dumps(storms_by_year)      # outside all loops

# ── Wind color by speed ───────────────────────────────────────────────────────────
# ── SCS color helpers ─────────────────────────────────────────────────────────────
def ef_color(mag):
    return {
        0: "#ffffb2",
        1: "#fecc5c",
        2: "#fd8d3c",
        3: "#f03b20",
        4: "#bd0026",
        5: "#6a0005",
    }.get(int(mag) if mag >= 0 else -1, "#aaaaaa")

def hail_color(size):
    if size >= 4.0:  return "#6a0005"
    if size >= 2.0:  return "#bd0026"
    if size >= 1.75: return "#f03b20"
    if size >= 1.0:  return "#fd8d3c"
    return "#fecc5c"
    
def wind_color(mag):
    """mag in mph for SPC wind reports"""
    if mag >= 100: return "#6a0005"
    if mag >= 75:  return "#bd0026"
    if mag >= 65:  return "#f03b20"
    if mag >= 55:  return "#fd8d3c"
    return "#fecc5c"

def add_tornado_layer(m, gdf):
    layer = fl.FeatureGroup(name="Tornado Tracks (2015-2025)", show=False)
    for _, row in gdf.iterrows():
        color = ef_color(row["mag"])
        fl.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, c=color: {
                "color": c, "weight": 1.5, "opacity": 0.7
            },
            tooltip=(
                f"<b>EF{int(row['mag']) if row['mag'] >= 0 else '?'}</b> — {row['st']}<br>"
                f"Date: {int(row['yr'])}-{int(row['mo']):02d}-{int(row['dy']):02d}<br>"
                f"Length: {row['len']:.1f} mi &nbsp; Width: {row['wid']:.0f} yds"
            )
        ).add_to(layer)
    layer.add_to(m)
    print(f"Tornado layer added — {len(gdf)} tracks")

def add_hail_layer(m, gdf):
    layer = fl.FeatureGroup(name="Hail Reports (2015-2025)", show=False)
    for _, row in gdf.iterrows():
        color = hail_color(row["mag"])
        fl.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, c=color: {
                "color": c, "weight": 1.5, "opacity": 0.6
            },
            tooltip=(
                f"<b>Hail: {row['mag']}\"</b><br>"
                f"Date: {int(row['yr'])}-{int(row['mo']):02d}-{int(row['dy']):02d}<br>"
                f"State: {row['st']}"
            )
        ).add_to(layer)
    layer.add_to(m)
    print(f"Hail layer added — {len(gdf)} reports")

def add_wind_layer(m, gdf):
    layer = fl.FeatureGroup(name="Wind Reports (2015-2025)", show=False)
    for _, row in gdf.iterrows():
        color = wind_color(row["mag"])
        fl.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, c=color: {
                "color": c, "weight": 1, "opacity": 0.5
            },
            tooltip=(
                f"<b>Wind: {row['mag']} mph</b><br>"
                f"Date: {int(row['yr'])}-{int(row['mo']):02d}-{int(row['dy']):02d}<br>"
                f"State: {row['st']}"
            )
        ).add_to(layer)
    layer.add_to(m)
    print(f"Wind layer added — {len(gdf)} reports")




# ── State boundaries (downloaded once) ───────────────────────────────────────────
states_dir = os.path.join(folder, "us_states")
os.makedirs(states_dir, exist_ok=True)


states_url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
r = requests.get(states_url, timeout=30)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(states_dir)
states_gdf     = gpd.read_file(states_dir).to_crs("EPSG:4326")
states_geojson = json.loads(states_gdf.to_json())

def load_us_cities(gazetteer_csv_path):
    """Census Gazetteer places file -> GeoDataFrame of city points."""
    df = pd.read_csv(gazetteer_csv_path, sep="|", dtype=str)
    df.columns = df.columns.str.strip()
    print(df.columns)
    df["INTPTLAT"] = df["INTPTLAT"].astype(float)
    df["INTPTLONG"] = df["INTPTLONG"].astype(float)
    geometry = [Point(xy) for xy in zip(df["INTPTLONG"], df["INTPTLAT"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# ── cities (downloaded once) ───────────────────────────────────────────
cities_dir = os.path.join(folder, "us_places")
os.makedirs(cities_dir, exist_ok=True)


cities_url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_Gaz_place_national.zip"
r = requests.get(cities_url, timeout=30)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(cities_dir)
cities_gdf = load_us_cities(os.path.join(cities_dir, "2025_Gaz_place_national.txt"))

# ── ZCTA (ZIP code) boundaries (downloaded once) ─────────────────────────────────
zcta_dir = os.path.join(folder, "us_zcta")
os.makedirs(zcta_dir, exist_ok=True)

zcta_url = "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
r = requests.get(zcta_url, timeout=60)
r.raise_for_status()
if not r.content.startswith(b"PK"):
    raise ValueError(f"Expected a zip file from {zcta_url}, got: {r.content[:200]}")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(zcta_dir)
zcta_gdf = gpd.read_file(zcta_dir).to_crs("EPSG:4326")

cities_gdf = gpd.sjoin(
    cities_gdf, zcta_gdf[["ZCTA5CE20", "geometry"]],
    how="left", predicate="within"
).rename(columns={"ZCTA5CE20": "ZIP"})


# ── Slim grid for JS embedding ────────────────────────────────────────────────────
grid_slim = grid_wgs[["geometry", "risk_count", "tiv_sum"]].copy()
grid_exp  = grid_wgs[["h3_id", "risk_count", "tiv_sum", "tiv_per_sqmi"]].copy()
grid_exp["cx"] = grid_wgs["centroid_x"]   # lon
grid_exp["cy"] = grid_wgs["centroid_y"]   # lat
grid_json = grid_exp[["h3_id","risk_count","tiv_sum","tiv_per_sqmi","cx","cy"]].to_json(orient="records")


# ── Save data files to disk immediately after building ────────────────────────────
with open(os.path.join(folder, "storms_data.json"), "w", encoding="utf-8") as f:
    f.write(storms_json)

with open(os.path.join(folder, "grid_data.json"), "w", encoding="utf-8") as f:
    f.write(grid_json)

# ── Save external portfolio GeoJSON ──────────────────────────────
ports_data = {}
for port in port_list:
    port_data = grid_wgs[grid_wgs["Port"] == port].copy().reset_index(drop=True)
    ports_data[port] = json.loads(port_data.to_json())

with open(os.path.join(folder, "ports_data.geojson"), "w", encoding="utf-8") as f:
    json.dump(ports_data, f)

print(f"storms_data.json: {os.path.getsize(os.path.join(folder, 'storms_data.json'))/1024/1024:.2f} MB")
print(f"grid_data.json:   {os.path.getsize(os.path.join(folder, 'grid_data.json'))/1024/1024:.2f} MB")
print(f"ports_data.geojson: {os.path.getsize(os.path.join(folder, 'ports_data.geojson'))/1024/1024:.2f} MB")

# ── Map builder helpers ───────────────────────────────────────────────────────────
def _add_basemaps(m):
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — National Geographic", name="NG World"
    ).add_to(m)
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — World Imagery", name="Sat"
    ).add_to(m)
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Dark Gray Canvas", name="High Contrast"
    ).add_to(m)



####################################################################################################
# ── Map 1 — Live Exposure ─────────────────────────────────────────────────────────
def build_exposure_map():

    center = grid_wgs.geometry.union_all().centroid
    m = fl.Map(location=[center.y, center.x], zoom_start=6, tiles="cartodbpositron")

    PortfolioLayers().add_to(m)
    #fl.LayerControl(collapsed=False).add_to(m)
    
    _add_basemaps(m)

    StateBoundaries().add_to(m)

    fl.raster_layers.WmsTileLayer(
        url="https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r.cgi",
        layers="nexrad-n0r-900913", name="NEXRAD Radar",
        fmt="image/png", transparent=True, overlay=True, show=False,
    ).add_to(m)

    # Live NHC cones
    def find_impact(cone_gdf, cities_gdf, states_gdf, storm_id, storm_name, folder):
        """Spatial join cone polygon against cities + states, write CSV, return summary."""
        cone_union = cone_gdf.geometry.union_all()
    
        minx, miny, maxx, maxy = cone_gdf.total_bounds
        pad = 0.5  # degrees buffer, ~35 miles, so edge-of-cone cities aren't clipped by the bbox cut
        cities_nearby = cities_gdf.cx[minx - pad:maxx + pad, miny - pad:maxy + pad]
    
        cities_hit = cities_nearby[cities_nearby.geometry.within(cone_union)].copy()
        cities_hit["storm_name"] = storm_name
    
        states_hit = states_gdf[states_gdf.geometry.intersects(cone_union)].copy()
        state_names = states_hit["NAME"].tolist() if "NAME" in states_hit.columns else []
    
        csv_path = f"{folder}/cone_impact_{storm_name}.csv"
        cities_hit[["NAME", "ZIP", "USPS", "storm_name"]].to_csv(
            csv_path, index=False
        )
    
        return {
            "storm_id": storm_id,
            "storm_name": storm_name,
            "states": state_names,
            "city_count": len(cities_hit),
            "csv_file": f"cone_impact_{storm_name}.csv",
        }

    cone_impacts = []   # collect across all storms in this rebuild

    try:
        active = requests.get(
            "https://www.nhc.noaa.gov/CurrentStorms.json", timeout=5).json()

        def _load_shp_from_zip(zip_bytes, name_contains, out_folder):
            z = zipfile.ZipFile(io.BytesIO(zip_bytes))
            shp_names = [n for n in z.namelist()
                         if n.lower().endswith(".shp") and name_contains in n.lower()]
            if not shp_names:
                return None
            tmp_dir = os.path.join(out_folder, "nhc_storm_tmp")
            z.extractall(tmp_dir)
            return gpd.read_file(os.path.join(tmp_dir, shp_names[0]))

        for storm in active.get("activeStorms", []):
            sid  = storm["id"]
            name = storm.get("name", sid)

            cone_zip_url = storm.get("trackCone", {}).get("zipFile")
            ww_zip_url   = (storm.get("windWatchesWarnings") or {}).get("zipFile")

            cone_gdf = pts_gdf = ww_gdf = None

            if cone_zip_url:
                zip_bytes = requests.get(cone_zip_url, timeout=10).content
                cone_gdf = _load_shp_from_zip(zip_bytes, "pgn", folder)
                pts_gdf  = _load_shp_from_zip(zip_bytes, "pts", folder)

            if ww_zip_url:
                ww_bytes = requests.get(ww_zip_url, timeout=10).content
                ww_gdf   = _load_shp_from_zip(ww_bytes, "ww", folder)

            if cone_gdf is not None:
                impact = find_impact(cone_gdf, cities_gdf, states_gdf, sid, name, folder)
                cone_impacts.append(impact)
                
                fl.GeoJson(
                    cone_gdf,
                    name=f"NHC Cone — {name}",
                    style_function=lambda x: {"fillColor":"#ff6600","color":"#cc3300",
                                              "weight":1.5,"fillOpacity":0.25},
                    tooltip=fl.GeoJsonTooltip(
                        fields=[c for c in ["STORMNAME","ADVDATE"] if c in cone_gdf.columns],
                        aliases=["Storm","Advisory issued"],
                        style="background-color:white;font-family:arial;font-size:12px;padding:8px;")
                ).add_to(m)
            
            if pts_gdf is not None:
                fl.GeoJson(
                    pts_gdf, name=f"NHC Track — {name}",
                    tooltip=fl.GeoJsonTooltip(
                        fields=[c for c in ["STORMNAME","TCDVLP","MAXWIND","VALIDTIME"] if c in pts_gdf.columns],
                        aliases=["Storm","Category","Max wind (kt)","Valid time"],
                        style="background-color:white;font-family:arial;font-size:12px;padding:8px;"),
                    marker=fl.CircleMarker(
                        radius=5, color="#cc3300", fill=True, fill_color="#ff6600")
                ).add_to(m)
            
            if ww_gdf is not None:
                fl.GeoJson(
                    ww_gdf, name=f"NHC Watches/Warnings — {name}",
                    style_function=lambda x: {
                        "fillColor": {"Hurricane Warning":"#ff0000","Hurricane Watch":"#ffb3d9",
                                       "Tropical Storm Warning":"#0000ff","Tropical Storm Watch":"#ffff00"
                                      }.get(x["properties"].get("TCWW",""), "#999999"),
                        "color":"#333333","weight":1,"fillOpacity":0.4},
                    tooltip=fl.GeoJsonTooltip(
                        fields=[c for c in ["TCWW"] if c in ww_gdf.columns],
                        aliases=["Alert type"],
                        style="background-color:white;font-family:arial;font-size:12px;padding:8px;"),
                    show=False
                ).add_to(m)              

    except requests.exceptions.RequestException:
        pass

    if cone_impacts:
        ConeImpactPanel(cone_impacts).add_to(m)
    

    Draw(draw_options={"polyline":False,"circle":False,"marker":False,
                       "circlemarker":False,"rectangle":True,"polygon":True},
         edit_options={"edit":True,"remove":True}).add_to(m)

    SelectionSummary(grid_slim.to_json()).add_to(m)
    RadarRefresh(interval_ms=300000).add_to(m)

    LayerControlStyle().add_to(m)
    FloodRegionSelector().add_to(m)
    fl.LayerControl(collapsed=True).add_to(m)
    FloatLegend(legend_html).add_to(m)
    FloatLegend(weather_legend_html).add_to(m)
    FloatLegend(flood_legend_html).add_to(m)
    RulerControl(position="topright", unit="mi").add_to(m)

    # add nhc legend
    nhc_legend = NHCLegend()
    nhc_legend.add_to(m)
    
    return m

# ── Map 2 — Historical Analysis ───────────────────────────────────────────────────
def build_historical_map():
    # print(f"port_list inside build_historical_map: {port_list}")
    # print(f"Port value counts:\n{grid_wgs['Port'].value_counts()}")
    
    center = grid_wgs.geometry.union_all().centroid
    m = fl.Map(
        location=[center.y, center.x],
        zoom_start=5,
        tiles = 'cartodbpositron',
        height="100%",
        width="100%"
    )

    
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — National Geographic",
        name="NG World"
    ).add_to(m)

    # Additional basemaps
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — World Imagery", name="Sat"
    ).add_to(m)
    fl.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Dark Gray Canvas", name="High Contrast"
    ).add_to(m)

    StateBoundaries().add_to(m)

    fl.LayerControl(collapsed=True).add_to(m)


    #######################################################################################
    
    HistoricalExposure(year_min, year_max).add_to(m)
    SCSYearSelector(2015, 2025).add_to(m)

    # ── SCS layers ────────────────────────────────────────────────────────────────

    PortfolioLayers().add_to(m)  
    FloatLegend(legend_html).add_to(m)
    FloatLegend(weather_legend_html).add_to(m)
    FloatLegend(scs_legend_html).add_to(m)
    
    LayerControlStyle().add_to(m)
    return m

# ── Export ────────────────────────────────────────────────────────────────────────
m_live = build_exposure_map()
m_hist = build_historical_map()

m_live.save(os.path.join(folder, "map_liveV2.html"))
m_hist.save(os.path.join(folder, "map_histV2.html"))

# ── SPC Storm Reports ─────────────────────────────────────────────────────────────
from spc_live_reports_1 import build_spc_json, inject_into_map_html
build_spc_json(output_dir=folder)
inject_into_map_html(os.path.join(folder, "map_liveV2.html"))

# ── USGS Earthquakes ──────────────────────────────────────────────────────────
from usgs_earthquakes import build_usgs_json, inject_into_map_html as inject_usgs
build_usgs_json(output_dir=folder)
inject_usgs(os.path.join(folder, "map_liveV2.html"))

# ── Coastal Wind Tiers ──────────────────────────────────────────────────────────
# from coastal_wind_tier import build_coastal_tier_json, inject_into_map_html as inject_coastal

# build_coastal_tier_json(loc_df, grid_wgs, h3_col="h3_id", tiv_col="Cov_A",out_path="coastal_tier.json")
# inject_coastal("map_liveV2.html")

# ── wildfire  ──────────────────────────────────────────────────────────

from wildfire_live import build_wildfire_json, inject_into_map_html
build_wildfire_json(output_dir=folder)
inject_into_map_html(folder + "/map_liveV2.html")



# Save data files separately
with open(os.path.join(folder, "storms_data.json"), "w", encoding="utf-8") as f:
    f.write(storms_json)

with open(os.path.join(folder, "grid_data.json"), "w", encoding="utf-8") as f:
    f.write(grid_json)

# ── Save states GeoJSON externally ───────────────────────────────
with open(os.path.join(folder, "states.geojson"), "w", encoding="utf-8") as f:
    json.dump(states_geojson, f)

print(f"states.geojson: {os.path.getsize(os.path.join(folder, 'states.geojson'))/1024/1024:.2f} MB")


# Wrapper stays exactly the same as before
# with open(os.path.join(folder, "hadron_portfolio_maps.html"), "w", encoding="utf-8") as f:
#    f.write(combined_html)

with open(os.path.join(folder, "hadron_portfolio_maps.html"), "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hadron Cat Risk — Exposure Maps</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: Arial, sans-serif; background: #1a1a2e; }
    .tab-bar { display: flex; background: #1a1a2e; padding: 8px 12px 0 12px; gap: 4px; }
    .tab-btn { padding: 8px 20px; border: none; border-radius: 6px 6px 0 0;
               cursor: pointer; font-size: 13px; font-weight: bold;
               background: #2d2d44; color: #aaa; }
    .tab-btn.active { background: white; color: #1a1a2e; }
    .tab-btn:hover:not(.active) { background: #3d3d5c; color: #ddd; }
    .tab-content { display: none; width: 100%; height: calc(100vh - 44px); }
    .tab-content.active { display: block; }
    .tab-content iframe { width: 100%; height: 100%; border: none; }
  </style>
</head>
<body>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('live', this)">🗺 Live Exposure</button>
    <button class="tab-btn"        onclick="switchTab('hist', this)">🌀 Storm History</button>
  </div>
  <div id="tab-live" class="tab-content active"><iframe id="iframe-live"></iframe></div>
  <div id="tab-hist" class="tab-content">        <iframe id="iframe-hist"></iframe></div>
  <script>
    function loadMapFile(path, iframeId) {
      fetch(path)
        .then(function(r) { return r.text(); })
        .then(function(html) {
          var blob = new Blob([html], {type: "text/html"});
          document.getElementById(iframeId).src = URL.createObjectURL(blob);
        });
    }
    loadMapFile("map_live.html", "iframe-live");
    loadMapFile("map_hist.html", "iframe-hist");
    function switchTab(name, btn) {
      document.querySelectorAll(".tab-content").forEach(function(el) { el.classList.remove("active"); });
      document.querySelectorAll(".tab-btn").forEach(function(el) { el.classList.remove("active"); });
      document.getElementById("tab-" + name).classList.add("active");
      btn.classList.add("active");
    }
  </script>
</body>
</html>""")

print(f"hadron_portfolio_maps.html: {os.path.getsize(os.path.join(folder, 'hadron_portfolio_maps.html'))} bytes")





print(f"map_live.html:              {os.path.getsize(os.path.join(folder, 'map_liveV2.html'))/1024/1024:.2f} MB")
print(f"map_hist.html:              {os.path.getsize(os.path.join(folder, 'map_histV2.html'))/1024/1024:.2f} MB")
print(f"storms_data.json:           {os.path.getsize(os.path.join(folder, 'storms_data.json'))/1024/1024:.2f} MB")
print(f"grid_data.json:             {os.path.getsize(os.path.join(folder, 'grid_data.json'))/1024/1024:.2f} MB")
print(f"hadron_portfolio_maps.html:  {os.path.getsize(os.path.join(folder, 'hadron_portfolio_maps.html'))/1024/1024:.2f} MB")

# ── Tab widget ────────────────────────────────────────────────────────────────────
live_out = widgets.Output()
hist_out = widgets.Output()

with live_out: display(build_exposure_map())
with hist_out: display(build_historical_map())

tab = widgets.Tab(children=[live_out, hist_out])
tab.set_title(0, "🗺  Live Exposure")
tab.set_title(1, "🌀 Storm History")

display(tab)

folder = r"C:\Users\BryanButler\OneDrive - Hadron Specialty Insurance EU\Documents\PythonFiles\USPortfolioMaps"
port   = 8000


with open(os.path.join(folder, "hadron_portfolio_maps.html"), "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hadron Cat Risk — Exposure Maps</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: Arial, sans-serif; background: #1a1a2e; }
    .tab-bar { display: flex; background: #1a1a2e; padding: 8px 12px 0 12px; gap: 4px; }
    .tab-btn { padding: 8px 20px; border: none; border-radius: 6px 6px 0 0;
               cursor: pointer; font-size: 13px; font-weight: bold;
               background: #2d2d44; color: #aaa; }
    .tab-btn.active { background: white; color: #1a1a2e; }
    .tab-btn:hover:not(.active) { background: #3d3d5c; color: #ddd; }
    .tab-content { display: none; width: 100%; height: calc(100vh - 44px); }
    .tab-content.active { display: block; }
    .tab-content iframe { width: 100%; height: 100%; border: none; }
  </style>
</head>
<body>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('live', this)">🗺 Live Exposure</button>
    <button class="tab-btn"        onclick="switchTab('hist', this)">🌀 Storm History</button>
  </div>
  <div id="tab-live" class="tab-content active">
    <iframe src="http://localhost:8000/map_liveV2.html"></iframe>
  </div>
  <div id="tab-hist" class="tab-content">
    <iframe src="http://localhost:8000/map_histV2.html"></iframe>
  </div>
  <script>
    function switchTab(name, btn) {
      document.querySelectorAll(".tab-content").forEach(function(el) { el.classList.remove("active"); });
      document.querySelectorAll(".tab-btn").forEach(function(el) { el.classList.remove("active"); });
      document.getElementById("tab-" + name).classList.add("active");
      btn.classList.add("active");
    }
  </script>
</body>
</html>""")

print(f"hadron_portfolio_maps.html: {os.path.getsize(os.path.join(folder, 'hadron_portfolio_maps.html'))} bytes")


import subprocess
import webbrowser
import time

# Kill existing server on port 8000
subprocess.run(
    ["powershell", "-Command",
     "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | "
     "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
    capture_output=True
)
time.sleep(1)

# Start fresh
subprocess.Popen(
    ["python", "no_cache_server.py", "8000"],
    cwd=folder,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(1)

# Open combined map
webbrowser.open("http://localhost:8000/hadron_portfolio_maps.html")
print("Serving at http://localhost:8000/hadron_portfolio_maps.html")

# refresh_live_data.py
# Run this anytime to update SPC + EQ data without rebuilding the map

import os
from spc_live_reports import build_spc_json
from usgs_earthquakes import build_usgs_json

folder = r"C:\Users\BryanButler\OneDrive - Hadron Specialty Insurance EU\Documents\PythonFiles\USPortfolioMaps"

build_spc_json(output_dir=folder)
build_usgs_json(output_dir=folder)

print("Live data refreshed — reload the browser tab to see updates.")





