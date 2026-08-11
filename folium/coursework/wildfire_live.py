"""
wildfire_live.py
-----------------
NIFC / Esri Living Atlas Current Wildfires — Pre-bake Module
Fetches live wildfire incident points + perimeters at map build time
and writes wildfire_data.json alongside your HTML, exactly like your
other external data files (ports_data.geojson, spc_reports.json, etc.).

Data source
-----------
Esri Living Atlas "USA_Wildfires_v1" FeatureServer, sourced from NIFC's
WFIGS/IRWIN interagency feed. Two layers:
  Layer 0  Current_Incidents   (points)    — updated ~continuously
  Layer 1  Current_Perimeters  (polygons)  — daily fire perimeters

  https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/
      USA_Wildfires_v1/FeatureServer/0   (incidents)
      USA_Wildfires_v1/FeatureServer/1   (perimeters)

No API key required — public Esri-hosted feature service.

Usage in your map build script
-------------------------------
from wildfire_live import build_wildfire_json, inject_into_map_html

# 1. Fetch + write JSON next to your map HTML
build_wildfire_json(output_dir="C:/path/to/your/map/output")

# 2. After map.save(), inject the toggle button + JS
inject_into_map_html("C:/path/to/your/map/output/live_map.html")

Filters applied
---------------
  Incidents  : IncidentTypeCategory in {WF, CX}  (wildfires + complexes)
               Prescribed fires (RX) included but flagged separately.
  Perimeters : joined back to incidents by IRWIN ID where possible so
               popups can show containment % (perimeters alone don't
               carry that attribute).
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WILDFIRE_SERVICE = "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/USA_Wildfires_v1/FeatureServer"

INCIDENT_FIELDS = [
    "IncidentName", "IncidentTypeCategory", "UniqueFireIdentifier",
    "DailyAcres", "CalculatedAcres", "PercentContained",
    "FireDiscoveryDateTime", "DiscoveryAcres", "FireDiscoveryAge",
    "POOCounty", "POOState", "FireCause", "FireCauseGeneral", "GACC",
    "TotalIncidentPersonnel", "IncidentManagementOrganization",
    "FireMgmtComplexity", "ResidencesDestroyed", "OtherStructuresDestroyed",
    "Injuries", "Fatalities", "ContainmentDateTime", "ModifiedOnDateTime",
    "IrwinID",
]

PERIMETER_FIELDS = [
    "IncidentName", "FeatureCategory", "GISAcres", "CreateDate",
    "DateCurrent", "PolygonDateTime", "ComplexName", "GACC",
    "IRWINID", "IncidentTypeCategory", "CreateDateAge",
]

# Acreage tiers, matching Esri's own symbology for consistency with
# anything your users may have seen on the NIFC dashboard.
ACREAGE_TIERS = [
    (1_000,      "0-999"),
    (10_000,     "1,000-9,999"),
    (50_000,     "10,000-49,999"),
    (300_000,    "50,000-299,999"),
    (float("inf"), "300,000 or more"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _epoch_to_iso(ms):
    """ArcGIS returns dates as epoch milliseconds; convert to readable UTC string."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return None


def _acreage_tier(acres):
    if acres is None:
        return "0-999"
    for threshold, label in ACREAGE_TIERS:
        if acres < threshold:
            return label
    return ACREAGE_TIERS[-1][1]


def _query_layer(layer_id: int, out_fields: list, where: str = "1=1") -> list:
    """Query an ArcGIS FeatureServer layer as GeoJSON, paginating if needed."""
    url = f"{WILDFIRE_SERVICE}/{layer_id}/query"
    features = []
    offset = 0
    page_size = 2000

    while True:
        params = {
            "where": where,
            "outFields": ",".join(out_fields),
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            raise RuntimeError(f"ArcGIS error on layer {layer_id}: {payload['error']}")

        batch = payload.get("features", [])
        features.extend(batch)

        if len(batch) < page_size:
            break
        offset += page_size

    return features


def _clean_incidents(features: list) -> list:
    records = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry")
        if not geom or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        if lat is None or lon is None:
            continue

        acres = props.get("DailyAcres") or props.get("CalculatedAcres") or 0

        records.append({
            "name":            props.get("IncidentName") or "Unnamed Incident",
            "type":            props.get("IncidentTypeCategory"),   # WF / RX / CX
            "lat":             lat,
            "lon":             lon,
            "acres":           acres,
            "tier":            _acreage_tier(acres),
            "contained_pct":   props.get("PercentContained"),
            "discovery_date":  _epoch_to_iso(props.get("FireDiscoveryDateTime")),
            "discovery_age":   props.get("FireDiscoveryAge"),
            "county":          props.get("POOCounty"),
            "state":           props.get("POOState"),
            "cause":           props.get("FireCauseGeneral") or props.get("FireCause"),
            "gacc":            props.get("GACC"),
            "personnel":       props.get("TotalIncidentPersonnel"),
            "complexity":      props.get("FireMgmtComplexity"),
            "residences_lost": props.get("ResidencesDestroyed"),
            "structures_lost": props.get("OtherStructuresDestroyed"),
            "injuries":        props.get("Injuries"),
            "fatalities":      props.get("Fatalities"),
            "irwin_id":        props.get("IrwinID"),
        })
    return records


def _clean_perimeters(features: list, incidents_by_irwin: dict) -> list:
    records = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        irwin = props.get("IRWINID")
        matched = incidents_by_irwin.get(irwin, {})

        records.append({
            "name":          props.get("IncidentName") or matched.get("name") or "Unnamed Incident",
            "category":      props.get("FeatureCategory"),   # Wildfire Daily Fire Perimeter / Prescribed Fire
            "acres":         props.get("GISAcres"),
            "date_current":  _epoch_to_iso(props.get("DateCurrent")),
            "gacc":          props.get("GACC"),
            "contained_pct": matched.get("contained_pct"),
            "geometry":      geom,
        })
    return records


# ---------------------------------------------------------------------------
# 1. Main build function  —  call this at map build time
# ---------------------------------------------------------------------------

def build_wildfire_json(output_dir: str) -> dict:
    """
    Fetch current wildfire incidents + perimeters and write
    wildfire_data.json into output_dir.

    Parameters
    ----------
    output_dir : str
        Directory where your map HTML lives (wildfire_data.json written here).

    Returns
    -------
    dict  —  summary of record counts
    """
    out_path = Path(output_dir) / "wildfire_data.json"
    summary = {}

    try:
        incident_features = _query_layer(0, INCIDENT_FIELDS)
        incidents = _clean_incidents(incident_features)
        summary["incidents"] = len(incidents)
        print(f"  [Wildfire] Incidents : {len(incidents):4d} records")
    except Exception as e:
        incidents = []
        summary["incidents"] = f"ERROR: {e}"
        print(f"  [Wildfire] Incidents : ERROR — {e}")

    incidents_by_irwin = {r["irwin_id"]: r for r in incidents if r.get("irwin_id")}

    try:
        perimeter_features = _query_layer(1, PERIMETER_FIELDS)
        perimeters = _clean_perimeters(perimeter_features, incidents_by_irwin)
        summary["perimeters"] = len(perimeters)
        print(f"  [Wildfire] Perimeters: {len(perimeters):4d} records")
    except Exception as e:
        perimeters = []
        summary["perimeters"] = f"ERROR: {e}"
        print(f"  [Wildfire] Perimeters: ERROR — {e}")

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "incidents": incidents,
        "perimeters": perimeters,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  [Wildfire] Written → {out_path}")
    return summary


# ---------------------------------------------------------------------------
# 2. JS + toggle button  —  matches your existing MacroElement / SPC pattern
# ---------------------------------------------------------------------------

def get_wildfire_toggle_js(json_path: str = "wildfire_data.json") -> str:
    """
    Returns the HTML/JS snippet to inject into your map.

    Parameters
    ----------
    json_path : str
        Relative path from the HTML file to wildfire_data.json.
        Default 'wildfire_data.json' assumes same directory.
    """
    return f"""
<!-- ===== Live Wildfires Layer (NIFC / IRWIN) ===== -->
<style>
  #wildfire-toggle-btn {{
    position: absolute;
    bottom: 74px;
    left: 10px;
    z-index: 1000;
    background: #6e2a0e;
    color: #ffffff;
    border: 2px solid #ff7a1a;
    border-radius: 6px;
    padding: 7px 14px;
    font-family: 'Aptos Display', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    transition: background 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }}
  #wildfire-toggle-btn:hover  {{ background: #ff7a1a; border-color: #ffcf80; }}
  #wildfire-toggle-btn.wf-off {{ background: #4a4a4a; border-color: #888; color: #ccc; }}
  #wildfire-toggle-btn .wf-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: #ff5722; display: inline-block; flex-shrink: 0;
  }}
  #wildfire-toggle-btn.wf-off .wf-dot {{ background: #888; }}
  .wf-count {{ font-size: 11px; opacity: 0.8; }}
</style>

<button id="wildfire-toggle-btn" title="Toggle live wildfire incidents + perimeters">
  <span class="wf-dot"></span>
  Wildfires
  <span class="wf-count" id="wf-count">loading…</span>
</button>

<script>
(function() {{

  function waitForMap(cb) {{
    const poll = setInterval(function() {{
      const keys = Object.keys(window).filter(k => {{
        try {{ return window[k] && window[k]._leaflet_id !== undefined; }} catch(e) {{ return false; }}
      }});
      if (keys.length) {{ clearInterval(poll); cb(window[keys[0]]); }}
    }}, 150);
  }}

  waitForMap(function(map) {{

    const wfLayer  = L.layerGroup().addTo(map);
    let   visible  = true;
    let   total    = 0;

    // ── Tier -> radius / color ──────────────────────────────────────────────

    const TIER_RADIUS = {{
      '0-999':             7,
      '1,000-9,999':        11,
      '10,000-49,999':      15,
      '50,000-299,999':     19,
      '300,000 or more':    24,
    }};

    function containmentColor(pct) {{
      if (pct == null) return '#9e9e9e';       // unknown
      if (pct >= 90)   return '#2e7d32';       // near fully contained
      if (pct >= 50)   return '#f9a825';       // partially contained
      return '#d32f2f';                        // low containment / active
    }}

    function typeBorder(type) {{
      if (type === 'RX') return '#4caf50';     // prescribed fire
      if (type === 'CX') return '#7b1fa2';     // incident complex
      return '#ffffff';                        // wildfire
    }}

    function incidentIcon(rec) {{
      const r = TIER_RADIUS[rec.tier] || 9;
      const fill = containmentColor(rec.contained_pct);
      const border = typeBorder(rec.type);
      const pulse = (rec.discovery_age === 0)
        ? 'box-shadow:0 0 0 4px rgba(255,87,34,0.35);' : '';
      return L.divIcon({{
        className: '',
        html: `<div style="width:${{r*2}}px;height:${{r*2}}px;border-radius:50%;
          background:${{fill}};border:2px solid ${{border}};
          box-shadow:0 1px 4px rgba(0,0,0,.5);${{pulse}}"></div>`,
        iconSize: [r*2, r*2], iconAnchor: [r, r],
      }});
    }}

    // ── Popups ───────────────────────────────────────────────────────────────

    function fmtAcres(a) {{
      if (a == null) return 'N/A';
      return Math.round(a).toLocaleString() + ' ac';
    }}

    function fmtPct(p) {{
      return (p == null) ? 'Unknown' : Math.round(p) + '%';
    }}

    function typeLabel(t) {{
      return {{WF: 'Wildfire', RX: 'Prescribed Fire', CX: 'Incident Complex'}}[t] || t || 'Incident';
    }}

    function buildIncidentPopup(r) {{
      const color = containmentColor(r.contained_pct);
      const loss = (r.residences_lost || r.structures_lost)
        ? `<br>🏠 ${{r.residences_lost || 0}} residences, ${{r.structures_lost || 0}} other structures lost`
        : '';
      const casualties = (r.fatalities || r.injuries)
        ? `<br>⚠️ ${{r.fatalities || 0}} fatalities, ${{r.injuries || 0}} injuries`
        : '';
      return `
        <div style="font-family:'Aptos Display',sans-serif;min-width:200px;max-width:270px;font-size:12px;line-height:1.6;">
          <div style="background:${{color}};color:#fff;padding:6px 10px;
            border-radius:4px 4px 0 0;font-weight:700;font-size:13px;">
            🔥 ${{r.name}}
          </div>
          <div style="padding:8px 10px;background:#f9f9f9;border-radius:0 0 4px 4px;">
            <span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.4px;">${{typeLabel(r.type)}}</span><br>
            <span style="font-size:16px;font-weight:700;color:${{color}};">${{fmtAcres(r.acres)}}</span>
            &nbsp;·&nbsp; ${{fmtPct(r.contained_pct)}} contained<br>
            <span style="color:#555;">📅 Discovered ${{r.discovery_date || 'N/A'}}</span><br>
            <span style="color:#333;">📍 ${{r.county || ''}} Co., ${{r.state || ''}}</span><br>
            <span style="color:#666;">Cause: ${{r.cause || 'Under investigation'}}</span>
            ${{r.personnel ? '<br>👥 ' + r.personnel + ' personnel assigned' : ''}}
            ${{loss}}${{casualties}}
          </div>
        </div>`;
    }}

    function buildPerimeterPopup(p) {{
      const isRx = (p.category || '').toLowerCase().includes('prescribed');
      const color = isRx ? '#4caf50' : containmentColor(p.contained_pct);
      return `
        <div style="font-family:'Aptos Display',sans-serif;min-width:190px;max-width:250px;font-size:12px;line-height:1.6;">
          <div style="background:${{color}};color:#fff;padding:6px 10px;
            border-radius:4px 4px 0 0;font-weight:700;font-size:13px;">
            🗺️ ${{p.name}}
          </div>
          <div style="padding:8px 10px;background:#f9f9f9;border-radius:0 0 4px 4px;">
            <span style="font-size:15px;font-weight:700;color:${{color}};">${{fmtAcres(p.acres)}}</span>
            ${{(p.contained_pct != null) ? ' · ' + fmtPct(p.contained_pct) + ' contained' : ''}}<br>
            <span style="color:#555;">Perimeter as of ${{p.date_current || 'N/A'}}</span><br>
            <span style="color:#666;">${{p.category || ''}}</span>
          </div>
        </div>`;
    }}

    // ── Fetch + render ───────────────────────────────────────────────────────

    fetch('{json_path}')
      .then(function(r) {{
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }})
      .then(function(data) {{
        wfLayer.clearLayers();
        total = 0;

        // Perimeters first (so incident points draw on top)
        (data.perimeters || []).forEach(function(p) {{
          if (!p.geometry) return;
          const isRx = (p.category || '').toLowerCase().includes('prescribed');
          const color = isRx ? '#4caf50' : containmentColor(p.contained_pct);
          const layer = L.geoJSON(p.geometry, {{
            style: {{
              color: color,
              weight: 1.5,
              fillColor: color,
              fillOpacity: 0.25,
            }}
          }});
          layer.bindPopup(buildPerimeterPopup(p), {{maxWidth: 260}});
          layer.bindTooltip(p.name, {{sticky: true}});
          wfLayer.addLayer(layer);
        }});

        // Incident points
        (data.incidents || []).forEach(function(r) {{
          if (r.lat == null || r.lon == null) return;
          total++;
          const marker = L.marker([r.lat, r.lon], {{icon: incidentIcon(r)}});
          marker.bindPopup(buildIncidentPopup(r), {{maxWidth: 280}});
          marker.bindTooltip(r.name + ' — ' + fmtAcres(r.acres),
            {{direction: 'top', offset: [0, -8], sticky: true}});
          wfLayer.addLayer(marker);
        }});

        document.getElementById('wf-count').textContent =
          '(' + total + (data.generated_utc ? ' · ' + data.generated_utc : '') + ')';
      }})
      .catch(function(err) {{
        console.error('[Wildfire] fetch error:', err);
        document.getElementById('wf-count').textContent = '(unavailable)';
      }});

    // ── Toggle ───────────────────────────────────────────────────────────────

    document.getElementById('wildfire-toggle-btn').addEventListener('click', function() {{
      visible = !visible;
      if (visible) {{
        wfLayer.addTo(map);
        this.classList.remove('wf-off');
      }} else {{
        map.removeLayer(wfLayer);
        this.classList.add('wf-off');
      }}
    }});

  }});  // end waitForMap

}})();
</script>
<!-- ===== / Live Wildfires Layer ===== -->
"""


# ---------------------------------------------------------------------------
# 3. Post-process saved map HTML  —  same pattern as your other injections
# ---------------------------------------------------------------------------

def inject_into_map_html(html_path: str, json_path: str = "wildfire_data.json") -> None:
    """
    Inject the wildfire toggle JS into a saved Folium HTML file.
    Call this after map.save(html_path).

    Parameters
    ----------
    html_path : str   Path to your saved map HTML.
    json_path : str   Relative path from HTML to wildfire_data.json.
    """
    snippet = get_wildfire_toggle_js(json_path=json_path)
    path = Path(html_path)

    html = path.read_text(encoding="utf-8")
    if "wildfire-toggle-btn" in html:
        print(f"  [Wildfire] Already injected — skipping {path.name}")
        return

    html = html.replace("</body>", snippet + "\n</body>")
    path.write_text(html, encoding="utf-8")
    print(f"  [Wildfire] Injected toggle into {path.name}")


# ---------------------------------------------------------------------------
# 4. Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile

    print("[Wildfire test] Fetching current incidents + perimeters...")
    summary = build_wildfire_json(output_dir=tempfile.gettempdir())
    print(f"\n[Wildfire test] Summary: {summary}")

    if "ipykernel" not in sys.modules:
        out = Path(tempfile.gettempdir()) / "wildfire_data.json"
        print(f"\n[Wildfire test] JSON written to: {out}")
        print("               Open it to verify structure before integrating.")
