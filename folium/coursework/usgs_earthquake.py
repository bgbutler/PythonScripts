"""
usgs_earthquakes.py
--------------------
USGS Earthquake Reports — Pre-bake Module
Fetches USGS earthquake data at map build time and writes usgs_earthquakes.json
alongside your HTML, exactly like spc_live_reports.py.

Spec
----
  Source    : USGS FDSN API (GeoJSON)
  Window    : Past 7 days
  Magnitude : M4.0+
  Scope     : CONUS (bounding box)
  Markers   : Circle, scaled by magnitude, colored by depth
  Popup     : Magnitude, depth, location, time, USGS event link

Usage in your map build script
-------------------------------
from usgs_earthquakes import build_usgs_json, inject_into_map_html

# 1. Fetch + write JSON next to your map HTML
build_usgs_json(output_dir=folder)

# 2. After map.save(), inject the toggle button + JS
inject_into_map_html(os.path.join(folder, "map_liveV2.html"))
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# CONUS bounding box
CONUS_BBOX = {
    "minlatitude":  24.0,
    "maxlatitude":  50.0,
    "minlongitude": -125.0,
    "maxlongitude": -65.0,
}

MIN_MAG    = 4.0
DAYS_BACK  = 7

# Depth color bins (km) — shallow = more damaging = warmer color
DEPTH_COLORS = [
    (0,   20,  "#d62728"),   # 0–20 km    red      — very shallow
    (20,  70,  "#ff7f0e"),   # 20–70 km   orange   — shallow
    (70,  150, "#ffd700"),   # 70–150 km  yellow   — intermediate
    (150, 300, "#2ca02c"),   # 150–300 km green    — deep
    (300, 999, "#1f77b4"),   # 300+ km    blue     — very deep
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_range() -> tuple:
    """Returns (starttime, endtime) ISO strings for past DAYS_BACK days."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def _depth_color(depth_km: float) -> str:
    """Returns hex color for a given depth in km."""
    if depth_km is None:
        return "#888888"
    for lo, hi, color in DEPTH_COLORS:
        if lo <= depth_km < hi:
            return color
    return "#1f77b4"


def _mag_radius(mag: float) -> int:
    """
    Scale circle radius by magnitude for JS divIcon.
    M4.0 → 6px, M5.0 → 9px, M6.0 → 13px, M7.0+ → 18px
    """
    if mag is None:
        return 6
    return max(6, min(20, round(3 + (mag - 4.0) * 3.5)))


def _fetch_usgs() -> list:
    """Fetch USGS GeoJSON and return list of cleaned event dicts."""
    starttime, endtime = _date_range()

    params = {
        "format":       "geojson",
        "starttime":    starttime,
        "endtime":      endtime,
        "minmagnitude": MIN_MAG,
        "orderby":      "time",
        **CONUS_BBOX,
    }

    resp = requests.get(USGS_API, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None, None])

        lon, lat, depth = coords[0], coords[1], coords[2]
        mag   = props.get("mag")
        place = props.get("place", "")
        ts    = props.get("time")
        url   = props.get("url", "")
        alert = props.get("alert")        # PAGER: green/yellow/orange/red or None
        tsunami = props.get("tsunami", 0)

        # Format time
        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            time_display = dt.strftime("%b %d, %Y %H:%M UTC")
        else:
            time_display = "N/A"

        # Magnitude display
        mag_display = f"M{mag:.1f}" if mag is not None else "M?"

        # Depth display
        depth_display = f"{depth:.1f} km" if depth is not None else "N/A"

        records.append({
            "lat":           lat,
            "lon":           lon,
            "depth_km":      round(depth, 1) if depth is not None else None,
            "magnitude":     mag,
            "mag_display":   mag_display,
            "depth_display": depth_display,
            "time_display":  time_display,
            "place":         place,
            "url":           url,
            "alert":         alert,
            "tsunami":       tsunami,
            "color":         _depth_color(depth),
            "radius":        _mag_radius(mag),
        })

    return records


# ---------------------------------------------------------------------------
# 1. Main build function
# ---------------------------------------------------------------------------

def build_usgs_json(output_dir: str) -> dict:
    """
    Fetch USGS earthquakes and write usgs_earthquakes.json into output_dir.

    Parameters
    ----------
    output_dir : str
        Directory where your map HTML lives.

    Returns
    -------
    dict  —  { "count": N, "starttime": ..., "endtime": ... }
    """
    out_path = Path(output_dir) / "usgs_earthquakes.json"
    starttime, endtime = _date_range()

    try:
        records = _fetch_usgs()
        payload = {
            "generated":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "starttime":  starttime,
            "endtime":    endtime,
            "min_mag":    MIN_MAG,
            "count":      len(records),
            "events":     records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"  [USGS] EQ M{MIN_MAG}+: {len(records):4d} events  "
              f"({starttime[:10]} → {endtime[:10]})")
        print(f"  [USGS] Written → {out_path}")
        return {"count": len(records), "starttime": starttime, "endtime": endtime}

    except Exception as e:
        print(f"  [USGS] ERROR — {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 2. JS + toggle button
# ---------------------------------------------------------------------------

def get_usgs_toggle_js(json_path: str = "usgs_earthquakes.json") -> str:
    """
    Returns the HTML/JS snippet to inject into your map.

    Parameters
    ----------
    json_path : str
        Relative path from the HTML file to usgs_earthquakes.json.
    """
    return f"""
<!-- ===== USGS Earthquake Layer ===== -->
<style>
  #usgs-toggle-btn {{
    position: absolute;
    bottom: 122px;
    left: 10px;
    z-index: 1000;
    background: #02473b;
    color: #ffffff;
    border: 2px solid #1a9f9c;
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
  #usgs-toggle-btn:hover   {{ background: #1a9f9c; border-color: #80d2eb; }}
  #usgs-toggle-btn.eq-off  {{ background: #4a4a4a; border-color: #888; color: #ccc; }}
  #usgs-toggle-btn .eq-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: #d62728; display: inline-block; flex-shrink: 0;
  }}
  #usgs-toggle-btn.eq-off .eq-dot {{ background: #888; }}
  .eq-count {{ font-size: 11px; opacity: 0.8; }}

  /* Depth legend */
  #usgs-depth-legend {{
    position: absolute;
    bottom: 170px;
    left: 10px;
    z-index: 999;
    background: rgba(2, 71, 59, 0.92);
    border: 1px solid #1a9f9c;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Aptos Display', sans-serif;
    font-size: 11px;
    color: #fff;
    display: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    min-width: 130px;
  }}
  #usgs-depth-legend.legend-visible {{ display: block; }}
  #usgs-depth-legend .legend-title {{
    font-size: 12px; font-weight: 700;
    margin-bottom: 6px; color: #80d2eb;
  }}
  .dleg-row {{ display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }}
  .dleg-swatch {{
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.4);
  }}
</style>

<!-- Toggle button -->
<button id="usgs-toggle-btn" title="Toggle USGS earthquakes (past 7 days, M4.0+)">
  <span class="eq-dot"></span>
  Earthquakes
  <span class="eq-count" id="usgs-count">loading…</span>
</button>

<!-- Depth legend (shown when layer is on) -->
<div id="usgs-depth-legend">
  <div class="legend-title">Depth (km)</div>
  <div class="dleg-row"><span class="dleg-swatch" style="background:#d62728"></span>0 – 20</div>
  <div class="dleg-row"><span class="dleg-swatch" style="background:#ff7f0e"></span>20 – 70</div>
  <div class="dleg-row"><span class="dleg-swatch" style="background:#ffd700"></span>70 – 150</div>
  <div class="dleg-row"><span class="dleg-swatch" style="background:#2ca02c"></span>150 – 300</div>
  <div class="dleg-row"><span class="dleg-swatch" style="background:#1f77b4"></span>300+</div>
  <div style="margin-top:6px;color:#aaa;font-size:10px;">Circle size = magnitude</div>
</div>

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

    const eqLayer = L.layerGroup().addTo(map);
    let visible   = true;
    let total     = 0;

    // ── Marker factory ────────────────────────────────────────────────────────

    function eqIcon(color, radius) {{
      const d = radius * 2;
      return L.divIcon({{
        className: '',
        html: `<div style="
          width:${{d}}px; height:${{d}}px;
          border-radius: 50%;
          background: ${{color}};
          border: 1.5px solid rgba(255,255,255,0.7);
          box-shadow: 0 0 4px rgba(0,0,0,0.5);
          opacity: 0.85;
        "></div>`,
        iconSize:   [d, d],
        iconAnchor: [radius, radius],
      }});
    }}

    // ── Popup ─────────────────────────────────────────────────────────────────

    function buildPopup(e) {{
      const alertBadge = e.alert
        ? `<span style="
            background:${{e.alert}};color:${{e.alert==='yellow'?'#333':'#fff'}};
            padding:1px 7px;border-radius:3px;font-size:11px;font-weight:700;
            text-transform:uppercase;margin-left:6px;">
            PAGER: ${{e.alert}}</span>`
        : '';
      const tsunamiBadge = e.tsunami
        ? `<span style="background:#1f77b4;color:#fff;padding:1px 7px;
            border-radius:3px;font-size:11px;font-weight:700;margin-left:4px;">
            TSUNAMI</span>`
        : '';
      return `
        <div style="font-family:'Aptos Display',sans-serif;
          min-width:200px;max-width:260px;font-size:12px;line-height:1.6;">
          <div style="background:#7b2d2d;color:#fff;padding:6px 10px;
            border-radius:4px 4px 0 0;font-weight:700;font-size:14px;
            display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
            🌍 ${{e.mag_display}}
            ${{alertBadge}}${{tsunamiBadge}}
          </div>
          <div style="padding:8px 10px;background:#f9f9f9;border-radius:0 0 4px 4px;">
            <span style="color:#555;">📍 ${{e.place || 'N/A'}}</span><br>
            <span style="color:#555;">⏰ ${{e.time_display}}</span><br>
            <span style="color:#555;">📏 Depth: ${{e.depth_display}}</span><br>
            ${{e.url
              ? `<a href="${{e.url}}" target="_blank"
                  style="color:#1a9f9c;font-size:11px;text-decoration:none;">
                  USGS Event Page ↗</a>`
              : ''}}
          </div>
        </div>`;
    }}

    // ── Fetch + render ────────────────────────────────────────────────────────

    fetch('{json_path}?v=' + Date.now())
      .then(function(r) {{
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }})
      .then(function(data) {{
        eqLayer.clearLayers();
        total = 0;
        const events = data.events || [];

        events.forEach(function(e) {{
          if (e.lat == null || e.lon == null) return;
          total++;

          const marker = L.marker(
            [e.lat, e.lon],
            {{ icon: eqIcon(e.color, e.radius) }}
          );

          marker.bindPopup(buildPopup(e), {{ maxWidth: 280 }});
          marker.bindTooltip(
            e.mag_display + ' — ' + e.place,
            {{ direction: 'top', offset: [0, -(e.radius + 2)], sticky: true }}
          );
          eqLayer.addLayer(marker);
        }});

        document.getElementById('usgs-count').textContent = '(' + total + ')';

        // Show depth legend when layer loads
        document.getElementById('usgs-depth-legend').classList.add('legend-visible');
      }})
      .catch(function(err) {{
        console.error('[USGS EQ] fetch error:', err);
        document.getElementById('usgs-count').textContent = '(unavailable)';
      }});

    // ── Toggle ────────────────────────────────────────────────────────────────

    document.getElementById('usgs-toggle-btn').addEventListener('click', function() {{
      visible = !visible;
      const legend = document.getElementById('usgs-depth-legend');
      if (visible) {{
        eqLayer.addTo(map);
        this.classList.remove('eq-off');
        document.getElementById('usgs-count').textContent = '(' + total + ')';
        legend.classList.add('legend-visible');
      }} else {{
        map.removeLayer(eqLayer);
        this.classList.add('eq-off');
        document.getElementById('usgs-count').textContent = '(off)';
        legend.classList.remove('legend-visible');
      }}
    }});

  }});  // end waitForMap

}})();
</script>
<!-- ===== / USGS Earthquake Layer ===== -->
"""


# ---------------------------------------------------------------------------
# 3. Post-process saved map HTML
# ---------------------------------------------------------------------------

def inject_into_map_html(html_path: str,
                         json_path: str = "usgs_earthquakes.json") -> None:
    """
    Inject the USGS earthquake toggle JS into a saved Folium HTML file.
    Call this after map.save(html_path).

    Parameters
    ----------
    html_path : str   Path to your saved map HTML.
    json_path : str   Relative path from HTML to usgs_earthquakes.json.
    """
    snippet = get_usgs_toggle_js(json_path=json_path)
    path    = Path(html_path)

    html = path.read_text(encoding="utf-8")
    if "usgs-toggle-btn" in html:
        print(f"  [USGS] Already injected — skipping {path.name}")
        return

    html = html.replace("</body>", snippet + "\n</body>")
    path.write_text(html, encoding="utf-8")
    print(f"  [USGS] Injected toggle into {path.name}")


# ---------------------------------------------------------------------------
# 4. Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile

    print("[USGS test] Fetching past 7 days M4.0+ CONUS earthquakes...")
    summary = build_usgs_json(output_dir=tempfile.gettempdir())
    print(f"\n[USGS test] Summary: {summary}")

    if "ipykernel" not in sys.modules:
        out = Path(tempfile.gettempdir()) / "usgs_earthquakes.json"
        print(f"\n[USGS test] JSON written to: {out}")
