"""
spc_live_reports.py
--------------------
SPC Storm Reports — Pre-bake Module
Fetches SPC data at map build time and writes spc_reports.json
alongside your HTML, exactly like your other external data files.

Usage in your map build script
-------------------------------
from spc_live_reports import build_spc_json, get_spc_toggle_js, inject_into_map_html

# 1. Fetch + write JSON next to your map HTML
build_spc_json(output_dir="C:/path/to/your/map/output")

# 2. After map.save(), inject the toggle button + JS
inject_into_map_html("C:/path/to/your/map/output/live_map.html")

Filters applied
---------------
  Tornado  : all EF ratings
  Hail     : size >= 1.0 inch
  Wind     : speed >= 65 mph
"""

import io
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPC_BASE = "https://www.spc.noaa.gov/climo/reports/{date}_rpts_{peril}.csv"

PERIL_CONFIGS = {
    "torn": {
        "label":      "Tornado",
        "mag_col":    "F-Scale",
        "mag_filter": None,       # all EF ratings
        "color":      "#d62728",
    },
    "hail": {
        "label":      "Hail",
        "mag_col":    "Size",
        "mag_filter": 1.0,        # >= 1.0 inch (after hundredths conversion)
        "color":      "#2ca02c",
    },
    "wind": {
        "label":      "Wind",
        "mag_col":    "Speed",
        "mag_filter": 65,         # >= 65 mph
        "color":      "#1f77b4",
    },
}

SPC_COLS = {
    "torn": ["Time", "F-Scale", "Location", "County", "State", "Lat", "Lon", "Comments"],
    "hail": ["Time", "Size",    "Location", "County", "State", "Lat", "Lon", "Comments"],
    "wind": ["Time", "Speed",   "Location", "County", "State", "Lat", "Lon", "Comments"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _yesterday() -> str:
    """Returns yesterday's date in YYMMDD format (UTC)."""
    d = datetime.now(timezone.utc) - timedelta(days=1)
    return d.strftime("%y%m%d")


def _date_label(date_str: str) -> str:
    """Converts YYMMDD to a readable label e.g. 'Jun 21, 2026'."""
    d = datetime.strptime(date_str, "%y%m%d")
    return d.strftime("%b %d, %Y")


def _fetch_spc_csv(peril: str, date_str: str) -> pd.DataFrame:
    url = SPC_BASE.format(date=date_str, peril=peril)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    cols = SPC_COLS[peril]
    df = pd.read_csv(
        io.StringIO(resp.text),
        header=0,
        names=cols,
        on_bad_lines="skip",
    )
    return df


def _filter_and_clean(df: pd.DataFrame, peril: str) -> pd.DataFrame:
    cfg = PERIL_CONFIGS[peril]
    mag_col = cfg["mag_col"]

    df = df.dropna(subset=["Lat", "Lon"])
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df = df.dropna(subset=["Lat", "Lon"])
    df = df[(df["Lat"] != 0) & (df["Lon"] != 0)]

    if cfg["mag_filter"] is not None:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")

        # SPC hail CSV encodes size in hundredths of an inch (e.g. 150 = 1.50 in)
        # Values >= 10 are in hundredths; values < 10 are already in inches (rare legacy rows)
        if peril == "hail":
            df[mag_col] = df[mag_col].apply(
                lambda x: x / 100.0 if pd.notna(x) and x >= 10 else x
            )

        df = df[df[mag_col] >= cfg["mag_filter"]]

    if peril == "torn":
        df["magnitude_display"] = df[mag_col].apply(
            lambda x: f"EF{int(x)}" if pd.notna(x) and str(x).strip() not in ["", "UNK"] else "EFU"
        )
    elif peril == "hail":
        df["magnitude_display"] = df[mag_col].apply(
            lambda x: f"{float(x):.2f} in" if pd.notna(x) else "N/A"
        )
    else:
        df["magnitude_display"] = df[mag_col].apply(
            lambda x: f"{int(x)} mph" if pd.notna(x) else "N/A"
        )

    df["peril"]  = peril
    df["label"]  = cfg["label"]
    df["color"]  = cfg["color"]

    df = df.rename(columns={
        "Lat": "lat", "Lon": "lon",
        "Time": "time", "Location": "location",
        "County": "county", "State": "state",
        "Comments": "comments",
        mag_col: "magnitude",
    })

    keep = ["lat", "lon", "time", "magnitude", "magnitude_display",
            "location", "county", "state", "comments", "peril", "label", "color"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# 1. Main build function  —  call this at map build time
# ---------------------------------------------------------------------------

def build_spc_json(output_dir: str, date_str: str = None) -> dict:
    """
    Fetch SPC reports and write spc_reports.json into output_dir.

    Parameters
    ----------
    output_dir : str
        Directory where your map HTML lives (spc_reports.json written here).
    date_str   : str, optional
        YYMMDD date string. Defaults to yesterday.

    Returns
    -------
    dict  —  summary of record counts per peril
    """
    date_str = date_str or _yesterday()
    out_path = Path(output_dir) / "spc_reports.json"

    payload = {
        "date":       date_str,
        "date_label": _date_label(date_str),
        "torn":       [],
        "hail":       [],
        "wind":       [],
    }

    summary = {}
    for peril in ["torn", "hail", "wind"]:
        try:
            df = _fetch_spc_csv(peril, date_str)
            df = _filter_and_clean(df, peril)
            # Convert NaN → None so JSON serialises cleanly
            records = df.where(pd.notna(df), None).to_dict(orient="records")
            payload[peril] = records
            summary[peril] = len(records)
            print(f"  [SPC] {peril.upper():4s}: {len(records):4d} records  ({date_str})")
        except Exception as e:
            summary[peril] = f"ERROR: {e}"
            print(f"  [SPC] {peril.upper():4s}: ERROR — {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  [SPC] Written → {out_path}")
    return summary


# ---------------------------------------------------------------------------
# 2. JS + toggle button  —  matches your existing MacroElement pattern
# ---------------------------------------------------------------------------

def get_spc_toggle_js(json_path: str = "spc_reports.json") -> str:
    """
    Returns the HTML/JS snippet to inject into your map.

    Parameters
    ----------
    json_path : str
        Relative path from the HTML file to spc_reports.json.
        Default 'spc_reports.json' assumes same directory.
    """
    return f"""
<!-- ===== SPC Storm Reports Layer ===== -->
<style>
  #spc-toggle-btn {{
    position: absolute;
    bottom: 28px;
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
  #spc-toggle-btn:hover  {{ background: #1a9f9c; border-color: #80d2eb; }}
  #spc-toggle-btn.spc-off {{ background: #4a4a4a; border-color: #888; color: #ccc; }}
  #spc-toggle-btn .spc-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: #cd9036; display: inline-block; flex-shrink: 0;
  }}
  #spc-toggle-btn.spc-off .spc-dot {{ background: #888; }}
  .spc-count {{ font-size: 11px; opacity: 0.8; }}
</style>

<button id="spc-toggle-btn" title="Toggle SPC storm reports">
  <span class="spc-dot"></span>
  SPC Reports
  <span class="spc-count" id="spc-count">loading…</span>
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

    const spcLayer  = L.layerGroup().addTo(map);
    let   visible   = true;
    let   total     = 0;

    // ── Marker factories ────────────────────────────────────────────────────

    function tornIcon(color) {{
      return L.divIcon({{
        className: '',
        html: `<div style="width:0;height:0;
          border-left:7px solid transparent;
          border-right:7px solid transparent;
          border-top:14px solid ${{color}};
          filter:drop-shadow(0 1px 2px rgba(0,0,0,.5))"></div>`,
        iconSize: [14,14], iconAnchor: [7,14],
      }});
    }}

    function hailIcon(color, size) {{
      const r = Math.min(14, Math.max(6, Math.round(6 + (size - 1) * 2.67)));
      return L.divIcon({{
        className: '',
        html: `<div style="width:${{r*2}}px;height:${{r*2}}px;border-radius:50%;
          background:${{color}};border:1.5px solid rgba(255,255,255,.6);
          box-shadow:0 1px 3px rgba(0,0,0,.4)"></div>`,
        iconSize: [r*2,r*2], iconAnchor: [r,r],
      }});
    }}

    function windIcon(color) {{
      return L.divIcon({{
        className: '',
        html: `<div style="width:11px;height:11px;background:${{color}};
          border:1.5px solid rgba(255,255,255,.6);
          box-shadow:0 1px 3px rgba(0,0,0,.4);
          transform:rotate(45deg)"></div>`,
        iconSize: [11,11], iconAnchor: [5,5],
      }});
    }}

    // ── Popup ────────────────────────────────────────────────────────────────

    function buildPopup(r, dateLabel) {{
      const bg = {{ torn:'#d62728', hail:'#2ca02c', wind:'#1f77b4' }};
      const em = {{ torn:'🌪️', hail:'🧊', wind:'💨' }};
      const cmt = r.comments
        ? `<br><span style="color:#666;font-size:11px;font-style:italic;">
            ${{r.comments.substring(0,130)}}${{r.comments.length>130?'…':''}}</span>`
        : '';
      return `
        <div style="font-family:'Aptos Display',sans-serif;min-width:185px;max-width:245px;font-size:12px;line-height:1.6;">
          <div style="background:${{bg[r.peril]}};color:#fff;padding:6px 10px;
            border-radius:4px 4px 0 0;font-weight:700;font-size:13px;
            display:flex;align-items:center;gap:6px;">
            ${{em[r.peril]}} ${{r.label}}
          </div>
          <div style="padding:8px 10px;background:#f9f9f9;border-radius:0 0 4px 4px;">
            <span style="font-size:16px;font-weight:700;color:${{bg[r.peril]}};">${{r.magnitude_display}}</span><br>
            <span style="color:#555;">⏰ ${{r.time || 'N/A'}} UTC &nbsp;·&nbsp; ${{dateLabel}}</span><br>
            <span style="color:#333;">📍 ${{r.location || ''}}, ${{r.county || ''}} Co., ${{r.state || ''}}</span>
            ${{cmt}}
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
        spcLayer.clearLayers();
        total = 0;
        const dateLabel = data.date_label || '';

        ['torn','hail','wind'].forEach(function(peril) {{
          const records = data[peril];
          if (!records || !records.length) return;

          records.forEach(function(r) {{
            if (r.lat == null || r.lon == null) return;
            total++;

            let marker;
            if      (peril === 'torn') {{ marker = L.marker([r.lat, r.lon], {{icon: tornIcon(r.color)}}); }}
            else if (peril === 'hail') {{ marker = L.marker([r.lat, r.lon], {{icon: hailIcon(r.color, parseFloat(r.magnitude)||1)}}); }}
            else                       {{ marker = L.marker([r.lat, r.lon], {{icon: windIcon(r.color)}}); }}

            marker.bindPopup(buildPopup(r, dateLabel), {{maxWidth: 260}});
            marker.bindTooltip(r.label + ': ' + r.magnitude_display,
              {{direction:'top', offset:[0,-8], sticky:true}});
            spcLayer.addLayer(marker);
          }});
        }});

        document.getElementById('spc-count').textContent = '(' + total + ')';
      }})
      .catch(function(err) {{
        console.error('[SPC] fetch error:', err);
        document.getElementById('spc-count').textContent = '(unavailable)';
      }});

    // ── Toggle ───────────────────────────────────────────────────────────────

    document.getElementById('spc-toggle-btn').addEventListener('click', function() {{
      visible = !visible;
      if (visible) {{
        spcLayer.addTo(map);
        this.classList.remove('spc-off');
        document.getElementById('spc-count').textContent = '(' + total + ')';
      }} else {{
        map.removeLayer(spcLayer);
        this.classList.add('spc-off');
        document.getElementById('spc-count').textContent = '(off)';
      }}
    }});

  }});  // end waitForMap

}})();
</script>
<!-- ===== / SPC Storm Reports Layer ===== -->
"""


# ---------------------------------------------------------------------------
# 3. Post-process saved map HTML  —  same pattern as your other injections
# ---------------------------------------------------------------------------

def inject_into_map_html(html_path: str, json_path: str = "spc_reports.json") -> None:
    """
    Inject the SPC toggle JS into a saved Folium HTML file.
    Call this after map.save(html_path).

    Parameters
    ----------
    html_path : str   Path to your saved map HTML.
    json_path : str   Relative path from HTML to spc_reports.json.
    """
    snippet = get_spc_toggle_js(json_path=json_path)
    path    = Path(html_path)

    html = path.read_text(encoding="utf-8")
    if "spc-toggle-btn" in html:
        print(f"  [SPC] Already injected — skipping {path.name}")
        return

    html = html.replace("</body>", snippet + "\n</body>")
    path.write_text(html, encoding="utf-8")
    print(f"  [SPC] Injected toggle into {path.name}")


# ---------------------------------------------------------------------------
# 4. Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile

    print("[SPC test] Fetching yesterday's reports...")
    summary = build_spc_json(output_dir=tempfile.gettempdir())
    print(f"\n[SPC test] Summary: {summary}")

    if "ipykernel" not in sys.modules:
        out = Path(tempfile.gettempdir()) / "spc_reports.json"
        print(f"\n[SPC test] JSON written to: {out}")
        print("           Open it to verify structure before integrating.")
