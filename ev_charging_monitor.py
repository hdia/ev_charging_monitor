#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Australian EV Charging Atlas (Open Charge Map, AU-wide)
Self-contained folium/Leaflet map with clustering, heatmap, filters, and route search.

Changes in v6:
- Initial view now uses fit-bounds to show all of Australia on load.
- Route polyline made thinner (configurable).
- Snapshot includes per-state counts (state abbreviations).
- Snapshot status lines include colored dots matching legend.
- Added clear knobs to control the route panel position; default stays top-right under Layer Control.
"""

from __future__ import annotations

import os
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np

from dotenv import load_dotenv

import folium
from folium import Map, TileLayer, FeatureGroup, LayerControl, CircleMarker
from folium.plugins import MarkerCluster, HeatMap
from branca.element import MacroElement, Template, Figure

# ============================================================
# 1) CONFIGURATION
# ============================================================
OCM_URL = "https://api.openchargemap.io/v3/poi/"
COUNTRY_CODE = "AU"
MAXRESULTS = 10000
HTTP_TIMEOUT = 90
FAST_KW = 50.0
ROUTE_PROXIMITY_KM = 5.0

# Route line style
ROUTE_LINE_WEIGHT = 2.0  # thinner than before (was 5)

OUTPUT_HTML = Path("outputs/index.html")
BACKUP_CSV = Path("data/processed/ocm_australia_backup.csv")
LATEST_SNAPSHOT_CSV = Path("data/processed/ocm_australia_latest.csv")

# Map start and bounds for Australia
MAP_START = {"lat": -28.0, "lon": 140.0, "zoom": 5}
#MAP_START = {"lat": -25.0, "lon": 140.0, "zoom": 5}
AUS_BOUNDS = [[-44.0, 110.0], [-10.0, 152.0]]  # SW to NE corners
#AUS_BOUNDS = [[-44.0, 112.0], [-10.0, 154.0]]  # SW to NE corners

FONT_FAMILY = "Inter, Arial, sans-serif"
BOX_BG = "rgba(255,255,255,0.78)"
TITLE_PLACEHOLDER = "Australian EV Infrastructure Monitor"
SUBTITLE_PLACEHOLDER = "Mapping the nation’s electric vehicle charging network"
THIRD_TITLE_PLACEHOLDER = "Swinburne University of Technology. Data snapshot from the Open Charge Map API"
HOWTO_FAST_LINE = f"Fast chargers are defined here as sites with ≥ {int(FAST_KW)} kW."

COL_STATUS = {"operational":"#16a34a","partial":"#f59e0b","down":"#ef4444","unknown":"#6b7280"}
COL_FAST = "#2563eb"
COL_PUBLIC = "#10b981"
COL_PRIVATE = "#8b5cf6"

# Route planner panel positioning knobs:
# If ROUTE_PANEL_AUTO is True, the panel will snap under the Layer Control.
# If False, it uses ROUTE_PANEL_TOP_PX/RIGHT_PX as fixed offsets from top-right.
ROUTE_PANEL_AUTO = False
ROUTE_PANEL_TOP_PX = 225
ROUTE_PANEL_RIGHT_PX = 10

# ============================================================
# 2) UI helpers
# ============================================================
def build_transparent_box(css_id: str, html_inner: str, position: str, offsets=(12,12), width_px=None) -> MacroElement:
    x, y = offsets
    if position == "topleft":
        pos_css = f"top: {y}px; left: {x}px;"
    elif position == "topright":
        pos_css = f"top: {y}px; right: {x}px;"
    elif position == "bottomleft":
        pos_css = f"bottom: {y}px; left: {x}px;"
    else:
        pos_css = f"bottom: {y}px; right: {x}px;"
    width_css = f"width:{int(width_px)}px;" if width_px else ""
    template_str = (
        "{% macro html(this, kwargs) %}\n"
        f"<div id=\"{css_id}\" style='"
        f"position: fixed; z-index: 100000; {pos_css}"
        f"background: {BOX_BG}; padding: 10px 12px; border-radius: 8px;"
        "box-shadow: 0 1px 4px rgba(0,0,0,0.2);"
        f"font-family: {FONT_FAMILY}; font-size: 12px; line-height: 1.30; color:#111;"
        f"pointer-events: auto; {width_css}'>"
        f"{html_inner}"
        "</div>\n"
        "<script>(function(){"
        f"var el=document.getElementById('{css_id}');"
        "if(el && el.parentNode && document.body && el.parentNode!==document.body){document.body.appendChild(el);} "
        "})();</script>\n"
        "{% endmacro %}\n"
    )
    macro = MacroElement(); macro._template = Template(template_str)
    return macro

def inject_label_css(map_obj: Map):
    css = f"""
    <style>
      .label-tooltip {{
        background: transparent;
        border: none;
        box-shadow: none;
        color: #111111;
        font-family: {FONT_FAMILY};
        font-size: 10px;
        font-weight: 400;
        text-shadow: 0 0 2px rgba(255,255,255,0.95), 0 0 6px rgba(255,255,255,0.95);
        pointer-events: none;
      }}
    </style>
    """
    m = MacroElement()
    m._template = Template("{% macro html(this, kwargs) %}" + css + "{% endmacro %}")
    map_obj.get_root().add_child(m)

def sum_icon_create_function_js() -> str:
    return """
function(cluster) {
  var sum = cluster.getChildCount();
  return new L.DivIcon({
    html: '<div><span>' + sum.toString() + '</span></div>',
    className: 'marker-cluster marker-cluster-small',
    iconSize: new L.Point(40, 40)
  });
}
"""

# ============================================================
# 3) Data
# ============================================================
def ensure_dirs():
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    LATEST_SNAPSHOT_CSV.parent.mkdir(parents=True, exist_ok=True)

def fetch_ocm_au(api_key: str | None) -> list[dict]:
    params = {
        "output": "json",
        "countrycode": "AU",
        "maxresults": str(MAXRESULTS),
        "include": "connections,operatorinfo,usagetype,statustype"
    }
    headers = {"X-API-Key": api_key} if api_key else {}
    print(">> Fetching live data from Open Charge Map...")
    r = requests.get(OCM_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT)
    print(">> HTTP", r.status_code)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected OCM response type")
    print(f">> Received {len(data)} items")
    return data

def normalise_ocm(pois: list[dict]) -> pd.DataFrame:
    rows = []
    for p in pois:
        addr = p.get("AddressInfo") or {}
        conns = p.get("Connections") or []
        op = p.get("OperatorInfo") or {}
        usage = p.get("UsageType") or {}
        status = p.get("StatusType") or {}

        max_power = None
        total_q = 0
        conn_titles = set()
        for c in conns:
            if not c: 
                continue
            try:
                pw = c.get("PowerKW", c.get("ConnectionPowerKW"))
                if pw is not None:
                    pwf = float(pw); max_power = pwf if (max_power is None or pwf > max_power) else max_power
            except Exception:
                pass
            q = c.get("Quantity", 1)
            try:
                total_q += int(q) if q is not None else 1
            except Exception:
                total_q += 1
            ct = c.get("ConnectionType") or {}
            ct_title = ct.get("Title") or ""
            if ct_title:
                conn_titles.add(ct_title)

        rows.append({
            "id": p.get("ID"),
            "title": addr.get("Title"),
            "town": addr.get("Town"),
            "state": addr.get("StateOrProvince"),
            "usage_type": usage.get("Title"),
            "status": status.get("Title"),
            "operator": op.get("Title"),
            "connection_types": ", ".join(sorted(conn_titles)) if conn_titles else "",
            "power_kw": max_power,
            "quantity": total_q if total_q > 0 else None,
            "lat": addr.get("Latitude"),
            "lon": addr.get("Longitude"),
        })
    df = pd.DataFrame.from_records(rows)
    for c in ["lat","lon","power_kw","quantity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()
    return df

# State normalisation for per-state counts
STATE_MAP = {
    "new south wales": "NSW", "nsw": "NSW",
    "victoria": "VIC", "vic": "VIC",
    "queensland": "QLD", "qld": "QLD",
    "south australia": "SA", "sa": "SA",
    "western australia": "WA", "wa": "WA",
    "tasmania": "TAS", "tas": "TAS",
    "northern territory": "NT", "nt": "NT",
    "australian capital territory": "ACT", "act": "ACT",
}
ORDER_STATES = ["NSW","VIC","QLD","WA","SA","TAS","ACT","NT"]

def normalise_state(s: str | None) -> str:
    if not s: return "UNK"
    key = str(s).strip().lower()
    return STATE_MAP.get(key, s.upper() if len(s) <= 4 else "UNK")

def classify_usage_simple(usage: str | None) -> str:
    if not usage: return "unknown"
    u = str(usage).lower()
    if "public" in u: return "public"
    if "private" in u or "restricted" in u: return "private"
    return "unknown"

def classify_status_simple(status: str | None) -> str:
    if not status: return "unknown"
    s = str(status).lower()
    if "operational" in s: return "operational"
    if "temporarily" in s or "partial" in s or "limited" in s: return "partial"
    if "faulted" in s or "down" in s or "not operational" in s: return "down"
    if "planned" in s: return "unknown"
    return "unknown"

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["usage_simple"] = df["usage_type"].apply(classify_usage_simple)
    df["status_simple"] = df["status"].apply(classify_status_simple)
    df["is_fast"] = df["power_kw"].fillna(0) >= FAST_KW
    df["state_abbrev"] = df["state"].apply(normalise_state)
    return df

# ============================================================
# 4) Map helpers
# ============================================================
def thousands(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return ""
    try: return f"{int(round(float(v))):,}"
    except Exception:
        try: return f"{float(v):,.0f}"
        except Exception: return str(v)

def popup_html(row, last_refresh_str: str) -> str:
    title = row.get("title","") or "Unknown"
    town = row.get("town","") or ""
    state = row.get("state","") or ""
    operator = row.get("operator","") or "Unknown"
    usage = row.get("usage_type","") or "Unknown"
    status = row.get("status","") or "Unknown"
    conn = row.get("connection_types","") or ""
    power_kw = row.get("power_kw", np.nan)
    qty = row.get("quantity", np.nan)
    pwr_txt = f"{power_kw:.0f} kW" if pd.notna(power_kw) else "n/a"
    qty_txt = thousands(qty) if pd.notna(qty) else "n/a"
    return (
        f'<div style="font-family:{FONT_FAMILY}; font-size:12px;">'
        f'<div style="font-weight:700; margin-bottom:4px;">{title}</div>'
        f'<div>{town}, {state}</div>'
        f'<div>Operator: <b>{operator}</b></div>'
        f'<div>Usage: <b>{usage}</b></div>'
        f'<div>Status: <b>{status}</b></div>'
        f'{f"<div>Connector(s): <b>{conn}</b></div>" if conn else ""}'
        f'<div>Power: <b>{pwr_txt}</b> · Ports: <b>{qty_txt}</b></div>'
        f'<div style="margin-top:6px; color:#374151; font-size:11px;">Source: <b>Open Charge Map</b></div>'
        f'<div style="margin-top:0px; color:#374151; font-size:10px;">Last data pull: <b>{last_refresh_str}</b></div>'
        '</div>'
    )

def status_color(s_simple: str) -> str:
    return COL_STATUS.get(s_simple or "unknown", COL_STATUS["unknown"])

def add_point_marker(lat, lon, color_hex, popup_html_str=None, tooltip=None) -> CircleMarker:
    cm = CircleMarker(location=(float(lat), float(lon)), radius=5.0, color=color_hex, weight=1.8,
                      fill=True, fill_color=color_hex, fill_opacity=0.75, opacity=1.0)
    if tooltip: folium.Tooltip(tooltip, permanent=False, direction="top", class_name="label-tooltip").add_to(cm)
    if popup_html_str: folium.Popup(popup_html_str, max_width=420).add_to(cm)
    return cm

def color_dot_hex(hex_color: str) -> str:
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{hex_color};margin-right:6px;vertical-align:-1px;"></span>'

# ============================================================
# 5) Build map
# ============================================================
def build_map(df: pd.DataFrame, last_refresh: str, next_refresh: str):
    status_counts = df["status_simple"].value_counts(dropna=False).to_dict()
    tot_sites = len(df)
    n_oper = int(status_counts.get("operational", 0))
    n_partial = int(status_counts.get("partial", 0))
    n_down = int(status_counts.get("down", 0))
    n_unknown = int(status_counts.get("unknown", 0))

    def pct(n): return f"{(100.0 * n / tot_sites):.0f}%" if tot_sites > 0 else "0%"

    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance between two points (km)."""
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    # Per-state counts
    state_counts = df["state_abbrev"].value_counts().to_dict()

    # Normalise duplicates and unknowns
    merged_counts = {}
    for k, v in state_counts.items():
        key = k.strip().upper()
        # If it looks like WA‎, wa, or has stray unicode, merge under WA
        if key.startswith("WA"):
            key = "WA"
        elif key not in ORDER_STATES:
            key = "UNK"
        merged_counts[key] = merged_counts.get(key, 0) + v

    # Build ordered string
    parts = []
    for abbr in ORDER_STATES:
        if abbr in merged_counts:
            parts.append(f"{abbr} <b>{thousands(merged_counts[abbr])}</b>")
    # Add 'UNK' if still present
    if "UNK" in merged_counts:
        parts.append(f"Unkown <b>{thousands(merged_counts['UNK'])}</b>")

    by_state_line = " · ".join(parts) if parts else "By state: n/a"


    df_public = df[df["usage_simple"] == "public"].copy()
    df_private = df[df["usage_simple"] == "private"].copy()
    df_fast = df[df["is_fast"]].copy()

    fig = Figure(width="100%", height="100%")
    m = Map(location=(MAP_START["lat"], MAP_START["lon"]), zoom_start=MAP_START["zoom"],
            tiles=None, control_scale=True, max_bounds=False)
    fig.add_child(m)

    TileLayer(tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
              attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
              name="Carto Light", control=True, show=True, no_wrap=False).add_to(m)
    TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False, no_wrap=False).add_to(m)

    inject_label_css(m)

    # Fit to Australia bounds to open with national coverage visible
#   try:
#       m.fit_bounds(AUS_BOUNDS)
#   except Exception:
#       pass

    cluster_all = MarkerCluster(name="Charger Clusters", show=True, icon_create_function=sum_icon_create_function_js())
    m.add_child(cluster_all)
    grp_all_points = FeatureGroup(name="All Chargers (points)", show=False); m.add_child(grp_all_points)
    grp_public = FeatureGroup(name="Public Only (points)", show=False); m.add_child(grp_public)
    grp_private = FeatureGroup(name="Private/Restricted (points)", show=False); m.add_child(grp_private)
    cluster_fast = MarkerCluster(name=f"Fast Chargers ≥ {int(FAST_KW)} kW (clusters)", show=False,
                                 icon_create_function=sum_icon_create_function_js()); m.add_child(cluster_fast)

    if not df.empty:
        heat_pts = df[["lat","lon"]].dropna().values.tolist()
        if heat_pts:
            HeatMap(heat_pts, radius=18, blur=22, max_zoom=9, min_opacity=0.25,
                    name="Heatmap (all chargers)", show=False).add_to(m)

    for _, r in df.iterrows():
        col = status_color(r.get("status_simple"))
        phtml = popup_html(r, last_refresh)
        add_point_marker(r["lat"], r["lon"], col, popup_html_str=phtml).add_to(cluster_all)
        add_point_marker(r["lat"], r["lon"], col, popup_html_str=phtml).add_to(grp_all_points)

    for _, r in df_public.iterrows():
        add_point_marker(r["lat"], r["lon"], COL_PUBLIC, popup_html_str=popup_html(r, last_refresh)).add_to(grp_public)
    for _, r in df_private.iterrows():
        add_point_marker(r["lat"], r["lon"], COL_PRIVATE, popup_html_str=popup_html(r, last_refresh)).add_to(grp_private)
    for _, r in df_fast.iterrows():
        add_point_marker(r["lat"], r["lon"], COL_FAST, popup_html_str=popup_html(r, last_refresh)).add_to(cluster_fast)

    # --- Urban Centres Layer ---
    import pandas as pd

    try:
        df_uc = pd.read_csv("working/urban_centres_full_clean.csv")
        df_uc = df_uc.rename(columns={"UCL_NAME_2021": "name"})

        # Quick extract of state from UCL_CODE_2021 prefix
        # (first two digits in the code map to ABS state codes)
        state_map = {
            "10": "New South Wales",
            "20": "Victoria",
            "30": "Queensland",
            "40": "South Australia",
            "50": "Western Australia",
            "60": "Tasmania",
            "70": "Northern Territory",
            "80": "Australian Capital Territory"
        }

        df_uc["state"] = df_uc["UCL_CODE_2021"].astype(str).str[:2].map(state_map)

        df_uc["state"] = df_uc["state"].fillna("Unspecified")

        # Remove "Remainder" records
        df_uc = df_uc[~df_uc["name"].str.contains("Remainder", case=False, na=False)]
        
        df_uc = df_uc[df_uc["population"] >= 500]  # keep reasonable centres only

        # Compute chargers within 5, 10, and 20 km of each centre (fast haversine method)

        import numpy as np

        def count_nearby_chargers(lat, lon, dist_km):
            """Return number of chargers within dist_km of given lat/lon."""
            lat1, lon1 = float(lat), float(lon)
            dlat = np.radians(df["lat"] - lat1)
            dlon = np.radians(df["lon"] - lon1)
            a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(df["lat"])) * np.sin(dlon / 2)**2
            distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return int((distances <= dist_km).sum())

        for dist in [5, 10, 20]:
            df_uc[f"chargers_{dist}km"] = df_uc.apply(lambda r: count_nearby_chargers(r["lat"], r["lon"], dist), axis=1)

        print(">> Computed nearby chargers for 5, 10, and 20 km radii")

        urban_layer = folium.FeatureGroup(
            name="Urban Centres and Localities (UCL)", overlay=True, control=True, show=False
        )
        
        def pop_style(pop):
            """Return (radius, color_hex, fill_opacity) based on population."""
            if pop < 5_000:
                return 2,  "#9ecae1", 0.75    # small towns – mid blue
            elif pop < 25_000:
                return 3,  "#6baed6", 0.78
            elif pop < 100_000:
                return 5,  "#4292c6", 0.80
            elif pop < 250_000:
                return 7,  "#2171b5", 0.83
            elif pop < 500_000:
                return 9,  "#08519c", 0.85
            elif pop < 750_000:
                return 10, "#08306b", 0.88
            elif pop < 1_000_000:
                return 11, "#4b0082", 0.90
            elif pop < 1_250_000:
                return 12, "#3a0068", 0.92
            elif pop < 2_000_000:
                return 13, "#2b004f", 0.94
            elif pop < 3_000_000:
                return 14, "#20003d", 0.95
            else:
                return 16, "#150029", 0.96    # Sydney / Melbourne
        
        for _, r in df_uc.iterrows():
#           radius, color = pop_style(r.population)
            radius, color, fop = pop_style(r.population)

            # Build custom popup HTML
            html = f"""
            <div style="width:250px; font-family:Arial, sans-serif;">
                <div style="font-size:12px; font-weight:bold; color:{color};">{r['name']}</div>
                <div style="font-size:12px;"><b>{r.state}</b></div>
                <div style="font-size:10px;">Population: {int(r.population):,}</div>
                <div style="font-size:10px;">
                Chargers within 20 km: {int(r['chargers_20km'])}
                </div>
                <div style="font-size:8px; color:#666;">Data sources: ABS (UCL 2021), OCM API</div>
            </div>
            """

            popup = folium.Popup(html, max_width=300)

            folium.CircleMarker(
                location=[r.lat, r.lon],
                radius=radius / 1.5,       # keep circles modest in size
                color=color,
                weight=0.8,
                fill=True,
                fillColor=color,
#               fillOpacity=0.75 if r.population < 100_000 else 0.85,
                fillOpacity=fop,
                popup=popup,
            ).add_to(urban_layer)

        urban_layer.add_to(m)
        print(f">> Added Urban Centres layer ({len(df_uc)} points)")

        # Save enriched dataset for external analysis (equity summary, etc.)
        df_uc.to_csv("working/urban_centres_with_charger_access.csv", index=False)
        print("✅ Saved working/urban_centres_with_charger_access.csv for equity analysis")

    except Exception as e:
        print("!! Could not load Urban Centres:", e)

    LayerControl(collapsed=False).add_to(m)

    # Title box
    title_html = (
        f'<div style="color:#111;font-size:20px; font-weight:700; margin-bottom:4px;">{TITLE_PLACEHOLDER}</div>'
        f'<div style="color:#111;font-size:14px; font-weight:500; margin-bottom:2px;">{SUBTITLE_PLACEHOLDER}</div>'
        f'<div style="color:#111;font-size:12px; font-weight:400;">{THIRD_TITLE_PLACEHOLDER}</div>'
#       f'<div style="color:#111;font-size:12px; font-weight:400;">AU-wide live snapshot</div>'
    )
    m.add_child(build_transparent_box("box-title", title_html, position="topleft", offsets=(50,10)))

    # --- Snapshot box with colored dots and per-state line ---
    dot_g = color_dot_hex(COL_STATUS["operational"])
    dot_o = color_dot_hex(COL_STATUS["partial"])
    dot_r = color_dot_hex(COL_STATUS["down"])
    dot_u = color_dot_hex(COL_STATUS["unknown"])

    # 1) Population coverage shares (5 / 10 / 20 km)
    shares = {5: 0, 10: 0, 20: 0}
    try:
        pop_total = df_uc["population"].sum()
        for dist_km in [5, 10, 20]:
            covered = df_uc.loc[df_uc[f"chargers_{dist_km}km"] > 0, "population"].sum()
            shares[dist_km] = 100 * covered / max(pop_total, 1)
    except Exception as e:
        print("!! Could not compute coverage summary:", e)

    # 2) Urban centres coverage (% of UCLs with ≥1 charger within radius)
    centre_shares = {5: 0, 10: 0, 20: 0}
    try:
        total_centres = max(len(df_uc), 1)
        for dist_km in [5, 10, 20]:
            covered_centres = (df_uc[f"chargers_{dist_km}km"] > 0).sum()
            centre_shares[dist_km] = 100 * covered_centres / total_centres
    except Exception as e:
        print("!! Could not compute urban-centre coverage:", e)

    # 3) Build bullets (include both lines BEFORE the timestamps)
    
    bullets = [
        'Data source: <b><a href="https://openchargemap.org/" target="_blank">Open Charge Map API</a></b>',
        f"Total sites: <b>{thousands(tot_sites)}</b>",
        f"{by_state_line}<br>",  # add <br> to end the state line cleanly
        f"{dot_g}&nbsp;Operational: <b>{thousands(n_oper)}</b> ({pct(n_oper)})",
        f"{dot_o}&nbsp;Partial: <b>{thousands(n_partial)}</b> ({pct(n_partial)})",
        f"{dot_r}&nbsp;Down: <b>{thousands(n_down)}</b> ({pct(n_down)})",
        f"{dot_u}&nbsp;Unknown status: <b>{thousands(n_unknown)}</b> ({pct(n_unknown)})",
        (
            f"Population access to chargers: "
            f"Within <b>5 km ({shares[5]:.1f}%)</b> · "
            f"<b>10 km ({shares[10]:.1f}%)</b> · "
            f"<b>20 km ({shares[20]:.1f}%)</b>"
        ),
        (
            f"Urban centres (UCL) with a charger: "
            f"Within <b>5 km ({centre_shares[5]:.1f}%)</b> · "
            f"<b>10 km ({centre_shares[10]:.1f}%)</b> · "
            f"<b>20 km ({centre_shares[20]:.1f}%)</b>"
        ),
        f"Last data pull: <b>{last_refresh}</b>",
        f"Next data pull: <b>{next_refresh}</b>",
    ]



    snapshot_html = (
        '<div style="color:#111; font-weight:600; font-size:12px; margin-bottom:6px;">'
        'Australian EV charging snapshot</div>'
        '<ul style="margin:3px 0 0 0; padding-left: 18px;">'
        '<li>' + "</li><li>".join(bullets) + "</li></ul>"
    )

    m.add_child(build_transparent_box("box-snapshot", snapshot_html, position="bottomleft", offsets=(10,48), width_px=540))

    # How-to box

    howto_bullets = [
    "Snapshot data obtained from the Open Charge Map (OCM) API at the time shown.",
    "Reporting to OCM is voluntary - some operators may not list all of their chargers or update them regularly.",
    'Search & routing use <a href="https://nominatim.openstreetmap.org/" target="_blank">Nominatim</a> & <a href="https://project-osrm.org/" target="_blank">OSRM</a> - lightweight open-data tools that may not match commercial map precision.',
    "Routes and charging station proximity around each route are approximate.",
    "Cluster badge shows the sum of counts inside each cluster at this zoom.",
    "Dots at highest zoom show site-level charging stations.",
    "Popups show values at snapshot time.",
    HOWTO_FAST_LINE,
    "Charging station availability reflects current data in OCM API at the time of retrieval.",
    "Charger status and uptime can change – always confirm current availability in your network’s app or live sources.",
    'The Urban Centres Layer represents ABS-defined <a href="https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/significant-urban-areas-urban-centres-and-localities-section-state/urban-centres-and-localities" target="_blank">Urban Centre and Localities (UCL)</a> with population ≥ 500 residents.'
]

    howto_html = (
        '<div style="color:#111; font-weight:600; font-size:12px; margin-bottom:6px;">How to read this map</div>'
        '<ul style="margin:3px 0 0 0; padding-left: 18px;">'
        '<li>' + "</li><li>".join(howto_bullets) + "</li></ul>"
    )
    m.add_child(build_transparent_box("box-howto", howto_html, position="bottomright", offsets=(12,48), width_px=675))

    # ---- Route planner UI + JS ----
    points = [{
        "lat": float(r["lat"]), "lon": float(r["lon"]),
        "status": str(r.get("status_simple") or "unknown"),
        "usage": str(r.get("usage_simple") or "unknown"),
        "fast": bool(r.get("is_fast")),
        "title": str(r.get("title") or ""),
        "operator": str(r.get("operator") or ""),
        "town": str(r.get("town") or ""),
        "state": str(r.get("state") or ""),
        "power_kw": float(r["power_kw"]) if pd.notna(r.get("power_kw")) else None
    } for _, r in df.iterrows()]
    js_points = json.dumps(points)

    panel_html_only = f"""
    <div id="route-search" style="position: fixed; z-index:100001; top: {ROUTE_PANEL_TOP_PX}px; right: {ROUTE_PANEL_RIGHT_PX}px;
      background:RGBA(255,255,255,0.78); padding:10px 12px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.2);
      font-family:Inter, Arial, sans-serif; font-size:12px; color:#111; width: 450px;">
      <div style="font-size:14px; font-weight:600; margin-bottom:6px;">Route planner</div>
               <div style="font-size:11.5px; color:#333; margin-top:6px;"> Use this box to plot a route and highlight chargers within {ROUTE_PROXIMITY_KM:.1f} km.  </div> <br>
      <label>Origin</label>
      <input id="origin-input" type="text" placeholder="Melbourne" list="origin-list" style="width:100%; margin-bottom:6px;" />
      <datalist id="origin-list"></datalist>
      <label>Destination</label>
      <input id="dest-input" type="text" placeholder="Perth" list="dest-list" style="width:100%; margin-bottom:6px;" />
      <datalist id="dest-list"></datalist>
      <div style="display:flex; gap:8px; margin-top:6px;">
        <button id="btn-find" style="flex:1; padding:6px 8px;">Find Route</button>
        <button id="btn-clear" style="padding:6px 8px;">Clear</button>
      </div>
      <div id="route-msg" style="margin-top:6px; color:#444;"></div>
    </div>
    """.strip()

    panel_html_js_literal = panel_html_only.replace('`', '\\`').replace('\\n',' ')

    script_html = f"""
    <script>
    (function() {{
      const EV_POINTS = {js_points};
      const PROX_KM = {ROUTE_PROXIMITY_KM:.1f};
      const PANEL_AUTO = {str(ROUTE_PANEL_AUTO).lower()};

      var wrap = document.createElement('div');
      wrap.innerHTML = `{panel_html_js_literal}`;
      var panel = wrap.firstElementChild;
      document.body.appendChild(panel);

      function findLeafletMap() {{
        for (var k in window) {{
          try {{
            var v = window[k];
            if (k.startsWith('map_') && v && typeof v.fitBounds === 'function' && v.addLayer) return v;
          }} catch(e) {{ }}
        }}
        if (window.map && typeof window.map.fitBounds === 'function') return window.map;
        return null;
      }}

      function placePanel() {{
        if (!PANEL_AUTO) return true;
        const lc = document.querySelector('.leaflet-control-layers');
        if (!lc) return false;
        const rect = lc.getBoundingClientRect();
        panel.style.top = (rect.bottom + 12) + 'px';
        panel.style.right = (window.innerWidth - rect.right + 12) + 'px';
        return true;
      }}
      (function retryPlace(n) {{ if (placePanel()) return; if (n>0) setTimeout(()=>retryPlace(n-1),250); }})(12);
      window.addEventListener('resize', placePanel);

      function whenMapReady(cb) {{
        let tries = 0;
        const iv = setInterval(() => {{
          const m = findLeafletMap();
          if (m) {{ clearInterval(iv); cb(m); return; }}
          if (++tries > 60) {{ clearInterval(iv); const msg=document.getElementById('route-msg'); if (msg) msg.textContent='Map not ready. Please reload the page.'; }}
        }}, 250);
      }}

      function haversineKm(lat1, lon1, lat2, lon2) {{
        const R = 6371.0;
        const dLat = (lat2-lat1) * Math.PI/180.0;
        const dLon = (lon2-lon1) * Math.PI/180.0;
        const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
        const c = 2*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R*c;
      }}
      function minDistKm(lat, lon, coords) {{
        let best = Infinity;
        for (let i=0; i<coords.length; i++) {{
          const p = coords[i];
          const d = haversineKm(lat, lon, p[1], p[0]);
          if (d < best) best = d;
        }}
        return best;
      }}

      function debounce(fn, ms) {{ let t; return function(...args) {{ clearTimeout(t); t = setTimeout(() => fn.apply(this,args), ms); }}; }}
      const geoCache = new Map();
      async function nominatimSuggest(q, which) {{
        const key = which + '|' + q.trim();
        if (!q || q.length < 3) return [];
        if (geoCache.has(key)) return geoCache.get(key);
        const url = 'https://nominatim.openstreetmap.org/search?format=jsonv2&countrycodes=au&limit=5&q=' + encodeURIComponent(q);
        try {{
          const resp = await fetch(url, {{ headers: {{ 'Accept': 'application/json' }} }});
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          const data = await resp.json();

const items = data.map(d => {{
    const parts = d.display_name.split(',').map(p => p.trim());
    const filtered = parts.filter(p => !/Australia/i.test(p) && !/^\\d{{4}}$/.test(p));
    const label = filtered.slice(0, 3).join(', ');
    return {{
        label: label,
        lat: parseFloat(d.lat),
        lon: parseFloat(d.lon)
    }};
}});

// Deduplicate by label (keep first occurrence)
const unique = [];
const seen = new Set();
for (const it of items) {{
    if (!seen.has(it.label)) {{
        seen.add(it.label);
        unique.push(it);
    }}
}}

geoCache.set(key, unique);
return unique;


        }} catch(e) {{ console.warn('Nominatim error', e); return []; }}
      }}
      function populateDatalist(id, items) {{
        const dl = document.getElementById(id);
        if (!dl) return;
        dl.innerHTML = '';
        items.forEach(it => {{ const opt=document.createElement('option'); opt.value=it.label; dl.appendChild(opt); }});
      }}
      const originEl = document.getElementById('origin-input');
      const destEl = document.getElementById('dest-input');
      originEl.addEventListener('input', debounce(async (ev)=>{{ populateDatalist('origin-list', await nominatimSuggest(ev.target.value,'o')); }},250));
      destEl.addEventListener('input', debounce(async (ev)=>{{ populateDatalist('dest-list', await nominatimSuggest(ev.target.value,'d')); }},250));

      function geocodeFirst(label, which) {{
        const key = which + '|' + label.trim();
        if (geoCache.has(key)) return Promise.resolve(geoCache.get(key)[0]);
        return nominatimSuggest(label, which).then(arr => arr[0] || null);
      }}

      whenMapReady(function(mapRef) {{
        var routeLayer = L.layerGroup().addTo(mapRef);
        var nearLayer = L.layerGroup().addTo(mapRef);
        var nearHeat = null;

        async function doRoute() {{
          const msg = document.getElementById('route-msg');
          if (msg) msg.textContent = 'Finding route...';
          const originTxt = originEl.value.trim();
          const destTxt = destEl.value.trim();
          if (!originTxt || !destTxt) {{ if (msg) msg.textContent = 'Please enter both origin and destination.'; return; }}

          const o = await geocodeFirst(originTxt, 'o');
          const d = await geocodeFirst(destTxt, 'd');
          if (!o || !d) {{ if (msg) msg.textContent = 'Could not find one or both places. Try being more specific.'; return; }}
      
          
            // --- Build OSRM query ---
            const osrmBase = "https://router.project-osrm.org";

            const coordPair = `${{o.lon}},${{o.lat}};${{d.lon}},${{d.lat}}`;

            // Prefer car/driving profile; retry if route not found
            let url = `${{osrmBase}}/route/v1/driving/${{coordPair}}?overview=full&geometries=geojson`;


            let geo = null;
            let totalKm = 0;
            let data = null;

            try {{
                let res = await fetch(url);
                data = await res.json();

                // Fallback: if OSRM returns no route, try car profile
                if (!data.routes || data.routes.length === 0) {{
                    console.warn("Primary OSRM route not found, retrying with car profile...");
                    url = `${{osrmBase}}/route/v1/car/${{coordPair}}?overview=full&geometries=geojson`;
                    res = await fetch(url);
                    data = await res.json();
                }}

                if (!data.routes || data.routes.length === 0) {{
                    if (msg) msg.textContent = "Route not found — try a nearby landmark or city centre.";
                    return;
                }}

                const route = data.routes[0];
                geo = route.geometry;
                totalKm = route.distance ? (route.distance / 1000.0) : 0;

            }} catch(e) {{
                console.warn("OSRM error", e);
                if (msg) msg.textContent = "Route unavailable. Try a nearby suburb or major city.";
                return;
            }}

          routeLayer.clearLayers();
          nearLayer.clearLayers();
          if (nearHeat) {{ try {{ nearHeat.remove(); }} catch(e){{}} nearHeat = null; }}

          const line = L.geoJSON(geo, {{ style: {{ color: '#2563eb', weight: {ROUTE_LINE_WEIGHT}, opacity: 0.85 }} }});
          line.addTo(routeLayer);
          mapRef.fitBounds(line.getBounds(), {{ padding: [24,24] }});

          const coords = geo.coordinates;
          const nearPts = [];
          EV_POINTS.forEach(pt => {{
            const dkm = minDistKm(pt.lat, pt.lon, coords);
            if (dkm <= PROX_KM) nearPts.push([pt.lat, pt.lon]);
            if (dkm <= PROX_KM) {{
              const c = pt.fast ? '#2563eb' : '#22c55e';
              const cm = L.circleMarker([pt.lat, pt.lon], {{ radius: 4, color: c, weight: 1.2, fill: true, fillColor: c, fillOpacity: 0.75 }});
              cm.addTo(nearLayer);
            }}
          }});

          if (nearPts.length && L.heatLayer) {{
            nearHeat = L.heatLayer(nearPts, {{ radius: 18, blur: 22, maxZoom: 9, minOpacity: 0.25 }}).addTo(mapRef);
          }}

          // --- Detect longest gap between nearby chargers along route (enhanced) ---
          let chargerCoords = [];
          for (let i = 0; i < coords.length; i++) {{
            const [lon, lat] = coords[i];
            const d = minDistKm(lat, lon, nearPts.map(p => [p[1], p[0]]));
            if (d <= 5) chargerCoords.push([lat, lon]);
          }}

          let maxGap = 0;
          if (chargerCoords.length >= 2) {{
            for (let i = 1; i < chargerCoords.length; i++) {{
              const [lat1, lon1] = chargerCoords[i - 1];
              const [lat2, lon2] = chargerCoords[i];
              const segKm = haversineKm(lat1, lon1, lat2, lon2);
              maxGap = Math.max(maxGap, segKm);
            }}
          }} else if (chargerCoords.length <= 1) {{
            // If zero or one charger found, treat entire route as unserved
            maxGap = totalKm;
          }}

// --- Highlight all long gaps (not just the longest one) ---
if (maxGap > 300 && chargerCoords.length >= 2) {{
    for (let i = 1; i < chargerCoords.length; i++) {{
        const [lat1, lon1] = chargerCoords[i - 1];
        const [lat2, lon2] = chargerCoords[i];
        const segKm = haversineKm(lat1, lon1, lat2, lon2);

        if (segKm > 300) {{
            // find this gap’s route indices
            let startIdx = 0;
            let endIdx = coords.length - 1;
            let bestStart = Infinity;
            let bestEnd = Infinity;

            for (let j = 0; j < coords.length; j++) {{
                const [lon, lat] = coords[j];
                const d1 = haversineKm(lat, lon, lat1, lon1);
                const d2 = haversineKm(lat, lon, lat2, lon2);
                if (d1 < bestStart) {{ bestStart = d1; startIdx = j; }}
                if (d2 < bestEnd)  {{ bestEnd  = d2; endIdx   = j; }}
            }}

            if (startIdx > endIdx) {{
                const tmp = startIdx; startIdx = endIdx; endIdx = tmp;
            }}

            const gapCoords = coords.slice(startIdx, endIdx + 1).map(c => [c[1], c[0]]);
            if (gapCoords.length > 1) {{
                const gapLine = L.polyline(gapCoords, {{
                    color: 'red',
                    weight: 2.5,
                    opacity: 1.0
                }});
                gapLine.addTo(routeLayer);
            }}
        }}
    }}
}}

// --- Detect possible ferry crossings or over-water gaps - option 2---
let ferryNotice = "";
const minLat = Math.min(o.lat, d.lat);
if (maxGap > 200 && totalKm > 400 && minLat < -38) {{
    ferryNotice = "<br>⛴️ Note: route may include a ferry or non-road crossing (e.g., Bass Strait).";
}}


// --- Display route info and any warnings ---
if (msg) {{
    msg.innerHTML = `Shortest-distance route between origin and destination found: <b>${{totalKm.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} km</b>.<br>Chargers within ${{PROX_KM.toFixed(1)}} km along the route are highlighted.`;
    if (maxGap > 300) {{
        msg.innerHTML += `<br><span style='color:red;'>⚠️ Route includes section(s) with limited chargers within ${{PROX_KM.toFixed(1)}} km.<br>⚠️ Route includes a stretch up to <b>${{maxGap.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} km</b> without coverage.</span>`;
    }}
    if (ferryNotice) {{
        msg.innerHTML += ferryNotice;
    }}
}}

}}
document.getElementById('btn-find').addEventListener('click', doRoute);
document.getElementById('btn-clear').addEventListener('click', function() {{
  routeLayer.clearLayers();
  nearLayer.clearLayers();
  if (nearHeat) {{ try {{ nearHeat.remove(); }} catch(e){{}} nearHeat = null; }}
  const msg = document.getElementById('route-msg');
  if (msg) msg.textContent = '';
}});
}}); // whenMapReady
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(script_html))

    m.save(str(OUTPUT_HTML))
    print(f">> Map saved to {OUTPUT_HTML.resolve()}")
       
# ============================================================
# 6) Main
# ============================================================
def main():
    print(">> Australian EV Charging Atlas (v6)")
    ensure_dirs()
    load_dotenv()
    api_key = os.getenv("OCM_API_KEY", "").strip()
    if api_key: print(">> Using OCM_API_KEY (loaded from .env)")
    else: print("!! No OCM_API_KEY found. Proceeding without header.")

    try:
        data = fetch_ocm_au(api_key)
        df = normalise_ocm(data)
        df = enrich_dataframe(df)
        df.to_csv(LATEST_SNAPSHOT_CSV, index=False)
        print(f">> Wrote latest snapshot to {LATEST_SNAPSHOT_CSV}")
    except Exception as e:
        print("!! Live fetch failed:", e)
        try:
            df = pd.read_csv(BACKUP_CSV)
            for c in ["power_kw","quantity","usage_type","status","operator","connection_types"]:
                if c not in df.columns: df[c] = np.nan
            for c in ["lat","lon","power_kw","quantity"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["lat","lon"]).copy()
            df = enrich_dataframe(df)
            print(">> Using backup CSV as data source.")
        except Exception as e2:
            print("!! Backup CSV also unavailable:", e2)
            return

    # timestamps
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Australia/Melbourne")
    except Exception:
        tz = timezone(timedelta(hours=10))
    now_local = datetime.now(tz)
    last_refresh = now_local.strftime("%d %b %Y %H:%M %Z")
    next_refresh = (now_local + timedelta(hours=24)).strftime("%d %b %Y %H:%M %Z")

    print(">> Building map...")
    build_map(df, last_refresh, next_refresh)
    # Ensure Netlify root file timestamp updates
    try:
        atlas_file = Path("outputs/index.html")
        os.utime(atlas_file, None)
        print(f">> Updated timestamp for {atlas_file}")
    except Exception as e:
        print("!! Could not update index.html timestamp:", e)

    print(">> Done. Upload outputs/index.html to Netlify.")


"""
    # --- Finalise and save ---
    # Make sure index.html is always written for Netlify
    try:
        import shutil
        OUTPUT_DIR = Path("outputs")
        atlas_file = OUTPUT_DIR / "index.html"
        if not atlas_file.exists():
            print(">> index.html not found, creating it now...")
        else:
            print(">> Replacing existing index.html...")
        shutil.copy(atlas_file, OUTPUT_DIR / "index.html")
        print(f">> Map copied to {OUTPUT_DIR / 'index.html'}")
    except Exception as e:
        print("!! Could not copy map to index.html:", e)
    print(">> Done. Upload outputs/index.html to Netlify.")
"""

if __name__ == "__main__":
    main()
