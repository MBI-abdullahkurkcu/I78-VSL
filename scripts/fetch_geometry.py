"""Road-following geometry for the INRIX TMC segments via the public OSRM router.

Same approach as the Santa Fe corridor map (tools/build_map.py): route from the
TMC start point to its end point with a few snapping settings, keep the
candidate whose routed length is closest to the INRIX segment length, and fall
back to a straight line.  Results are cached per TMC in dashboard/data/osrm_cache/
so the router is only called once per segment.

Output: dashboard/data/tmc_geometry.json
    {"<tmc>": {"coords": [[lat, lon], ...], "routed_mi": 3.52, "straight": false}, ...}

Run:  python -I scripts/fetch_geometry.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "Input Data")
OUT = os.path.join(ROOT, "dashboard", "data", "tmc_geometry.json")
CACHE = os.path.join(ROOT, "dashboard", "data", "osrm_cache")
OSRM = "https://router.project-osrm.org/route/v1/driving/"


def bearing(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a[1], a[0], b[1], b[0]])
    x = math.sin(lo2 - lo1) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def haversine_mi(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a[1], a[0], b[1], b[0]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def route_geometry(tmc, a, b, miles):
    """a, b are (lon, lat). Returns (coords as [lon, lat], straight_flag, routed_mi)."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"{tmc}.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            cands = json.load(f)
    else:
        br = round(bearing(a, b))
        cands = []
        for rad, tol in ((25, 30), (50, 45), (100, 60), (50, None)):
            q = f"{a[0]},{a[1]};{b[0]},{b[1]}?overview=full&geometries=geojson&radiuses={rad};{rad}"
            if tol:
                q += f"&bearings={br},{tol};{br},{tol}"
            try:
                req = urllib.request.Request(OSRM + q, headers={"User-Agent": "I78-VSL-dashboard/1.0"})
                d = json.load(urllib.request.urlopen(req, timeout=30))
                if d.get("routes"):
                    cands.append({"rad": rad, "tol": tol, "mi": d["routes"][0]["distance"] / 1609.34,
                                  "coords": d["routes"][0]["geometry"]["coordinates"]})
            except Exception as e:  # keep going; the straight line is the fallback
                cands.append({"rad": rad, "tol": tol, "err": str(e)[:80]})
            time.sleep(0.3)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cands, f)
    straight = haversine_mi(a, b)
    target = miles if miles else max(straight, 0.01)
    best = None
    for c in cands:
        if "coords" not in c or len(c["coords"]) < 2:
            continue
        ratio = c["mi"] / target
        if 0.6 <= ratio <= 1.6 and (best is None or abs(ratio - 1) < abs(best[0] - 1)):
            best = (ratio, c)
    if best:
        return [[round(x, 6), round(y, 6)] for x, y in best[1]["coords"]], False, round(best[1]["mi"], 3)
    return [[round(a[0], 6), round(a[1], 6)], [round(b[0], 6), round(b[1], 6)]], True, round(straight, 3)


def main():
    base = None
    for pat in ("INRIX*", "Inrix*", "inrix*"):
        hits = [h for h in sorted(glob.glob(os.path.join(INPUT, pat))) if os.path.isdir(h)]
        if hits:
            base = hits[0]
            break
    if not base:
        raise SystemExit("no INRIX folder under Input Data/")
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(base, "*TMC_Identification*.csv")))]
    tmcs = pd.concat(frames).drop_duplicates("tmc", keep="last")
    out = {}
    for r in tmcs.itertuples():
        a = (float(r.start_longitude), float(r.start_latitude))
        b = (float(r.end_longitude), float(r.end_latitude))
        coords, straight, mi = route_geometry(r.tmc, a, b, float(r.miles or 0))
        out[r.tmc] = {"coords": [[y, x] for x, y in coords], "routed_mi": mi, "straight": straight, "inrix_mi": round(float(r.miles), 3)}
        print(f"  {r.tmc} {r.direction[:2]} {r.intersection:30s} inrix {float(r.miles):5.2f} mi  routed {mi:5.2f} mi  pts {len(coords):4d}{'  STRAIGHT' if straight else ''}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT} ({len(out)} segments, {sum(v['straight'] for v in out.values())} straight fallbacks)")


if __name__ == "__main__":
    main()
