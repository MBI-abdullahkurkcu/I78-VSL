"""Fetch I-78 mileposts from NJDOT's Standard Route Id and Milepost feature service.

Source:
  https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/
    New_Jersey_Standard_Route_Id_And_Milepost/FeatureServer/0

Writes dashboard/data/i78_mileposts.json with:
  {
    "source": "...",
    "fetched": "<ISO timestamp>",
    "count": N,
    "mp_min": 0.0, "mp_max": 14.0,
    "points": [{"mp": 0.0, "lat":..., "lon":...}, ...]   // interleaved EB / WB averaged by MP
  }

Runs once; re-run only when the NJDOT service or the corridor limits change.
Needs internet. No API key required.
"""
import json, os, sys, urllib.request, urllib.parse, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "dashboard", "data", "i78_mileposts.json")

# MP window slightly wider than the VSL corridor (2.8 to 13.2) so signs/crashes just outside the limits still snap.
MP_MIN, MP_MAX = 0.0, 14.0
SRIS = ("00000078__", "00000078_W")

def fetch():
    base = "https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/New_Jersey_Standard_Route_Id_And_Milepost/FeatureServer/0/query"
    params = {
        "where": "SRI IN (" + ",".join(f"'{s}'" for s in SRIS) + f") AND MP<={MP_MAX}",
        "outFields": "SRI,MP,Latitude,Longitude",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": "5000",
        "orderByFields": "SRI,MP",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "I-78-VSL-dashboard research (contact: project team)"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.loads(fh.read().decode("utf-8", "replace"))

def main():
    gj = fetch()
    if gj.get("error"):
        sys.exit(f"service returned error: {gj['error']}")
    feats = gj.get("features", [])
    # Group points by rounded MP, averaging EB and WB lat/lon so the corridor centerline
    # is a single smooth line of real mileposts.
    by_mp = {}
    for f in feats:
        p = f.get("properties", {}) or {}
        mp = p.get("MP"); lat = p.get("Latitude"); lon = p.get("Longitude")
        if mp is None or lat is None or lon is None: continue
        if mp < MP_MIN or mp > MP_MAX: continue
        key = round(float(mp) * 20) / 20  # 0.05 mi granularity
        by_mp.setdefault(key, []).append((float(lat), float(lon)))
    points = []
    for mp in sorted(by_mp):
        rows = by_mp[mp]
        lat = sum(r[0] for r in rows) / len(rows)
        lon = sum(r[1] for r in rows) / len(rows)
        points.append({"mp": round(mp, 3), "lat": round(lat, 6), "lon": round(lon, 6)})
    out = {
        "source": "https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/New_Jersey_Standard_Route_Id_And_Milepost/FeatureServer/0",
        "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(points),
        "mp_min": points[0]["mp"] if points else None,
        "mp_max": points[-1]["mp"] if points else None,
        "points": points,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(points)} points, MP {out['mp_min']} to {out['mp_max']} ({kb:.1f} KB)")

if __name__ == "__main__":
    main()
