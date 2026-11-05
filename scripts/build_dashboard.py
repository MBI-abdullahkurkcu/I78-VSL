"""Build the standalone I-78 VSL dashboard.

Reads everything under "Input Data/" (plus the INRIX summary produced by
scripts/preprocess_inrix.py), derives the summary tables the dashboard needs,
and injects one JSON blob into dashboard/template.html, writing the
self-contained page to I78_VSL_Dashboard.html at the repository root.

Run from the repo root:
    python -I scripts/preprocess_inrix.py     # only when new INRIX files arrive (slow)
    python -I scripts/build_dashboard.py      # every time (fast)

Each dataset has a loader that returns a {"status": "missing"} dict when its
folder is absent, so the page renders an "awaiting data" card instead of failing.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "Input Data")
TEMPLATE = os.path.join(ROOT, "dashboard", "template.html")
OUT_HTML = os.path.join(ROOT, "I78_VSL_Dashboard.html")
OUT_JSON = os.path.join(ROOT, "dashboard", "data", "dashboard_data.json")
INRIX_SUMMARY = os.path.join(ROOT, "dashboard", "data", "inrix_summary.json")
CORRIDOR = os.path.join(ROOT, "config", "corridor.json")

# Daily weather classes, ordered by severity. The dashboard colors and the
# scenario exposure counts use these ids.
CLASSES = [
    {"id": 0, "key": "dry",       "label": "Dry / no precipitation"},
    {"id": 1, "key": "wet",       "label": "Wet pavement (rain >= 0.10 in, area mean)"},
    {"id": 2, "key": "heavy",     "label": "Heavy rain (>= 1.0 in at any station)"},
    {"id": 3, "key": "freezing",  "label": "Freezing precipitation / black-ice risk (precip with Tmin <= 32 F, snow < 0.5 in)"},
    {"id": 4, "key": "snow_lt",   "label": "Light snow (0.1 to < 1 in at any station)"},
    {"id": 5, "key": "snow_sig",  "label": "Significant snow (1 to < 3 in)"},
    {"id": 6, "key": "snow_maj",  "label": "Major snow (>= 3 in)"},
]

ADVERSE_SURFACE = {"Wet", "Snowy", "Icy", "Slush", "Water (Standing/Moving)"}
ADVERSE_WEATHER = {"Rain", "Snow", "Fog/Smog/Smoke", "Sleet/Hail", "Freezing Rain", "Severe Crosswinds"}
TRUCK_TYPES = {"Tractor Semi- Trailer", "Single Unit (2 axle)", "Truck 3Axle w/Trailer", "Truck Tractor (Bobtail)",
               "Single Unit (3 axle)", "Truck 2Axle w/Trailer", "Other Truck*", "Tractor Double"}
SEV_MAP = {"Fatal Injury (K)": "K", "Suspected Serious Injury (A)": "A", "Suspected Minor Injury (B)": "B",
           "Possible Injury (C)": "C", "No Apparent Injury (O)": "O"}


def _r(x, n=2):
    """Round and convert NaN to None so JSON stays clean."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), n)


def _vc(series, top=None):
    vc = series.fillna("Unknown").value_counts()
    if top:
        vc = vc.head(top)
    return [{"k": str(k), "n": int(v)} for k, v in vc.items()]


def find_dir(*candidates: str) -> str | None:
    for pat in candidates:
        hits = sorted(glob.glob(os.path.join(INPUT, pat)))
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return hits[0]
    return None


def file_manifest(folder, max_count_mb=250):
    """List the files in a dataset folder with size, row and column counts.

    Row counts are a raw line count (header excluded) so no file content is
    parsed or carried into the output; very large files only report a size.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        rec = {"name": os.path.relpath(path, folder).replace(os.sep, "/"), "size_mb": round(size / 1048576, 2), "rows": None, "cols": None}
        if path.lower().endswith(".csv"):
            with open(path, "rb") as f:
                header = f.readline()
                rec["cols"] = header.count(b",") + 1 if header.strip() else None
                if size <= max_count_mb * 1048576:
                    n = 0
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        n += chunk.count(b"\n")
                    rec["rows"] = n
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Climate (NOAA GHCN-Daily pull)
# --------------------------------------------------------------------------
def load_climate():
    base = find_dir("Climate*", "climate*")
    if not base:
        return {"status": "missing", "expected_folder": "Input Data/Climate*/<area>/daily_composite.csv"}
    area_dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)
                 and os.path.exists(os.path.join(d, "daily_composite.csv"))]
    if not area_dirs:
        return {"status": "missing", "expected_folder": "Input Data/Climate*/<area>/daily_composite.csv"}
    area = area_dirs[0]
    run_info = {}
    ri = os.path.join(base, "run_info.json")
    if os.path.exists(ri):
        with open(ri, encoding="utf-8") as f:
            run_info = json.load(f)

    d = pd.read_csv(os.path.join(area, "daily_composite.csv"), parse_dates=["date"])
    st = pd.read_csv(os.path.join(area, "stations.csv"))
    per = pd.read_csv(os.path.join(area, "period_by_station.csv"))
    d = d.sort_values("date").reset_index(drop=True)

    snow_max = d["SNOW_max"].fillna(0)
    prcp_max = d["PRCP_max"].fillna(0)
    prcp_mean = d["PRCP_mean"].fillna(0)
    tmin = d["TMIN_min"]
    cls = np.zeros(len(d), dtype=int)
    cls[(prcp_mean >= 0.10)] = 1
    cls[(prcp_max >= 1.0)] = 2
    cls[(prcp_max >= 0.01) & (tmin <= 32) & (snow_max < 0.5)] = 3
    cls[(snow_max >= 0.1)] = 4
    cls[(snow_max >= 1.0)] = 5
    cls[(snow_max >= 3.0)] = 6
    d["cls"] = cls
    d["freeze"] = (tmin <= 32).astype(int)
    d["freeze_thaw"] = ((tmin <= 32) & (d["TMAX_max"] > 32)).astype(int)
    d["snow_cover"] = (d["SNWD_max"].fillna(0) >= 1).astype(int)
    d["any_adverse"] = (cls > 0).astype(int)

    annual = []
    for yr, g in d.groupby("year"):
        row = {"year": int(yr), "days": int(len(g)),
               "prcp_total": _r(g["PRCP_mean"].sum(), 1), "snow_total": _r(g["SNOW_max"].sum(), 1),
               "freeze_days": int(g["freeze"].sum()), "freeze_thaw_days": int(g["freeze_thaw"].sum()),
               "snow_cover_days": int(g["snow_cover"].sum()), "adverse_days": int(g["any_adverse"].sum()),
               "max_daily_prcp": _r(g["PRCP_max"].max()), "max_daily_snow": _r(g["SNOW_max"].max(), 1),
               "tmax_max": _r(g["TMAX_max"].max(), 0), "tmin_min": _r(g["TMIN_min"].min(), 0)}
        for c in CLASSES:
            row[c["key"]] = int((g["cls"] == c["id"]).sum())
        annual.append(row)

    nyears = d["year"].nunique()
    monthly = []
    for m, g in d.groupby("month"):
        row = {"month": int(m), "prcp_total": _r(g["PRCP_mean"].sum() / nyears, 2),
               "snow_total": _r(g["SNOW_max"].sum() / nyears, 1), "freeze_days": _r(g["freeze"].sum() / nyears, 1),
               "adverse_days": _r(g["any_adverse"].sum() / nyears, 1), "tmax_mean": _r(g["TMAX_mean"].mean(), 1),
               "tmin_mean": _r(g["TMIN_mean"].mean(), 1)}
        for c in CLASSES:
            row[c["key"]] = _r((g["cls"] == c["id"]).sum() / nyears, 1)
        monthly.append(row)

    ym = []
    for (yr, m), g in d.groupby(["year", "month"]):
        ym.append({"ym": f"{int(yr)}-{int(m):02d}", "prcp": _r(g["PRCP_mean"].sum(), 2), "snow": _r(g["SNOW_max"].sum(), 1),
                   "wet_days": int(((g["cls"] == 1) | (g["cls"] == 2)).sum()), "snow_days": int((g["cls"] >= 4).sum()),
                   "freezing_days": int((g["cls"] == 3).sum()), "freeze_days": int(g["freeze"].sum())})

    d["dow"] = d["date"].dt.dayofweek
    dow_month = [[int(((d["month"] == m) & (d["dow"] == w) & (d["cls"] > 0)).sum()) for w in range(7)] for m in range(1, 13)]

    top_snow = d.nlargest(10, "SNOW_max")[["date", "SNOW_max", "SNWD_max", "TMIN_min"]]
    top_rain = d.nlargest(10, "PRCP_max")[["date", "PRCP_max", "PRCP_mean", "TMIN_min"]]
    events = {
        "snow": [{"date": r.date.strftime("%Y-%m-%d"), "snow_max": _r(r.SNOW_max, 1), "snwd_max": _r(r.SNWD_max, 1), "tmin": _r(r.TMIN_min, 0)} for r in top_snow.itertuples()],
        "rain": [{"date": r.date.strftime("%Y-%m-%d"), "prcp_max": _r(r.PRCP_max), "prcp_mean": _r(r.PRCP_mean), "tmin": _r(r.TMIN_min, 0)} for r in top_rain.itertuples()],
    }

    per = per.set_index("station")
    stations = []
    for r in st.itertuples():
        p = per.loc[r.station] if r.station in per.index else None
        stations.append({
            "id": r.station, "name": r.name, "lat": _r(r.latitude, 5), "lon": _r(r.longitude, 5), "network": r.network,
            "dist_mi": _r(r.distance_miles, 1), "elev_m": _r(r.elevation_m, 0), "record_start": str(r.record_start),
            "record_end": str(r.record_end), "days": int(r.days_with_any),
            "cov": {"PRCP": _r(r.cov_PRCP), "SNOW": _r(r.cov_SNOW), "SNWD": _r(r.cov_SNWD), "TMAX": _r(r.cov_TMAX), "TMIN": _r(r.cov_TMIN)},
            "composite": {"PRCP": bool(r.in_composite_PRCP), "SNOW": bool(r.in_composite_SNOW), "TMAX": bool(r.in_composite_TMAX)},
            "prcp_total": _r(p["prcp_total_in"], 1) if p is not None else None,
            "prcp_days_ge_1": int(p["prcp_days_ge_1"]) if p is not None else None,
            "snow_total": _r(p["snow_total_in"], 1) if p is not None else None,
            "snow_days_ge_1": int(p["snow_days_ge_1"]) if p is not None else None,
            "snow_max": _r(p["snow_max_in"], 1) if p is not None else None,
            "snow_max_date": str(p["snow_max_date"]) if p is not None else None,
            "freeze_days": int(p["freeze_days_tmin_le_32"]) if p is not None else None,
        })

    daily = {
        "date": [x.strftime("%Y-%m-%d") for x in d["date"]],
        "prcp_mean": [_r(x) for x in d["PRCP_mean"]], "prcp_max": [_r(x) for x in d["PRCP_max"]],
        "snow_max": [_r(x, 1) for x in d["SNOW_max"]], "snwd_max": [_r(x, 1) for x in d["SNWD_max"]],
        "tmax": [_r(x, 0) for x in d["TMAX_max"]], "tmin": [_r(x, 0) for x in d["TMIN_min"]],
        "cls": [int(x) for x in d["cls"]],
    }
    period = {
        "start": daily["date"][0], "end": daily["date"][-1], "days": int(len(d)), "years": nyears,
        "per_year": {c["key"]: _r((d["cls"] == c["id"]).sum() / nyears, 1) for c in CLASSES},
        "days_in_class": {c["key"]: int((d["cls"] == c["id"]).sum()) for c in CLASSES},
        "per_year_freeze": _r(d["freeze"].sum() / nyears, 1), "per_year_freeze_thaw": _r(d["freeze_thaw"].sum() / nyears, 1),
        "per_year_snow_cover": _r(d["snow_cover"].sum() / nyears, 1), "per_year_adverse": _r(d["any_adverse"].sum() / nyears, 1),
        "prcp_total": _r(d["PRCP_mean"].sum(), 1), "snow_total": _r(d["SNOW_max"].sum(), 1),
    }
    out = {
        "status": "loaded", "source": "NOAA GHCN-Daily via NCEI Access Data Service", "folder": os.path.relpath(area, ROOT),
        "files": file_manifest(base), "run_info": run_info, "classes": CLASSES, "period": period, "annual": annual, "monthly": monthly, "ym": ym,
        "dow_month": dow_month, "events": events, "stations": stations, "daily": daily,
        "gaps": [
            "GHCN-Daily has no visibility, fog, wind, or pavement-temperature fields. Fog / low-visibility exposure needs hourly ASOS (Lehigh Valley Intl KABE is the nearest first-order station) or NJDOT RWIS feeds.",
            "Daily totals cannot resolve time of day; hour-of-day overlap with traffic comes from INRIX speeds and the crash time stamps.",
            "Temperature comes from only two COOP stations (Phillipsburg, Flemington 5 NNW); the ridge top on Jugtown Mountain is likely colder than either.",
        ],
    }
    out["_cls_by_date"] = dict(zip(daily["date"], daily["cls"]))  # internal join key, stripped before output
    return out


# --------------------------------------------------------------------------
# NJ crash extract (Safety Voyager / public portal tables)
# --------------------------------------------------------------------------
def load_crashes(cls_by_date=None):
    base = find_dir("Crash*", "crash*", "NJ Crash*", "NJTR*")
    if not base or not os.path.exists(os.path.join(base, "Crash.csv")):
        return {"status": "missing", "expected_folder": "Input Data/Crash_<years>/Crash.csv (+ Vehicle.csv, Unit.csv, Person.csv)",
                "expected": "NJDOT crash tables for the I-78 municipalities (Crash, Vehicle, Unit, Person) 2021 to 2025."}
    c = pd.read_csv(os.path.join(base, "Crash.csv"), low_memory=False)
    n_extract = len(c)
    munis = sorted(c["Municipality"].dropna().unique().tolist())
    c = c[c["routenumber"] == 78].copy()
    c["date"] = pd.to_datetime(c["dateofcrash"], format="%m/%d/%Y", errors="coerce")
    c = c[c["date"].notna()].copy()
    c["year"] = c["date"].dt.year
    c["month"] = c["date"].dt.month
    c["dow"] = c["date"].dt.dayofweek
    t = pd.to_numeric(c["timeofcrash"], errors="coerce")
    c["hour"] = (t // 100).where(t.notna(), np.nan)
    c["dir"] = c["Roadway Direction"].map({"East": "E", "West": "W"}).fillna("X")
    c["mp"] = pd.to_numeric(c["milepost"], errors="coerce")
    c.loc[(c["mp"] > 20), "mp"] = np.nan  # a few rows carry the SRI-level milepost (60+); treat as unknown
    c["sev"] = c["Highest Injury Severity Rating"].map(SEV_MAP).fillna("O")
    c["surface"] = c["Surface Condition"].fillna("Unknown")
    c["weather"] = c["Weather Condition"].fillna("Unknown")
    c["light"] = c["Light Condition"].fillna("Unknown")
    c["ctype"] = c["Crash Type"].fillna("Unknown")
    c["grade"] = c["Road Character - Grade"].fillna("Unknown")
    c["align"] = c["Road Horizontal Alignment"].fillna("Unknown")
    c["tcz"] = c["Traffic Control Zone"].fillna("None")
    c["adverse"] = c["surface"].isin(ADVERSE_SURFACE) | c["weather"].isin(ADVERSE_WEATHER)
    c["injury_crash"] = c["sev"].isin(["K", "A", "B", "C"])
    c["ka"] = c["sev"].isin(["K", "A"])
    c["injuries"] = pd.to_numeric(c["injurycount"], errors="coerce").fillna(0).astype(int)
    c["fatalities"] = pd.to_numeric(c["fatalitycount"], errors="coerce").fillna(0).astype(int)
    c["same_dir"] = c["ctype"].isin(["Same Direction (Rear End)", "Same Direction (Side Swipe)"])
    c["workzone"] = c["tcz"].isin(["Construction Zone", "Maintenance Zone", "Utility Zone"])

    # vehicles: truck involvement
    truck_ids = set()
    vpath = os.path.join(base, "Vehicle.csv")
    if os.path.exists(vpath):
        v = pd.read_csv(vpath, low_memory=False)
        is_truck = v["Vehicle Type"].isin(TRUCK_TYPES) | (v["Weight Rating"] == "> 26,001 lbs")
        truck_ids = set(v.loc[is_truck, "crashreportid"].astype(int))
    c["truck"] = c["id_cr"].astype(int).isin(truck_ids)

    # units: contributing factors
    speed_ids, follow_ids, surface_ids = set(), set(), set()
    upath = os.path.join(base, "Unit.csv")
    if os.path.exists(upath):
        u = pd.read_csv(upath, low_memory=False)
        f = u[["Factor 1", "Factor 2"]].fillna("")
        speed_ids = set(u.loc[(f == "Unsafe Speed").any(axis=1), "crashreportid"].astype(int))
        follow_ids = set(u.loc[(f == "Following Too Closely").any(axis=1), "crashreportid"].astype(int))
        surface_ids = set(u.loc[(f == "Road Surface Condition").any(axis=1), "crashreportid"].astype(int))
    c["f_speed"] = c["id_cr"].astype(int).isin(speed_ids)
    c["f_follow"] = c["id_cr"].astype(int).isin(follow_ids)
    c["f_surface"] = c["id_cr"].astype(int).isin(surface_ids)

    # climate class on the crash day
    if cls_by_date:
        c["cls"] = c["date"].dt.strftime("%Y-%m-%d").map(cls_by_date).fillna(-1).astype(int)
    else:
        c["cls"] = -1

    # secondary crashes: another crash same direction within the prior 2 h and 0-3 mi upstream
    c = c.sort_values("date").reset_index(drop=True)
    c["dt"] = c["date"] + pd.to_timedelta(c["hour"].fillna(12) * 60 + (t % 100).fillna(0), unit="m")
    sec = np.zeros(len(c), dtype=bool)
    for d_, g in c[c["dir"].isin(["E", "W"])].groupby("dir"):
        g = g.sort_values("dt")
        times = g["dt"].values; mps = g["mp"].values; idx = g.index.values
        for i in range(len(g)):
            if np.isnan(mps[i]):
                continue
            lo = times[i] - np.timedelta64(120, "m")
            j = i - 1
            while j >= 0 and times[j] >= lo:
                if not np.isnan(mps[j]):
                    gap = (mps[j] - mps[i]) if d_ == "E" else (mps[i] - mps[j])  # prior crash downstream of this one
                    if 0 <= gap <= 3:
                        sec[idx[i]] = True
                        break
                j -= 1
    c["secondary"] = sec

    years = sorted(c["year"].unique().tolist())
    nyears = len(years)
    mp_valid = c["mp"].dropna()
    mp_lo, mp_hi = float(mp_valid.quantile(0.005)), float(mp_valid.quantile(0.995))
    section_len = max(0.5, mp_hi - mp_lo)

    by_year = []
    for y, g in c.groupby("year"):
        by_year.append({"year": int(y), "total": int(len(g)), **{s: int((g["sev"] == s).sum()) for s in "KABCO"},
                        "injuries": int(g["injuries"].sum()), "fatalities": int(g["fatalities"].sum()),
                        "adverse": int(g["adverse"].sum()), "adverse_injury": int((g["adverse"] & g["injury_crash"]).sum()),
                        "truck": int(g["truck"].sum()), "workzone": int(g["workzone"].sum()), "secondary": int(g["secondary"].sum()),
                        "same_dir": int(g["same_dir"].sum())})

    def by_field(field):
        rows = []
        for k, g in c.groupby(field):
            rows.append({"k": str(k), "n": int(len(g)), "injury": int(g["injury_crash"].sum()), "ka": int(g["ka"].sum()),
                         "truck": int(g["truck"].sum())})
        return sorted(rows, key=lambda r: -r["n"])

    # milepost bins by direction
    bins = np.arange(np.floor(mp_lo * 2) / 2, np.ceil(mp_hi * 2) / 2 + 0.5, 0.5)
    mp_rows = []
    for b in bins[:-1]:
        g = c[(c["mp"] >= b) & (c["mp"] < b + 0.5)]
        row = {"mp": float(b)}
        for d_ in ("E", "W"):
            gd = g[g["dir"] == d_]
            row[d_] = int(len(gd)); row[d_ + "_adv"] = int(gd["adverse"].sum()); row[d_ + "_inj"] = int(gd["injury_crash"].sum())
            row[d_ + "_truck"] = int(gd["truck"].sum())
        mp_rows.append(row)

    hour_dow = [[int(((c["dow"] == w) & (c["hour"] == hh)).sum()) for hh in range(24)] for w in range(7)]
    hour_dow_adv = [[int(((c["dow"] == w) & (c["hour"] == hh) & c["adverse"]).sum()) for hh in range(24)] for w in range(7)]
    by_hour = [{"hour": hh, "n": int((c["hour"] == hh).sum()), "adverse": int(((c["hour"] == hh) & c["adverse"]).sum()),
                "injury": int(((c["hour"] == hh) & c["injury_crash"]).sum())} for hh in range(24)]
    by_month = [{"month": m, "n": int((c["month"] == m).sum()), "adverse": int(((c["month"] == m) & c["adverse"]).sum()),
                 "injury": int(((c["month"] == m) & c["injury_crash"]).sum())} for m in range(1, 13)]

    # crashes by climate class of the day (for the relative-risk table)
    by_cls = []
    for k in range(len(CLASSES)):
        g = c[c["cls"] == k]
        by_cls.append({"cls": k, "n": int(len(g)), "injury": int(g["injury_crash"].sum()), "adverse_coded": int(g["adverse"].sum()),
                       "truck": int(g["truck"].sum())})
    unmatched = int((c["cls"] < 0).sum())

    # grade x direction
    grade_dir = []
    for (g_, d_), g in c[c["dir"].isin(["E", "W"])].groupby(["grade", "dir"]):
        grade_dir.append({"grade": g_, "dir": d_, "n": int(len(g)), "adverse": int(g["adverse"].sum()), "truck": int(g["truck"].sum()),
                          "injury": int(g["injury_crash"].sum())})

    # top crash days
    day_counts = c.groupby(c["date"].dt.strftime("%Y-%m-%d")).agg(n=("id_cr", "size"), inj=("injury_crash", "sum"), adv=("adverse", "sum"), cls=("cls", "first"))
    top_days = [{"date": d_, "n": int(r.n), "injury": int(r.inj), "adverse": int(r.adv), "cls": int(r.cls)} for d_, r in day_counts.sort_values("n", ascending=False).head(15).iterrows()]
    # daily counts for the statistical tests (all mainline crashes, not only the geocoded ones)
    day_all = c.groupby(c["date"].dt.strftime("%Y-%m-%d")).agg(n=("id_cr", "size"), inj=("injury_crash", "sum"), truck=("truck", "sum"),
                                                                adv=("adverse", "sum"), same_dir=("same_dir", "sum"), wz=("workzone", "sum"))
    crash_daily = [{"date": d_, "n": int(r.n), "inj": int(r.inj), "truck": int(r.truck), "adv": int(r.adv), "same_dir": int(r.same_dir), "wz": int(r.wz)}
                   for d_, r in day_all.iterrows()]

    adverse = c[c["adverse"]]
    stats = {
        "n": int(len(c)), "n_extract": int(n_extract), "years": years, "nyears": nyears,
        "mp_lo": _r(mp_lo, 1), "mp_hi": _r(mp_hi, 1), "section_len": _r(section_len, 1),
        "injury_crashes": int(c["injury_crash"].sum()), "injuries": int(c["injuries"].sum()), "fatalities": int(c["fatalities"].sum()),
        "ka": int(c["ka"].sum()),
        "adverse_n": int(len(adverse)), "adverse_share": _r(len(adverse) / len(c), 3),
        "adverse_injury_crashes": int(adverse["injury_crash"].sum()), "adverse_injuries": int(adverse["injuries"].sum()),
        "adverse_fatalities": int(adverse["fatalities"].sum()),
        "adverse_inj_per_mi_yr": _r((adverse["injuries"].sum() + adverse["fatalities"].sum()) / section_len / nyears, 2),
        "all_inj_per_mi_yr": _r((c["injuries"].sum() + c["fatalities"].sum()) / section_len / nyears, 2),
        "same_dir_share": _r(c["same_dir"].mean(), 3),
        "truck_n": int(c["truck"].sum()), "truck_share": _r(c["truck"].mean(), 3),
        "workzone_n": int(c["workzone"].sum()), "secondary_n": int(c["secondary"].sum()), "secondary_share": _r(c["secondary"].mean(), 3),
        "f_speed_n": int(c["f_speed"].sum()), "f_follow_n": int(c["f_follow"].sum()), "f_surface_n": int(c["f_surface"].sum()),
        "dark_n": int(c["light"].str.startswith("Dark").sum()), "fog_n": int((c["weather"] == "Fog/Smog/Smoke").sum()),
        "snow_ice_n": int(c["surface"].isin(["Snowy", "Icy", "Slush"]).sum()), "wet_n": int((c["surface"] == "Wet").sum()),
        "crash_day_cls_unmatched": unmatched,
        "pdo": int((c["sev"] == "O").sum()), "inj_crash_by_sev": {s: int((c["sev"] == s).sum()) for s in "KABCO"},
    }

    pts = c[c["latitude"].notna() & c["longitude"].notna()]
    points = [{"lat": _r(r.latitude, 5), "lon": _r(r.longitude, 5), "date": r.date.strftime("%Y-%m-%d"), "hour": None if pd.isna(r.hour) else int(r.hour),
               "dir": r.dir, "mp": _r(r.mp, 2), "sev": r.sev, "surface": r.surface, "weather": r.weather, "type": r.ctype,
               "truck": bool(r.truck), "wz": bool(r.workzone), "sec": bool(r.secondary), "cls": int(r.cls)}
              for r in pts.itertuples()]
    points = [p for p in points if 40.5 < p["lat"] < 40.8 and -75.3 < p["lon"] < -74.8]

    return {
        "status": "loaded", "folder": os.path.relpath(base, ROOT), "files": file_manifest(base), "municipalities": munis,
        "source": "NJDOT crash tables (Crash, Vehicle, Unit) filtered to route 78; municipalities in the extract: " + ", ".join(munis),
        "stats": stats, "by_year": by_year, "by_surface": by_field("surface"), "by_weather": by_field("weather"),
        "by_light": by_field("light"), "by_type": by_field("ctype"), "by_grade": by_field("grade"), "by_align": by_field("align"),
        "by_tcz": by_field("tcz"), "by_sev": [{"k": s, "n": int((c["sev"] == s).sum())} for s in "KABCO"],
        "mp_bins": mp_rows, "hour_dow": hour_dow, "hour_dow_adv": hour_dow_adv, "by_hour": by_hour, "by_month": by_month,
        "by_cls": by_cls, "grade_dir": grade_dir, "top_days": top_days, "points": points, "daily": crash_daily,
        "caveats": [
            "Extract is by municipality (Greenwich Twp, Bethlehem Twp) so I-78 coverage runs about MP %s to %s; Pohatcong Twp (MP 0 to 3.7) and Union Twp (MP 10.4 to 13) are not yet included." % (_r(mp_lo, 1), _r(mp_hi, 1)),
            "Only route-78 mainline records are used (%d of %d rows in the extract); ramp and crossroad crashes are excluded." % (len(c), n_extract),
            "Adverse = surface Wet/Snowy/Icy/Slush/Water or weather Rain/Snow/Fog/Sleet/Freezing rain, following CDOT Chapter 3.1.1.",
            "Truck-involved = any unit coded as a truck type or GVWR > 26,001 lbs.",
            "Secondary = a crash in the same direction within 2 h after, and 0 to 3 mi upstream of, an earlier crash (time-of-day and milepost based, so an approximation).",
            "Person-level fields are not used; no personal identifiers are carried into the dashboard.",
        ],
    }


# --------------------------------------------------------------------------
# INRIX (reads the summary produced by scripts/preprocess_inrix.py)
# --------------------------------------------------------------------------
def load_inrix():
    base = find_dir("INRIX*", "Inrix*", "inrix*", "Travel Time*")
    if not base:
        return {"status": "missing", "expected_folder": "Input Data/INRIX_<years>/",
                "expected": "INRIX XD/TMC 5-minute speeds for I-78 EB and WB plus the TMC_Identification files."}
    if not os.path.exists(INRIX_SUMMARY):
        return {"status": "found_unparsed", "folder": os.path.relpath(base, ROOT),
                "note": "Raw files found. Run `python -I scripts/preprocess_inrix.py` to build dashboard/data/inrix_summary.json."}
    with open(INRIX_SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    s["status"] = "loaded"
    s["folder"] = os.path.relpath(base, ROOT)
    s["file_manifest"] = file_manifest(base)
    s["rows_parsed"] = int(sum(t.get("n", 0) for t in s.get("tmcs", [])))
    # road-following geometry from scripts/fetch_geometry.py (OSRM); straight end points otherwise
    gp = os.path.join(ROOT, "dashboard", "data", "tmc_geometry.json")
    geom = {}
    if os.path.exists(gp):
        with open(gp, encoding="utf-8") as f:
            geom = json.load(f)
    for t in s["tmcs"]:
        g = geom.get(t["tmc"])
        t["geom"] = g["coords"] if g and len(g.get("coords", [])) >= 2 else [[t["lat0"], t["lon0"]], [t["lat1"], t["lon1"]]]
        t["geom_routed"] = bool(g and not g.get("straight"))
    line = []
    for t in sorted([t for t in s["tmcs"] if t["dir"] == "E"], key=lambda t: t["order"]):
        for c in t["geom"]:
            if not line or line[-1] != c:
                line.append(c)
    s["centerline_routed"] = line
    s["geometry_source"] = "OSRM-routed (%d of %d segments)" % (sum(t["geom_routed"] for t in s["tmcs"]), len(s["tmcs"])) if geom else "straight lines between INRIX end points"
    s["source"] = "INRIX TMC 5-minute speeds and travel times, %s to %s, %d TMC segments" % (s["period"]["start"], s["period"]["end"], len(s["tmcs"]))
    # keep the event list small for the page
    s["events"] = s["events"][:200]
    return s


def load_workzones():
    base = find_dir("Work Zone*", "Workzone*", "workzone*", "Lane Closure*")
    if not base:
        return {"status": "missing", "expected_folder": "Input Data/Work Zones/",
                "expected": "NJDOT lane-closure / permit log or OpenReach work-zone events for I-78, with start and end time stamps, direction, mile posts and lanes closed. Interim proxy: crashes coded 'Construction Zone' in the crash extract."}
    return {"status": "found_unparsed", "folder": os.path.relpath(base, ROOT), "files": file_manifest(base)}


# --------------------------------------------------------------------------
# Cross-dataset analysis
# --------------------------------------------------------------------------
def cross_analysis(climate, inrix, crashes):
    out = {}
    if climate.get("status") != "loaded":
        return out
    cls_by_date = climate["_cls_by_date"]
    if inrix.get("status") == "loaded":
        d = pd.DataFrame(inrix["daily"])
        d = d[d["speed"].notna()].copy()
        d["cls"] = d["date"].map(cls_by_date).fillna(-1).astype(int)
        d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
        rows = []
        for (dir_, k), g in d[d["cls"] >= 0].groupby(["dir", "cls"]):
            rows.append({"dir": dir_, "cls": int(k), "days": int(len(g)), "speed": _r(g["speed"].mean(), 1),
                         "tt_mean": _r(g["tt_mean"].mean(), 2), "tt95": _r(g["tt95"].mean(), 2),
                         "h_lt50": _r(g["h_lt50"].mean(), 2), "h_any_lt35": _r(g["h_any_lt35"].mean(), 2),
                         "p_days_with_slowdown": _r((g["h_any_lt35"] > 0).mean(), 3),
                         "min_speed": _r(g["min_speed"].mean(), 1)})
        out["speed_by_cls"] = rows
        # weekday only
        rows = []
        for (dir_, k), g in d[(d["cls"] >= 0) & (d["dow"] < 5)].groupby(["dir", "cls"]):
            rows.append({"dir": dir_, "cls": int(k), "days": int(len(g)), "speed": _r(g["speed"].mean(), 1), "tt_am": _r(g["tt_am"].mean(), 2),
                         "tt_pm": _r(g["tt_pm"].mean(), 2), "h_any_lt35": _r(g["h_any_lt35"].mean(), 2)})
        out["speed_by_cls_weekday"] = rows
        # events on adverse vs dry days
        ev = pd.DataFrame(inrix.get("events", []))
        allev = inrix.get("event_stats", {})
        out["event_stats"] = allev
    if crashes.get("status") == "loaded" and inrix.get("status") == "loaded":
        # crash days vs corridor delay: mean h_any_lt35 on days with >=1 crash vs none
        d = pd.DataFrame(inrix["daily"])
        d = d[d["speed"].notna()].copy()
        crash_days = {}
        for p in crashes["points"]:
            crash_days[p["date"]] = crash_days.get(p["date"], 0) + 1
        d["crashes"] = d["date"].map(crash_days).fillna(0).astype(int)
        rows = []
        for dir_, g in d.groupby("dir"):
            for lab, m in (("no crash", g["crashes"] == 0), ("1 crash", g["crashes"] == 1), ("2+ crashes", g["crashes"] >= 2)):
                gg = g[m]
                rows.append({"dir": dir_, "crash_day": lab, "days": int(len(gg)), "h_any_lt35": _r(gg["h_any_lt35"].mean(), 2),
                             "tt95": _r(gg["tt95"].mean(), 2), "p_days_with_slowdown": _r((gg["h_any_lt35"] > 0).mean(), 3)})
        out["delay_by_crash_day"] = rows
    # formal statistical tests (scipy): rate ratios with CIs, Poisson regression, rank tests, correlation matrix
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:  # python -I does not add the script directory
            sys.path.insert(0, here)
        import stats_tests
        out["stats"] = stats_tests.run_all(climate, crashes, inrix, [c["label"].split(" (")[0] for c in CLASSES])
    except Exception as e:  # keep the build alive; the tab explains the gap
        print("  stats_tests failed:", repr(e))
        out["stats"] = {"error": repr(e)}
    return out


def main():
    with open(CORRIDOR, encoding="utf-8") as f:
        corridor = json.load(f)
    climate = load_climate()
    crashes = load_crashes(climate.get("_cls_by_date"))
    inrix = load_inrix()
    workzones = load_workzones()
    cross = cross_analysis(climate, inrix, crashes)
    climate.pop("_cls_by_date", None)
    data = {
        "meta": {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "repo": "I78-VSL", "builder": "scripts/build_dashboard.py"},
        "corridor": corridor, "climate": climate, "inrix": inrix, "crashes": crashes, "workzones": workzones, "cross": cross,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    # Full-page INRIX map first: the dashboard embeds a copy of it (tab 2, "INRIX travel time" view).
    inrix_html = build_inrix_map(data, corridor)

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    if "/*__DATA__*/" not in html:
        sys.exit("template is missing the /*__DATA__*/ placeholder")
    html = html.replace("/*__DATA__*/", blob, 1)
    if inrix_html and "/*__INRIX_MAP__*/null" in html:
        # JS string literal holding the whole page; it becomes the iframe's srcdoc. "</" is escaped so the
        # inner </script> tags cannot close the dashboard's own script block.
        html = html.replace("/*__INRIX_MAP__*/null", json.dumps(inrix_html, ensure_ascii=False).replace("</", "<\\/"), 1)
    # Inline Leaflet so the standalone page does not depend on a CDN (only the basemap tiles need the network).
    html2 = inline_leaflet(html)
    if html2 is html:
        print("  warning: dashboard/vendor/leaflet-1.9.4.* not found; page keeps the CDN links")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html2)
    kb = os.path.getsize(OUT_HTML) / 1024
    print(f"wrote {os.path.relpath(OUT_HTML, ROOT)} ({kb:,.0f} KB)")
    for k in ("climate", "inrix", "crashes", "workzones"):
        print(f"  {k:10s} {data[k]['status']}")


def inline_leaflet(html):
    vendor = os.path.join(ROOT, "dashboard", "vendor")
    lj, lc = os.path.join(vendor, "leaflet-1.9.4.js"), os.path.join(vendor, "leaflet-1.9.4.css")
    if not (os.path.exists(lj) and os.path.exists(lc)):
        return html
    with open(lj, encoding="utf-8") as f:
        js = f.read().replace("</script", "<\\/script")
    with open(lc, encoding="utf-8") as f:
        css = f.read()
    html = html.replace('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">', "<style>/* Leaflet 1.9.4, BSD-2-Clause, inlined from dashboard/vendor */\n" + css + "</style>", 1)
    html = html.replace('<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>', "<script>/* Leaflet 1.9.4, BSD-2-Clause, inlined from dashboard/vendor */\n" + js + "</script>", 1)
    return html


def build_inrix_map(data, corridor):
    """Stand-alone full-page INRIX map (I78_INRIX_Map.html) from dashboard/inrix_map_template.html."""
    inrix = data["inrix"]
    tpl = os.path.join(ROOT, "dashboard", "inrix_map_template.html")
    out = os.path.join(ROOT, "I78_INRIX_Map.html")
    bins_path = os.path.join(ROOT, "dashboard", "data", "inrix_bins15.json")
    if inrix.get("status") != "loaded" or not os.path.exists(tpl):
        print("  INRIX map: skipped (INRIX not loaded or template missing)")
        return
    bins = {"daytypes": ["all", "weekday", "weekend"], "bins": {}}
    if os.path.exists(bins_path):
        with open(bins_path, encoding="utf-8") as f:
            bins = json.load(f)
    else:
        print("  INRIX map: dashboard/data/inrix_bins15.json missing; re-run preprocess_inrix.py for the time-of-day bands")
    dist = inrix.get("tmc_dist", {})
    wc = inrix.get("worst_tmc_counts", {})
    tmcs = []
    for t in inrix["tmcs"]:
        d = dist.get(t["tmc"], {})
        tmcs.append({"tmc": t["tmc"], "dir": t["dir"], "order": t["order"], "name": t["name"], "miles": t["miles"], "geom": t["geom"],
                     "ref_speed": t["ref_speed"], "n": t["n"], "speed_mean": t["speed_mean"], "p_lt50": t["p_lt50"], "p_lt35": t["p_lt35"],
                     "episodes": int(wc.get(t["tmc"], 0)), "ff_tt": d.get("ff_tt", round(t["miles"] / max(t["ref_speed"], 1) * 60, 2)),
                     "pct": d.get("pct", {"all": [None] * 7}), "pct_labels": d.get("pct_labels", [5, 25, 50, 75, 85, 95, 99]),
                     "edges": d.get("edges", []), "hist": (d.get("hist") or {}).get("all", []), "bti": d.get("bti"), "p_gt_1_5ff": d.get("p_gt_1_5ff")})
    short = lambda lab: "PA line" if lab.startswith("PA-NJ") else (lab.split(" - ")[0] if lab.startswith("Exit") else lab.split(" ")[0])
    payload = {
        "meta": {"generated": data["meta"]["generated"], "period": inrix["period"], "rows": inrix.get("rows_parsed", 0), "geometry": inrix.get("geometry_source", "")},
        "tmcs": tmcs, "daytypes": bins["daytypes"], "bins": bins["bins"],
        "events": [e for e in inrix.get("events", [])],
        "interchanges": [{"mp": x["mp"], "lat": x["lat"], "lon": x["lon"], "short": short(x["label"]), "label": x["label"]} for x in corridor["interchanges"]],
    }
    with open(tpl, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = inline_leaflet(html.replace("/*__DATA__*/null", blob, 1))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out) / 1024:,.0f} KB)")
    return html


if __name__ == "__main__":
    main()
