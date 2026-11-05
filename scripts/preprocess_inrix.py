"""Aggregate the INRIX 5-minute TMC files into small tables for the dashboard.

Input : Input Data/INRIX*/I78-Travel-Time-VSL-5-Minutes-*.csv and *TMC_Identification.csv
Output: dashboard/data/inrix_summary.json (small) and dashboard/data/inrix_corridor_5min.csv
        (per-direction corridor travel time at every 5-minute stamp, ~1 M rows, for later use)

Run from the repo root:
    python -I scripts/preprocess_inrix.py

The build script (scripts/build_dashboard.py) reads inrix_summary.json; it does not
touch the raw files, so this only needs to be re-run when new INRIX data arrives.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "Input Data")
OUT_DIR = os.path.join(ROOT, "dashboard", "data")
OUT_JSON = os.path.join(OUT_DIR, "inrix_summary.json")
OUT_5MIN = os.path.join(OUT_DIR, "inrix_corridor_5min.csv")

SLOW = 50      # mph: corridor "degraded" threshold (speed-harmonization relevant)
VERY_SLOW = 35 # mph: queue / incident threshold on any single segment
EVENT_MIN_BINS = 3  # 15 minutes


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def find_inrix_dir():
    for pat in ("INRIX*", "Inrix*", "inrix*"):
        hits = [p for p in glob.glob(os.path.join(INPUT, pat)) if os.path.isdir(p)]
        if hits:
            return hits[0]
    sys.exit("no INRIX folder under Input Data/")


def load_tmcs(base):
    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(glob.glob(os.path.join(base, "*TMC_Identification.csv")))]
    t = pd.concat(frames).drop_duplicates("tmc").sort_values(["direction", "road_order"]).reset_index(drop=True)
    t["dir"] = t["direction"].str[0]  # E / W
    return t


def main():
    base = find_inrix_dir()
    tmcs = load_tmcs(base)
    tmc_dir = dict(zip(tmcs.tmc, tmcs.dir))
    tmc_miles = dict(zip(tmcs.tmc, tmcs.miles))
    dirs = {d: list(tmcs.loc[tmcs.dir == d, "tmc"]) for d in ("E", "W")}
    dir_miles = {d: float(tmcs.loc[tmcs.dir == d, "miles"].sum()) for d in dirs}

    files = sorted(glob.glob(os.path.join(base, "*5-Minutes*.csv")))
    per_tmc_parts = []
    corridor_parts = []
    dist_parts = []  # per-record travel times kept for the segment distributions
    ref_speed = {}

    for f in files:
        log(f"reading {os.path.basename(f)}")
        df = pd.read_csv(f, usecols=["tmc_code", "measurement_tstamp", "speed", "reference_speed",
                                     "travel_time_minutes", "confidence_score"],
                         dtype={"tmc_code": "category", "speed": "float32", "reference_speed": "float32",
                                "travel_time_minutes": "float32", "confidence_score": "float32"})
        df["ts"] = pd.to_datetime(df["measurement_tstamp"], format="%Y-%m-%d %H:%M:%S")
        df = df.drop(columns=["measurement_tstamp"])
        df = df[df["tmc_code"].isin(tmc_dir.keys())]
        log(f"  {len(df):,} rows, {df.ts.min()} to {df.ts.max()}")

        # reference speed per tmc (median over file)
        for k, v in df.groupby("tmc_code", observed=True)["reference_speed"].median().items():
            ref_speed.setdefault(k, []).append(float(v))

        # per-tmc stats by (tmc, year-month, hour)
        g = df.assign(ym=df.ts.dt.strftime("%Y-%m"), hour=df.ts.dt.hour, wd=df.ts.dt.dayofweek < 5)
        part = g.groupby(["tmc_code", "ym", "hour", "wd"], observed=True).agg(
            n=("speed", "size"), speed_mean=("speed", "mean"),
            lt50=("speed", lambda s: float((s < SLOW).sum())),
            lt35=("speed", lambda s: float((s < VERY_SLOW).sum())),
            conf30=("confidence_score", lambda s: float((s >= 30).sum())),
        ).reset_index()
        per_tmc_parts.append(part)
        dist_parts.append(pd.DataFrame({"tmc": df["tmc_code"].astype(str), "tt": df["travel_time_minutes"].astype("float32"),
                                        "spd": df["speed"].astype("float32"),
                                        "hour": df["ts"].dt.hour.astype("int8"), "wd": (df["ts"].dt.dayofweek < 5),
                                        "dow": df["ts"].dt.dayofweek.astype("int8"),
                                        "bin15": (df["ts"].dt.hour * 4 + df["ts"].dt.minute // 15).astype("int16")}))

        # corridor pivot
        tt = df.pivot(index="ts", columns="tmc_code", values="travel_time_minutes")
        sp = df.pivot(index="ts", columns="tmc_code", values="speed")
        for d, cols in dirs.items():
            cols = [c for c in cols if c in tt.columns]
            miles = np.array([tmc_miles[c] for c in cols], dtype="float64")
            ttd = tt[cols]
            present = ttd.notna()
            miles_present = present.values @ miles
            tt_sum = ttd.fillna(0).values @ np.ones(len(cols))
            ok = miles_present >= 0.8 * dir_miles[d]
            corr_tt = np.where(ok, tt_sum * dir_miles[d] / np.where(miles_present > 0, miles_present, np.nan), np.nan)
            corr_speed = np.where(ok, dir_miles[d] / (corr_tt / 60.0), np.nan)
            min_speed = sp[cols].min(axis=1).values
            slow_tmc = sp[cols].idxmin(axis=1).astype(str).values
            corridor_parts.append(pd.DataFrame({"ts": ttd.index, "dir": d, "tt_min": corr_tt.astype("float32"),
                                                "speed": corr_speed.astype("float32"), "min_tmc_speed": min_speed.astype("float32"),
                                                "slow_tmc": slow_tmc}))
        del df, tt, sp, g

    log("combining")
    cor = pd.concat(corridor_parts).sort_values(["dir", "ts"]).reset_index(drop=True)
    cor.loc[cor.min_tmc_speed.isna(), "slow_tmc"] = ""
    ref = {k: float(np.median(v)) for k, v in ref_speed.items()}
    ff_tt = {d: sum(tmc_miles[c] / ref[c] * 60 for c in dirs[d] if c in ref) for d in dirs}

    os.makedirs(OUT_DIR, exist_ok=True)
    cor.to_csv(OUT_5MIN, index=False, float_format="%.2f")
    log(f"wrote {OUT_5MIN} ({len(cor):,} rows)")

    cor["date"] = cor.ts.dt.strftime("%Y-%m-%d")
    cor["hour"] = cor.ts.dt.hour
    cor["dow"] = cor.ts.dt.dayofweek
    cor["daytype"] = np.where(cor.dow < 5, "weekday", np.where(cor.dow == 5, "saturday", "sunday"))
    cor["ym"] = cor.ts.dt.strftime("%Y-%m")

    # ---- daily per direction -------------------------------------------
    daily = []
    for (d, date), g in cor.groupby(["dir", "date"]):
        v = g.tt_min.dropna()
        if len(v) < 100:
            daily.append({"dir": d, "date": date, "n": int(len(v))})
            continue
        am = g[(g.hour >= 6) & (g.hour < 9)].tt_min.mean()
        pm = g[(g.hour >= 15) & (g.hour < 18)].tt_min.mean()
        daily.append({"dir": d, "date": date, "n": int(len(v)),
                      "speed": round(float(g.speed.mean()), 1),
                      "tt_mean": round(float(v.mean()), 2), "tt95": round(float(v.quantile(0.95)), 2),
                      "tt_max": round(float(v.max()), 2),
                      "min_speed": round(float(g.min_tmc_speed.min()), 1),
                      "h_lt50": round(float((g.speed < SLOW).sum()) / 12, 2),
                      "h_any_lt35": round(float((g.min_tmc_speed < VERY_SLOW).sum()) / 12, 2),
                      "tt_am": None if pd.isna(am) else round(float(am), 2),
                      "tt_pm": None if pd.isna(pm) else round(float(pm), 2)})

    # ---- hourly profile by direction and daytype ------------------------
    prof = []
    for (d, dt_, hr), g in cor.groupby(["dir", "daytype", "hour"]):
        v = g.tt_min.dropna()
        prof.append({"dir": d, "daytype": dt_, "hour": int(hr), "speed": round(float(g.speed.mean()), 1),
                     "tt_mean": round(float(v.mean()), 2), "tt85": round(float(v.quantile(0.85)), 2),
                     "tt95": round(float(v.quantile(0.95)), 2),
                     "p_lt50": round(float((g.speed < SLOW).mean()), 4),
                     "p_any_lt35": round(float((g.min_tmc_speed < VERY_SLOW).mean()), 4)})

    # ---- monthly ----------------------------------------------------------
    monthly = []
    for (d, ym), g in cor.groupby(["dir", "ym"]):
        v = g.tt_min.dropna()
        monthly.append({"dir": d, "ym": ym, "speed": round(float(g.speed.mean()), 1),
                        "tti": round(float(v.mean() / ff_tt[d]), 3), "pti": round(float(v.quantile(0.95) / ff_tt[d]), 3),
                        "h_lt50": round(float((g.speed < SLOW).sum()) / 12, 1),
                        "h_any_lt35": round(float((g.min_tmc_speed < VERY_SLOW).sum()) / 12, 1),
                        "coverage": round(float(len(v)) / max(1, len(g)), 3)})

    # ---- events: contiguous bins with any segment < VERY_SLOW -------------
    events = []
    for d, g in cor.groupby("dir"):
        g = g.sort_values("ts")
        flag = (g.min_tmc_speed < VERY_SLOW).values
        ts = g.ts.values; tt = g.tt_min.values; ms = g.min_tmc_speed.values; st = g.slow_tmc.values
        i = 0; n = len(flag)
        while i < n:
            if flag[i]:
                j = i
                while j + 1 < n and (flag[j + 1] or (j + 2 < n and flag[j + 2])) and (ts[j + 1] - ts[j]) <= np.timedelta64(10, "m"):
                    j += 1
                nb = j - i + 1
                if nb >= EVENT_MIN_BINS:
                    seg = slice(i, j + 1)
                    excess = np.nansum(np.clip(tt[seg] - ff_tt[d], 0, None))  # minutes per vehicle, summed over bins
                    vals, counts = np.unique(st[seg][st[seg] != ""], return_counts=True)
                    worst = vals[counts.argmax()] if len(vals) else ""
                    events.append({"dir": d, "start": pd.Timestamp(ts[i]).strftime("%Y-%m-%d %H:%M"),
                                   "end": pd.Timestamp(ts[j]).strftime("%Y-%m-%d %H:%M"),
                                   "dur_min": int(nb * 5), "min_speed": round(float(np.nanmin(ms[seg])), 1),
                                   "max_tt": round(float(np.nanmax(tt[seg])), 1),
                                   "excess_veh_min_per_vph": round(float(excess * 5 / 60), 2),  # veh-min of delay per 1 veh/h of flow
                                   "worst_tmc": worst})
                i = j + 1
            else:
                i += 1
    ev = pd.DataFrame(events)

    # ---- per-TMC summaries ------------------------------------------------
    pt = pd.concat(per_tmc_parts)
    tmc_rows = []
    for k, g in pt.groupby("tmc_code", observed=True):
        n = g.n.sum()
        r = tmcs.loc[tmcs.tmc == k].iloc[0]
        wd = g[g.wd]
        prof_h = wd.groupby("hour").apply(lambda x: float((x.speed_mean * x.n).sum() / x.n.sum()), include_groups=False)
        tmc_rows.append({"tmc": k, "dir": r.dir, "order": int(r.road_order), "name": r.intersection, "miles": round(float(r.miles), 3),
                         "lat0": float(r.start_latitude), "lon0": float(r.start_longitude), "lat1": float(r.end_latitude), "lon1": float(r.end_longitude),
                         "ref_speed": round(ref.get(k, float("nan")), 1), "n": int(n),
                         "speed_mean": round(float((g.speed_mean * g.n).sum() / n), 1),
                         "p_lt50": round(float(g.lt50.sum() / n), 4), "p_lt35": round(float(g.lt35.sum() / n), 4),
                         "p_conf30": round(float(g.conf30.sum() / n), 3),
                         "weekday_hour_speed": [round(float(prof_h.get(h, float("nan"))), 1) for h in range(24)]})

    # ---- per-TMC travel-time distributions (segment click panel) ---------
    log("segment distributions")
    dd = pd.concat(dist_parts, ignore_index=True)
    del dist_parts
    dd = dd[dd.tt.notna() & (dd.tt > 0)]
    PCT = [5, 25, 50, 75, 85, 95, 99]
    tmc_dist = {}
    for k, g in dd.groupby("tmc", sort=False):
        ff = float(tmc_miles[k]) / ref[k] * 60.0 if k in ref and ref[k] > 0 else float(g.tt.median())
        tt = g.tt.values.astype("float64")
        q_all = np.percentile(tt, PCT)
        hi = max(float(np.percentile(tt, 99.5)), ff * 2.0)
        lo = max(0.0, ff * 0.7)
        edges = np.linspace(lo, hi, 41)
        hist = {}
        for lab, m in (("all", np.ones(len(g), bool)), ("weekday", g.wd.values), ("weekend", ~g.wd.values)):
            c, _ = np.histogram(np.clip(tt[m], lo, hi - 1e-6), bins=edges)
            hist[lab] = [int(x) for x in c]
        pct = {"all": [round(float(x), 2) for x in q_all]}
        for lab, m in (("weekday", g.wd.values), ("weekend", ~g.wd.values)):
            pct[lab] = [round(float(x), 2) for x in np.percentile(tt[m], PCT)] if m.sum() > 50 else None
        hourly = {}
        for lab, m in (("weekday", g.wd.values), ("weekend", ~g.wd.values)):
            gh = g[m]
            rows = []
            for hh in range(24):
                v = gh.tt.values[gh.hour.values == hh].astype("float64")
                rows.append([round(float(x), 2) for x in np.percentile(v, [5, 25, 50, 75, 95])] if len(v) > 50 else None)
            hourly[lab] = rows
        tmc_dist[k] = {"ff_tt": round(ff, 2), "n": int(len(g)), "pct_labels": PCT, "pct": pct,
                       "edges": [round(float(x), 2) for x in edges], "hist": hist, "hourly": hourly,
                       "p_gt_1_5ff": round(float((tt > 1.5 * ff).mean()), 4), "p_gt_2ff": round(float((tt > 2.0 * ff).mean()), 4),
                       "mean_tt": round(float(tt.mean()), 2), "bti": round(float(q_all[5] / q_all[2] - 1), 3)}
    # ---- 15-minute bins by day type (full-page INRIX map) -------------------
    log("15-minute bins")
    DAYTYPES = [("all", None), ("weekday", lambda d: d < 5), ("weekend", lambda d: d >= 5)] + \
               [(nm, (lambda k: (lambda d: d == k))(i)) for i, nm in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])]
    bins15 = {}
    for k, g in dd.groupby("tmc", sort=False):
        out_k = {}
        tt = g.tt.values.astype("float64"); spd = g.spd.values.astype("float64"); b = g.bin15.values; dow = g.dow.values
        for nm, fn in DAYTYPES:
            m = np.ones(len(g), bool) if fn is None else fn(dow)
            res = {q: [None] * 96 for q in ("p5", "p25", "p50", "p75", "p95", "mean", "spd")}
            res["n"] = [0] * 96
            tb, sb, bb = tt[m], spd[m], b[m]
            order = np.argsort(bb, kind="stable"); tb, sb, bb = tb[order], sb[order], bb[order]
            starts = np.searchsorted(bb, np.arange(97))
            for i in range(96):
                v = tb[starts[i]:starts[i + 1]]
                res["n"][i] = int(len(v))
                if len(v) >= 20:
                    q = np.percentile(v, [5, 25, 50, 75, 95])
                    res["p5"][i], res["p25"][i], res["p50"][i], res["p75"][i], res["p95"][i] = [round(float(x), 2) for x in q]
                    res["mean"][i] = round(float(v.mean()), 2)
                    res["spd"][i] = round(float(sb[starts[i]:starts[i + 1]].mean()), 1)
            out_k[nm] = res
        bins15[k] = out_k
    with open(os.path.join(OUT_DIR, "inrix_bins15.json"), "w", encoding="utf-8") as fh:
        json.dump({"daytypes": [nm for nm, _ in DAYTYPES], "bins": bins15}, fh, separators=(",", ":"))
    log("wrote inrix_bins15.json")
    del dd

    # per TMC x year-month slow share (for the segment heat map)
    tmc_ym = pt.groupby(["tmc_code", "ym"], observed=True).apply(lambda x: float(x.lt50.sum() / x.n.sum()), include_groups=False).reset_index(name="p_lt50")

    summary = {
        "source_folder": os.path.relpath(base, ROOT),
        "files": [os.path.basename(f) for f in files],
        "thresholds": {"slow_mph": SLOW, "very_slow_mph": VERY_SLOW, "event_min_minutes": EVENT_MIN_BINS * 5},
        "dir_miles": {k: round(v, 2) for k, v in dir_miles.items()},
        "ff_tt_min": {k: round(v, 2) for k, v in ff_tt.items()},
        "period": {"start": str(cor.ts.min())[:10], "end": str(cor.ts.max())[:10], "bins": int(len(cor))},
        "tmcs": tmc_rows,
        "tmc_dist": tmc_dist,
        "tmc_ym": tmc_ym.to_dict(orient="records"),
        "daily": daily, "hourly_profile": prof, "monthly": monthly,
        "events": ev.sort_values("excess_veh_min_per_vph", ascending=False).head(400).to_dict(orient="records") if len(ev) else [],
        "event_counts": ev.groupby("dir").size().to_dict() if len(ev) else {},
        # all-episode aggregates (the "events" list above keeps only the 400 largest)
        "worst_tmc_counts": ev.groupby("worst_tmc").size().to_dict() if len(ev) else {},
        "event_hour_counts": {d: [int(x) for x in pd.to_datetime(g["start"]).dt.hour.value_counts().reindex(range(24), fill_value=0)]
                              for d, g in ev.groupby("dir")} if len(ev) else {},
        "event_days": {d: g.assign(day=g["start"].str[:10]).groupby("day").size().to_dict() for d, g in ev.groupby("dir")} if len(ev) else {},
        "event_stats": {d: {"n": int(len(g)), "n_per_year": round(len(g) / max(1, cor.ts.dt.year.nunique()), 1),
                            "median_dur_min": float(g.dur_min.median()), "mean_dur_min": round(float(g.dur_min.mean()), 1),
                            "total_excess_veh_min_per_vph": round(float(g.excess_veh_min_per_vph.sum()), 1)}
                        for d, g in ev.groupby("dir")} if len(ev) else {},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, separators=(",", ":"))
    log(f"wrote {OUT_JSON} ({os.path.getsize(OUT_JSON)/1024:,.0f} KB); events: {summary['event_counts']}")


if __name__ == "__main__":
    main()
