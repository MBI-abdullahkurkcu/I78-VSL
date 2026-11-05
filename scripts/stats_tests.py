"""Statistical tests for the I-78 VSL dashboard (Tab 2, "Statistical correlations").

Everything works on a daily panel joining the climate composite, the crash
extract and the INRIX daily corridor summaries.  Only numpy, pandas and scipy
are used; the Poisson regression is a small IRLS implementation so no
statsmodels dependency is needed.

Outputs are plain dicts/lists so build_dashboard.py can drop them into the JSON
blob unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

CLS_KEYS = ["dry", "wet", "heavy", "freezing", "snow_lt", "snow_sig", "snow_maj"]


def _r(x, n=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
    except TypeError:
        pass
    return round(float(x), n)


def _p(x):
    """p-values keep more precision than other figures (they can be tiny)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(f"{x:.3g}")


# --------------------------------------------------------------------------
# Daily panel
# --------------------------------------------------------------------------
def build_panel(climate: dict, crashes: dict | None, inrix: dict | None) -> pd.DataFrame:
    d = pd.DataFrame(climate["daily"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    d["freeze"] = (d["tmin"] <= 32).astype(int)
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["year"] = d.index.year
    d["ym"] = d.index.strftime("%Y-%m")
    for c in ["n", "inj", "truck", "adv", "same_dir", "wz"]:
        d["cr_" + c] = 0
    if crashes and crashes.get("status") == "loaded" and crashes.get("daily"):
        cd = pd.DataFrame(crashes["daily"])
        cd["date"] = pd.to_datetime(cd["date"])
        cd = cd.set_index("date")
        for c in ["n", "inj", "truck", "adv", "same_dir", "wz"]:
            if c in cd:
                d["cr_" + c] = cd[c].reindex(d.index).fillna(0).astype(int)
    for dir_ in "EW":
        for c in ["speed", "h_any_lt35", "h_lt50", "tt95", "min_speed", "tt_mean"]:
            d[f"{c}_{dir_}"] = np.nan
        d[f"ev_{dir_}"] = 0
    if inrix and inrix.get("status") == "loaded":
        ix = pd.DataFrame(inrix["daily"])
        ix["date"] = pd.to_datetime(ix["date"])
        for dir_, g in ix.groupby("dir"):
            g = g.set_index("date")
            for c in ["speed", "h_any_lt35", "h_lt50", "tt95", "min_speed", "tt_mean"]:
                d[f"{c}_{dir_}"] = g[c].reindex(d.index)
        for dir_, days in (inrix.get("event_days") or {}).items():
            ev = pd.Series(days)
            ev.index = pd.to_datetime(ev.index)
            d[f"ev_{dir_}"] = ev.reindex(d.index).fillna(0).astype(int)
    d["ev"] = d["ev_E"] + d["ev_W"]
    d["h35"] = d[["h_any_lt35_E", "h_any_lt35_W"]].sum(axis=1, min_count=1)
    d["speed"] = d[["speed_E", "speed_W"]].mean(axis=1)
    return d


# --------------------------------------------------------------------------
# Rate ratios with confidence intervals
# --------------------------------------------------------------------------
def _poisson_ci(n, days):
    """Exact (Garwood) 95% CI for a Poisson rate n / days."""
    if days <= 0:
        return None, None
    lo = stats.chi2.ppf(0.025, 2 * n) / 2 / days if n > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2 / days
    return lo, hi


def rate_ratios(panel: pd.DataFrame, outcome: str = "cr_n") -> dict:
    rows = []
    dry = panel[panel["cls"] == 0]
    n0, d0 = int(dry[outcome].sum()), int(len(dry))
    for k, key in enumerate(CLS_KEYS):
        g = panel[panel["cls"] == k]
        n1, d1 = int(g[outcome].sum()), int(len(g))
        rate = n1 / d1 if d1 else None
        lo_r, hi_r = _poisson_ci(n1, d1)
        if n1 > 0 and n0 > 0 and d1 > 0:
            rr = (n1 / d1) / (n0 / d0)
            se = math.sqrt(1 / n1 + 1 / n0)
            lo, hi = math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)
            z = math.log(rr) / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            rr = lo = hi = p = None
        rows.append({"cls": k, "key": key, "days": d1, "n": n1, "rate_per_100d": _r(100 * rate, 2) if rate is not None else None,
                     "rate_lo": _r(100 * lo_r, 2) if lo_r is not None else None, "rate_hi": _r(100 * hi_r, 2) if hi_r is not None else None,
                     "rr": _r(rr, 3), "rr_lo": _r(lo, 3), "rr_hi": _r(hi, 3), "p": _p(p)})
    # chi-square goodness of fit: do crashes distribute across classes in proportion to days?
    obs = np.array([r["n"] for r in rows], dtype=float)
    days = np.array([r["days"] for r in rows], dtype=float)
    keep = days > 0
    exp = obs[keep].sum() * days[keep] / days[keep].sum()
    chi2, p = stats.chisquare(obs[keep], exp)
    # adverse (any non-dry) vs dry pooled
    adv = panel[panel["cls"] > 0]
    n1, d1 = int(adv[outcome].sum()), int(len(adv))
    rr = (n1 / d1) / (n0 / d0) if n1 and n0 else None
    se = math.sqrt(1 / n1 + 1 / n0) if n1 and n0 else None
    pooled = {"days": d1, "n": n1, "rr": _r(rr, 3), "rr_lo": _r(math.exp(math.log(rr) - 1.96 * se), 3) if rr else None,
              "rr_hi": _r(math.exp(math.log(rr) + 1.96 * se), 3) if rr else None,
              "p": _p(2 * (1 - stats.norm.cdf(abs(math.log(rr) / se)))) if rr else None}
    return {"outcome": outcome, "rows": rows, "chi2": _r(chi2, 2), "chi2_dof": int(keep.sum() - 1), "chi2_p": _p(p), "pooled_adverse": pooled}


# --------------------------------------------------------------------------
# Poisson regression (IRLS) with quasi-Poisson scaling
# --------------------------------------------------------------------------
def poisson_glm(y: np.ndarray, X: np.ndarray, names: list[str], max_iter=50, tol=1e-8) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = math.log(max(y.mean(), 1e-6))
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta + (y - mu) / mu
        W = mu
        XtW = X.T * W
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = np.exp(np.clip(eta, -30, 30))
    cov = np.linalg.pinv((X.T * mu) @ X)
    pearson = float(np.sum((y - mu) ** 2 / mu))
    df = n - p
    disp = pearson / df if df > 0 else 1.0
    scale = math.sqrt(max(disp, 1.0))  # quasi-Poisson: inflate SE when over-dispersed, never deflate
    se = np.sqrt(np.diag(cov)) * scale
    z = beta / se
    pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
    # null deviance / deviance for a pseudo-R2
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.nansum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
        mu0 = y.mean()
        dev0 = 2 * np.nansum(np.where(y > 0, y * np.log(y / mu0), 0) - (y - mu0))
    terms = [{"term": nm, "coef": _r(b, 4), "rr": _r(math.exp(b), 3), "lo": _r(math.exp(b - 1.96 * s), 3), "hi": _r(math.exp(b + 1.96 * s), 3),
              "se": _r(s, 4), "z": _r(zz, 2), "p": _p(pp)} for nm, b, s, zz, pp in zip(names, beta, se, z, pv)]
    return {"n": int(n), "params": int(p), "dispersion": _r(disp, 3), "deviance": _r(dev, 1), "null_deviance": _r(dev0, 1),
            "pseudo_r2": _r(1 - dev / dev0, 4) if dev0 else None, "terms": terms}


def _design(panel: pd.DataFrame, cls_labels: list[str], extra: list[str] | None = None):
    cols, names = [np.ones(len(panel))], ["Intercept"]
    for k in range(1, 7):
        cols.append((panel["cls"] == k).astype(float).values)
        names.append("Weather: " + cls_labels[k])
    for dow, lab in enumerate(["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], start=1):
        cols.append((panel["dow"] == dow).astype(float).values)
        names.append("Weekday: " + lab + " (vs Mon)")
    for m in range(2, 13):
        cols.append((panel["month"] == m).astype(float).values)
        names.append("Month: " + ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m] + " (vs Jan)")
    cols.append((panel["year"] - panel["year"].min()).astype(float).values)
    names.append("Year trend (per year)")
    for e in extra or []:
        cols.append(panel[e].fillna(0).astype(float).values)
        names.append(e)
    return np.column_stack(cols), names


def adjusted_models(panel: pd.DataFrame, cls_labels: list[str]) -> dict:
    out = {}
    X, names = _design(panel, cls_labels)
    for key, outcome, label in (("all", "cr_n", "All mainline crashes per day"), ("injury", "cr_inj", "Injury crashes per day"),
                                ("truck", "cr_truck", "Truck-involved crashes per day")):
        if panel[outcome].sum() == 0:
            continue
        m = poisson_glm(panel[outcome].values, X, names)
        m["label"] = label
        out[key] = m
    # crashes vs continuous weather drivers (per inch / per 10 degF), same controls
    panel = panel.copy()
    panel["prcp_in"] = panel["prcp_mean"].fillna(0)
    panel["snow_in"] = panel["snow_max"].fillna(0)
    panel["snwd_in"] = panel["snwd_max"].fillna(0)
    panel["tmin_10f"] = (panel["tmin"].fillna(panel["tmin"].mean()) - 32) / 10.0
    cols, names2 = [np.ones(len(panel))], ["Intercept"]
    for dow, lab in enumerate(["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], start=1):
        cols.append((panel["dow"] == dow).astype(float).values); names2.append("Weekday: " + lab + " (vs Mon)")
    for m in range(2, 13):
        cols.append((panel["month"] == m).astype(float).values)
        names2.append("Month: " + ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m] + " (vs Jan)")
    cols.append((panel["year"] - panel["year"].min()).astype(float).values); names2.append("Year trend (per year)")
    for c, nm in (("prcp_in", "Precipitation, area mean (per inch)"), ("snow_in", "Snowfall, station max (per inch)"),
                  ("snwd_in", "Snow depth, station max (per inch)"), ("tmin_10f", "Minimum temperature (per 10 degF above 32)")):
        cols.append(panel[c].values); names2.append(nm)
    Xc = np.column_stack(cols)
    m = poisson_glm(panel["cr_n"].values, Xc, names2)
    m["label"] = "All crashes per day on continuous weather drivers"
    out["continuous"] = m
    return out


# --------------------------------------------------------------------------
# Speed by weather class: non-parametric tests
# --------------------------------------------------------------------------
def _cliffs_delta(a, b):
    """Probability that a random a exceeds a random b minus the reverse."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return None
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def speed_tests(panel: pd.DataFrame, cls_labels: list[str]) -> dict:
    out = {}
    for dir_ in "EW":
        col = f"speed_{dir_}"
        d = panel[panel[col].notna()]
        groups = [d.loc[d["cls"] == k, col].values for k in range(7)]
        valid = [g for g in groups if len(g) >= 2]
        kw = stats.kruskal(*valid) if len(valid) >= 2 else None
        dry = groups[0]
        rows = []
        for k in range(7):
            g = groups[k]
            if len(g) == 0:
                continue
            row = {"cls": k, "label": cls_labels[k], "days": int(len(g)), "median": _r(np.median(g), 1), "mean": _r(np.mean(g), 1),
                   "p10": _r(np.percentile(g, 10), 1), "p90": _r(np.percentile(g, 90), 1)}
            if k > 0 and len(g) >= 2 and len(dry) >= 2:
                mw = stats.mannwhitneyu(g, dry, alternative="two-sided")
                tt = stats.ttest_ind(g, dry, equal_var=False)
                row.update({"diff_median": _r(np.median(g) - np.median(dry), 1), "diff_mean": _r(np.mean(g) - np.mean(dry), 1),
                            "mw_p": _p(mw.pvalue), "welch_p": _p(tt.pvalue), "cliffs_delta": _r(_cliffs_delta(g, dry), 3)})
            rows.append(row)
        # slowdown exposure (hours any segment < 35) by class: same tests
        hcol = f"h_any_lt35_{dir_}"
        hd = panel[panel[hcol].notna()]
        hgroups = [hd.loc[hd["cls"] == k, hcol].values for k in range(7)]
        hrows = []
        for k in range(7):
            g = hgroups[k]
            if len(g) == 0:
                continue
            row = {"cls": k, "label": cls_labels[k], "days": int(len(g)), "mean_h": _r(np.mean(g), 2), "p_days_any": _r(np.mean(g > 0), 3)}
            if k > 0 and len(g) >= 2:
                mw = stats.mannwhitneyu(g, hgroups[0], alternative="two-sided")
                # share of days with any slowdown: two-proportion z via chi2
                a, b = int((g > 0).sum()), int((hgroups[0] > 0).sum())
                tbl = np.array([[a, len(g) - a], [b, len(hgroups[0]) - b]])
                chi = stats.chi2_contingency(tbl, correction=False)
                row.update({"mw_p": _p(mw.pvalue), "prop_p": _p(chi[1]), "ratio_mean_h": _r(np.mean(g) / np.mean(hgroups[0]), 2) if np.mean(hgroups[0]) > 0 else None})
            hrows.append(row)
        out[dir_] = {"kruskal_H": _r(kw.statistic, 2) if kw else None, "kruskal_p": _p(kw.pvalue) if kw else None, "rows": rows, "slowdown_rows": hrows}
    return out


# --------------------------------------------------------------------------
# Correlation matrix
# --------------------------------------------------------------------------
PANEL_VARS = [
    ("prcp_mean", "Precip (area mean, in)"), ("prcp_max", "Precip (station max, in)"), ("snow_max", "Snowfall (in)"),
    ("snwd_max", "Snow depth (in)"), ("tmax", "Max temp (F)"), ("tmin", "Min temp (F)"),
    ("cr_n", "Crashes / day"), ("cr_inj", "Injury crashes / day"), ("cr_truck", "Truck crashes / day"), ("cr_adv", "Adverse-coded crashes / day"),
    ("speed_E", "EB mean speed"), ("speed_W", "WB mean speed"), ("h_any_lt35_E", "EB hours < 35 mph"), ("h_any_lt35_W", "WB hours < 35 mph"),
    ("tt95_E", "EB 95th pct travel time"), ("tt95_W", "WB 95th pct travel time"), ("ev", "Slowdown episodes / day"),
]


def correlation_matrix(panel: pd.DataFrame) -> dict:
    keys = [k for k, _ in PANEL_VARS if k in panel and panel[k].notna().sum() > 30 and panel[k].nunique() > 1]
    labels = [dict(PANEL_VARS)[k] for k in keys]
    n = len(keys)
    sp = [[None] * n for _ in range(n)]
    spp = [[None] * n for _ in range(n)]
    pe = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j < i:
                sp[i][j], spp[i][j], pe[i][j] = sp[j][i], spp[j][i], pe[j][i]
                continue
            a, b = panel[keys[i]], panel[keys[j]]
            m = a.notna() & b.notna()
            if m.sum() < 30:
                continue
            if i == j:
                sp[i][j], spp[i][j], pe[i][j] = 1.0, 0.0, 1.0
                continue
            rs = stats.spearmanr(a[m], b[m])
            rp = stats.pearsonr(a[m], b[m])
            sp[i][j], spp[i][j], pe[i][j] = _r(rs.statistic, 3), _p(rs.pvalue), _r(rp.statistic, 3)
    return {"keys": keys, "labels": labels, "spearman": sp, "spearman_p": spp, "pearson": pe, "n_days": int(len(panel))}


# --------------------------------------------------------------------------
# Binned dose-response
# --------------------------------------------------------------------------
def _bin_table(panel: pd.DataFrame, col: str, edges: list[float], labels: list[str]) -> list[dict]:
    v = panel[col]
    rows = []
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        m = (v >= lo) & (v < hi) if i < len(labels) - 1 else (v >= lo)
        if i == 0:
            m = (v < hi) | v.isna() if lo is None else m
        g = panel[m]
        d = int(len(g))
        n = int(g["cr_n"].sum())
        lo_r, hi_r = _poisson_ci(n, d)
        rows.append({"bin": lab, "days": d, "crashes": n, "injury": int(g["cr_inj"].sum()), "truck": int(g["cr_truck"].sum()),
                     "rate_per_100d": _r(100 * n / d, 2) if d else None, "rate_lo": _r(100 * lo_r, 2) if d else None, "rate_hi": _r(100 * hi_r, 2) if d else None,
                     "speed_E": _r(g["speed_E"].mean(), 1), "speed_W": _r(g["speed_W"].mean(), 1),
                     "h35_E": _r(g["h_any_lt35_E"].mean(), 2), "h35_W": _r(g["h_any_lt35_W"].mean(), 2), "ev_per_day": _r(g["ev"].mean(), 2)})
    return rows


def dose_response(panel: pd.DataFrame) -> dict:
    out = {}
    out["precip"] = {"var": "Daily precipitation, area mean (in)",
                     "rows": _bin_table(panel, "prcp_mean", [0, 0.01, 0.10, 0.25, 0.50, 1.0, 99], ["0 (dry)", "0.01-0.10", "0.10-0.25", "0.25-0.50", "0.50-1.0", "1.0+"])}
    out["snow"] = {"var": "Daily snowfall, station max (in)",
                   "rows": _bin_table(panel, "snow_max", [0, 0.1, 1, 3, 6, 99], ["0", "0.1-1", "1-3", "3-6", "6+"])}
    out["snwd"] = {"var": "Snow depth on the ground, station max (in)",
                   "rows": _bin_table(panel, "snwd_max", [0, 0.5, 2, 6, 99], ["0", "0.5-2", "2-6", "6+"])}
    out["tmin"] = {"var": "Minimum temperature (F)",
                   "rows": _bin_table(panel, "tmin", [-99, 20, 28, 32.01, 40, 50, 999], ["< 20", "20-27", "28-32", "33-39", "40-49", "50+"])}
    # Spearman on the continuous driver, all days and conditional on any precip
    sp = {}
    for k, col in (("precip_all", "prcp_mean"), ("snow_all", "snow_max"), ("tmin_all", "tmin")):
        m = panel[col].notna()
        r = stats.spearmanr(panel.loc[m, col], panel.loc[m, "cr_n"])
        sp[k] = {"r": _r(r.statistic, 3), "p": _p(r.pvalue), "n": int(m.sum())}
    m = panel["prcp_mean"].fillna(0) > 0
    r = stats.spearmanr(panel.loc[m, "prcp_mean"], panel.loc[m, "cr_n"])
    sp["precip_wet_days_only"] = {"r": _r(r.statistic, 3), "p": _p(r.pvalue), "n": int(m.sum())}
    out["spearman"] = sp
    return out


# --------------------------------------------------------------------------
# Lags, crash vs delay, monthly aggregates
# --------------------------------------------------------------------------
def lag_tests(panel: pd.DataFrame) -> list[dict]:
    rows = []
    for col, lab in (("prcp_mean", "Precipitation"), ("snow_max", "Snowfall"), ("snwd_max", "Snow depth"), ("freeze", "Freeze night (Tmin <= 32)")):
        for lag in (0, 1, 2):
            x = panel[col].shift(lag)
            m = x.notna()
            r = stats.spearmanr(x[m], panel.loc[m, "cr_n"])
            rows.append({"driver": lab, "lag_days": lag, "r": _r(r.statistic, 3), "p": _p(r.pvalue), "n": int(m.sum())})
    return rows


def crash_vs_delay(panel: pd.DataFrame) -> dict:
    out = {}
    for dir_ in "EW":
        col = f"h_any_lt35_{dir_}"
        d = panel[panel[col].notna()]
        r = stats.spearmanr(d["cr_n"], d[col])
        a, b = d.loc[d["cr_n"] > 0, col], d.loc[d["cr_n"] == 0, col]
        mw = stats.mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else None
        ev = d[f"ev_{dir_}"]
        ea, eb = ev[d["cr_n"] > 0], ev[d["cr_n"] == 0]
        mwe = stats.mannwhitneyu(ea, eb, alternative="two-sided") if len(ea) and len(eb) else None
        # share of crash days with at least one episode vs non-crash days
        tbl = np.array([[int((ea > 0).sum()), int((ea == 0).sum())], [int((eb > 0).sum()), int((eb == 0).sum())]])
        chi = stats.chi2_contingency(tbl, correction=False) if tbl.min() >= 0 and tbl.sum() else None
        out[dir_] = {"spearman_r": _r(r.statistic, 3), "spearman_p": _p(r.pvalue), "n": int(len(d)),
                     "h35_crash_days": _r(a.mean(), 2), "h35_no_crash_days": _r(b.mean(), 2), "h35_mw_p": _p(mw.pvalue) if mw else None,
                     "ev_crash_days": _r(ea.mean(), 2), "ev_no_crash_days": _r(eb.mean(), 2), "ev_mw_p": _p(mwe.pvalue) if mwe else None,
                     "p_episode_crash_days": _r((ea > 0).mean(), 3), "p_episode_no_crash_days": _r((eb > 0).mean(), 3), "p_episode_chi2_p": _p(chi[1]) if chi else None,
                     "crash_days": int(len(a)), "no_crash_days": int(len(b))}
    # same-day: crashes vs corridor mean speed and 95th pct travel time
    for dir_ in "EW":
        d = panel[panel[f"speed_{dir_}"].notna()]
        r1 = stats.spearmanr(d["cr_n"], d[f"speed_{dir_}"])
        r2 = stats.spearmanr(d["cr_n"], d[f"tt95_{dir_}"])
        out[dir_]["speed_r"], out[dir_]["speed_p"] = _r(r1.statistic, 3), _p(r1.pvalue)
        out[dir_]["tt95_r"], out[dir_]["tt95_p"] = _r(r2.statistic, 3), _p(r2.pvalue)
    return out


def monthly_tests(panel: pd.DataFrame) -> dict:
    g = panel.groupby("ym").agg(crashes=("cr_n", "sum"), injury=("cr_inj", "sum"), adverse_days=("cls", lambda s: int((s > 0).sum())),
                                snow_days=("cls", lambda s: int((s >= 4).sum())), prcp=("prcp_mean", "sum"), snow=("snow_max", "sum"),
                                freeze_days=("freeze", "sum"), episodes=("ev", "sum"), h35=("h35", "sum"), speed=("speed", "mean"), days=("cr_n", "size"))
    g["crash_rate"] = g["crashes"] / g["days"] * 30
    rows = []
    for x, lab in (("adverse_days", "Adverse days"), ("snow_days", "Snow days"), ("prcp", "Total precipitation"), ("snow", "Total snowfall"),
                   ("freeze_days", "Freeze nights"), ("episodes", "Slowdown episodes"), ("h35", "Hours any segment < 35 mph"), ("speed", "Mean corridor speed")):
        m = g[x].notna() & g["crash_rate"].notna()
        if m.sum() < 6:
            continue
        rp = stats.pearsonr(g.loc[m, x], g.loc[m, "crash_rate"])
        rs = stats.spearmanr(g.loc[m, x], g.loc[m, "crash_rate"])
        rows.append({"driver": lab, "pearson_r": _r(rp.statistic, 3), "pearson_p": _p(rp.pvalue), "spearman_r": _r(rs.statistic, 3), "spearman_p": _p(rs.pvalue), "n_months": int(m.sum())})
    series = [{"ym": ym, "crashes": int(r.crashes), "crash_rate": _r(r.crash_rate, 2), "adverse_days": int(r.adverse_days), "snow": _r(r.snow, 1),
               "prcp": _r(r.prcp, 2), "episodes": int(r.episodes), "h35": _r(r.h35, 1), "speed": _r(r.speed, 1)} for ym, r in g.iterrows()]
    return {"rows": rows, "series": series}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_all(climate: dict, crashes: dict | None, inrix: dict | None, cls_labels: list[str]) -> dict:
    panel = build_panel(climate, crashes, inrix)
    has_cr = bool(crashes and crashes.get("status") == "loaded" and panel["cr_n"].sum() > 0)
    has_ix = bool(inrix and inrix.get("status") == "loaded" and panel["speed_E"].notna().sum() > 30)
    out = {"panel_days": int(len(panel)), "has_crashes": has_cr, "has_inrix": has_ix,
           "crash_days": int((panel["cr_n"] > 0).sum()) if has_cr else 0,
           "crash_per_day_mean": _r(panel["cr_n"].mean(), 3) if has_cr else None,
           "crash_per_day_var": _r(panel["cr_n"].var(), 3) if has_cr else None}
    if has_cr:
        out["rate_ratios"] = {"all": rate_ratios(panel, "cr_n"), "injury": rate_ratios(panel, "cr_inj"), "truck": rate_ratios(panel, "cr_truck")}
        out["models"] = adjusted_models(panel, cls_labels)
        out["dose"] = dose_response(panel)
        out["lags"] = lag_tests(panel)
    if has_ix:
        out["speed_tests"] = speed_tests(panel, cls_labels)
    if has_cr and has_ix:
        out["crash_vs_delay"] = crash_vs_delay(panel)
    out["corr"] = correlation_matrix(panel)
    out["monthly"] = monthly_tests(panel)
    return out
