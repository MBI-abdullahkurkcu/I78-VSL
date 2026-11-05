# I-78 Variable Speed Limit (VSL) study

Working repository for the I-78 VSL concept-of-operations work in western New Jersey
(PA-NJ state border to Exit 12 / Perryville, Warren and Hunterdon counties).

## What is here

| Path | Content |
|---|---|
| `I78_VSL_Dashboard.html` | Standalone dashboard (open in any browser). Tabs: (1) data explorer, (2) map with every dataset and a straight-line diagram, plus a "Map view" switch that embeds the full-page INRIX map (`I78_INRIX_Map.html` is inlined at build time and loaded into an iframe on demand), (3) correlations and statistical tests, (4) key scenarios, (5) potential impact, (6) method. The dataset chips in the header open a detail panel with the source files (size, rows, columns), coverage and key statistics. Leaflet is inlined, so only the basemap tiles need the network (Esri by default; OpenStreetMap tiles show "access blocked" for pages opened from a local file). |
| `I78_INRIX_Map.html` | Full-page INRIX map (built from `dashboard/inrix_map_template.html`, modelled on the Traffic-Scripts `inrix_tt_map` page): segments colored by travel-time index, planning-time index, speed, travel time per mile or whole-period slowdown shares; day-type, direction and 15-minute time-of-day slider with play; legend with editable class limits and click-to-hide classes; click a segment for its time-of-day percentile bands, travel-time distribution and worst episodes. State is kept in the URL hash. The same page is embedded in tab 2 of the dashboard; the embedded copy follows the dashboard theme and does not write the URL hash. |
| `scripts/fetch_geometry.py` | Road-following geometry for the 24 INRIX segments from the public OSRM router (same method as the Santa Fe corridor map), cached per TMC in `dashboard/data/osrm_cache/` and written to `dashboard/data/tmc_geometry.json`. Run once, or again when TMCs change. |
| `scripts/stats_tests.py` | Statistical tests on a daily panel (rate ratios with CIs, quasi-Poisson regression, Kruskal-Wallis / Mann-Whitney, Spearman matrix, dose-response bins, lags, monthly). Needs scipy. Called by the builder. |
| `dashboard/vendor/` | Leaflet 1.9.4 JS and CSS (BSD-2-Clause), inlined into the page at build time. |
| `Input Data/Climate*` | NOAA GHCN-Daily pull, 17 stations within 8 mi of the corridor, 2021-2025. |
| `Input Data/Crash_*` | NJDOT crash tables (Crash, Vehicle, Unit, Person, ...) for the I-78 municipalities. Only route-78 rows are used; no person-level fields reach the dashboard. |
| `Input Data/INRIX_*` | INRIX TMC 5-minute speeds, EB and WB, 2021-2025 (raw files are git-ignored because of size). |
| `config/corridor.json` | Corridor definition: centerline, interchanges, VSL zones, speed floors. Edit here. |
| `scripts/preprocess_inrix.py` | Aggregates the 12.6 M INRIX rows into `dashboard/data/inrix_summary.json`. Slow (about 1 min); run only when INRIX files change. |
| `scripts/build_dashboard.py` | Loads every dataset, computes the summaries and writes the dashboard. Fast. |
| `dashboard/template.html` | Dashboard source (HTML/CSS/JS). The build script injects the data blob. |
| `Ref/` | Colorado DOT Variable Speed Limits Guidelines (Oct 2023), the method reference. |

## Rebuild

```
python -I scripts/fetch_geometry.py     # once, or when the TMC list changes (needs internet)
python -I scripts/preprocess_inrix.py   # only when new INRIX files arrive
python -I scripts/build_dashboard.py    # always
```

Requires Python 3.12 with pandas and numpy.

## Adding a dataset

1. Drop the files under `Input Data/<name>/`.
2. Add a loader in `scripts/build_dashboard.py` that returns a dict with `"status": "loaded"` plus the summaries.
3. Add the matching view in `dashboard/template.html` and rebuild.

## Known gaps

- Crash extract covers only Greenwich Twp and Bethlehem Twp (about MP 4 to 11); Pohatcong Twp and Union Twp are still needed.
- No visibility / fog data (needs hourly ASOS for KABE or RWIS).
- No work-zone log (crashes coded "Construction Zone" are the interim proxy).
- No traffic volumes (needed for vphpl and vehicle-hours of delay).
