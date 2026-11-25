"""Parse the newest VSL/ITS KMZ in dashboard/data/KMZs/ into dashboard/data/vsl_assets.json.

The dashboard reads the JSON at build time; re-run this whenever a new KMZ is dropped in.
Output schema:
  {"meta": {...}, "folders": [{"name": "...", "kind": "...", "features": [{name, lat, lon}|{name, coords:[[lat,lon],...]}]}]}
kind is a slug the dashboard maps to a marker shape and color.
"""
import os, sys, json, zipfile, glob, re
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KMZ_DIR = os.path.join(ROOT, "dashboard", "data", "KMZs")
OUT = os.path.join(ROOT, "dashboard", "data", "vsl_assets.json")

# newest KMZ wins; filename suffix "_V\d+" or mtime
def pick_kmz():
    files = sorted(glob.glob(os.path.join(KMZ_DIR, "*.kmz")))
    if not files:
        sys.exit("no KMZ in " + KMZ_DIR)
    def key(p):
        m = re.search(r"_V(\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else -1, os.path.getmtime(p), os.path.basename(p))
    return max(files, key=key)

# folder name -> kind slug (keeps the schema stable even if a later KMZ renames things)
FOLDER_KIND = {
    "project limits": "zone_limit",
    "vsl": "vsl_sign",
    "speed limit signs": "speed_limit",
    "ramp exit signs": "ramp_exit",
    "transformers": "transformer",
    "ex css": "css",
    "ex foc": "foc",
    "ex dms": "dms",
}

NS = {"k": "http://www.opengis.net/kml/2.2"}

def strip_ns(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

def parse_coords(text):
    pts = []
    for tok in text.strip().split():
        parts = tok.split(",")
        if len(parts) < 2: continue
        lon, lat = float(parts[0]), float(parts[1])
        pts.append([round(lat, 6), round(lon, 6)])
    return pts

def iter_children(elem, local):
    for c in list(elem):
        if strip_ns(c.tag) == local:
            yield c

def placemark_geom(pm):
    """Return ('point', [lat,lon]) or ('line', [[lat,lon],...]) or None."""
    for geom in pm.iter():
        tag = strip_ns(geom.tag)
        if tag == "Point":
            for c in geom:
                if strip_ns(c.tag) == "coordinates":
                    pts = parse_coords(c.text or "")
                    if pts: return ("point", pts[0])
        elif tag in ("LineString", "LinearRing"):
            for c in geom:
                if strip_ns(c.tag) == "coordinates":
                    pts = parse_coords(c.text or "")
                    if pts: return ("line", pts)
    return None

def walk_folders(doc_root):
    """Walk Document > Folder (possibly nested one level) and yield (folder_name, placemarks)."""
    # top-level document
    document = None
    for c in doc_root:
        if strip_ns(c.tag) == "Document":
            document = c; break
    if document is None: return
    # One wrapping folder "I-78" is expected; descend into any nested folders, else take top folder as is.
    def emit(folder):
        pms = [c for c in folder if strip_ns(c.tag) == "Placemark"]
        if pms:
            name = None
            for c in folder:
                if strip_ns(c.tag) == "name": name = (c.text or "").strip()
            yield (name or "Unnamed", pms)
        for sub in folder:
            if strip_ns(sub.tag) == "Folder":
                yield from emit(sub)
    for c in document:
        if strip_ns(c.tag) == "Folder":
            yield from emit(c)

def main():
    src = pick_kmz()
    with zipfile.ZipFile(src) as z:
        kml = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if not kml: sys.exit("no KML inside " + src)
        text = z.read(kml).decode("utf-8", "replace")
    root = ET.fromstring(text)

    folders_out = []
    seen_kinds = {}
    for name, pms in walk_folders(root):
        kind = FOLDER_KIND.get(name.strip().lower())
        if not kind:
            # skip unknown folders silently but report them
            print(f"  note: skipping unknown folder {name!r}")
            continue
        feats = []
        for pm in pms:
            n = ""
            for c in pm:
                if strip_ns(c.tag) == "name":
                    n = (c.text or "").strip()
            # Skip Google Earth's default "Untitled Placemark" entries — they carry no identifying label.
            if n.lower() == "untitled placemark":
                continue
            g = placemark_geom(pm)
            if not g: continue
            typ, data = g
            if typ == "point":
                feats.append({"name": n, "lat": data[0], "lon": data[1]})
            else:
                feats.append({"name": n, "coords": data})
        folders_out.append({"name": name, "kind": kind, "features": feats})
        seen_kinds[kind] = seen_kinds.get(kind, 0) + len(feats)

    out = {
        "meta": {"source": os.path.relpath(src, ROOT).replace("\\", "/"),
                 "folders": len(folders_out),
                 "features": sum(len(f["features"]) for f in folders_out),
                 "by_kind": seen_kinds},
        "folders": folders_out,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT)/1024:,.1f} KB)")
    for f in folders_out:
        print(f"  {f['kind']:14s} {len(f['features']):4d}  ({f['name']})")

if __name__ == "__main__":
    main()
