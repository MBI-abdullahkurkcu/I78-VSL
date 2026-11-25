"""Resize the sign/transformer icons in icons/ to 32x32 PNGs and write them to icons/sized/.

Each kind in `KINDS` below is matched to an asset kind slug used by scripts/parse_kmz.py.
The resized PNG is kept in the repo (icons/sized/<kind>.png) and the builder reads it
and embeds it as a base64 data URI in the generated HTML, so the maps don't need
network or file-protocol access to the icon files.

Run after a new icon is dropped in icons/; the builder reads from icons/sized/.
"""
import os, sys, base64, json
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "icons")
OUT = os.path.join(ROOT, "icons", "sized")
os.makedirs(OUT, exist_ok=True)

# source filename -> (kind slug, output name, target px)
KINDS = [
    ("ramp-exit.png",                "ramp_exit",   32),
    ("static-speed-limit-sign.jpg",  "speed_limit", 32),
    ("transformer.png",              "transformer", 32),
    ("dynamic-message-sign-dms.png", "dms",         48),  # larger: the DMS board was unreadable at 32
    ("CSS.png",                      "css",         32),
    ("VSL-sign.png",                 "vsl_sign",    48),  # larger: VSL sign text is hard to read at 32
]

def resize_square(path, px):
    im = Image.open(path).convert("RGBA")
    # scale to fit a px-by-px square, preserving aspect ratio
    w, h = im.size
    r = min(px / w, px / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.paste(im, ((px - nw) // 2, (px - nh) // 2), im)
    return canvas

def main():
    index = {}
    for fname, kind, px in KINDS:
        src = os.path.join(SRC, fname)
        if not os.path.exists(src):
            print(f"  skip: {fname} not found")
            continue
        im = resize_square(src, px)
        dst = os.path.join(OUT, kind + ".png")
        im.save(dst, "PNG", optimize=True)
        b = open(dst, "rb").read()
        index[kind] = {"file": os.path.relpath(dst, ROOT).replace("\\", "/"),
                       "bytes": len(b), "px": px}
        print(f"  {kind:14s} {len(b):>6,d} B  ({fname})")
    # tiny manifest so the builder knows which kinds have icons
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"wrote {len(index)} sized icons + manifest")

if __name__ == "__main__":
    main()
