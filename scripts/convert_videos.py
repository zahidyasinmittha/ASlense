# fix_dataset_encoding.py
"""
Reads the 'videos' column from a TSV/CSV, checks each MP4 in E:\final_dataset_videos,
converts non‑web‑safe files to H.264/AAC, and overwrites the originals.

Run:  python fix_dataset_encoding.py
"""

import csv, json, subprocess, tempfile, os
from pathlib import Path

# ── configuration ──────────────────────────────────────────────────────────
CSV_PATH  = Path(r"sample_videos.csv")                 # your TSV file (tab‑separated)
DATA_DIR  = Path(r"E:\final_dataset_videos")   # folder with the videos
FFMPEG    = "ffmpeg"                           # must be in PATH
FFPROBE   = "ffprobe"                          # must be in PATH
DELIM     = ","                               # delimiter used in your file
# ───────────────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    "Return container + video/audio codec names."
    result = subprocess.check_output([
        FFPROBE, "-v", "error",
        "-show_entries", "format=format_name:stream=index,codec_type,codec_name",
        "-of", "json", str(path)
    ], text=True)
    data = json.loads(result)
    fmt = data["format"]["format_name"].split(",")[0]
    v   = next((s["codec_name"] for s in data["streams"] if s["codec_type"]=="video"), "none")
    a   = next((s["codec_name"] for s in data["streams"] if s["codec_type"]=="audio"), "none")
    return {"fmt": fmt, "v": v, "a": a}

def browser_ok(info: dict) -> bool:
    return (info["fmt"] in ("mov", "mp4") and
            info["v"] == "h264" and
            info["a"] in ("aac", "mp3", "none"))

def convert_inplace(src: Path):
    tmp = Path(tempfile.mktemp(dir=src.parent, suffix=".mp4"))
    print(f"      ffmpeg → {tmp.name}")
    subprocess.run([
        FFMPEG, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(tmp)
    ], check=True)
    os.replace(tmp, src)  # atomic overwrite

# ── MAIN ───────────────────────────────────────────────────────────────────
print(f"\nReading video list from {CSV_PATH} …")
to_fix = []

with open(CSV_PATH, newline='', encoding="utf-8") as fh:
    reader = csv.DictReader(fh, delimiter=DELIM)
    for row in reader:
        fname = row["videos"].strip()
        video = DATA_DIR / fname
        if not video.exists():
            print(f"[MISSING] {fname} (not found in {DATA_DIR})")
            continue

        info = probe(video)
        if browser_ok(info):
            print(f"[OK   ] {fname:40} {info['v']}/{info['a']}")
        else:
            print(f"[FIX  ] {fname:40} {info['v']}/{info['a']}")
            to_fix.append(video)

print(f"\nFound {len(to_fix)} file(s) needing conversion.\n")

for video in to_fix:
    print(f"→ Converting {video.name} …")
    convert_inplace(video)
    new_info = probe(video)
    status = "✓ now playable" if browser_ok(new_info) else "✗ still not playable"
    print(f"  Done. New codecs: {new_info['v']}/{new_info['a']}  [{status}]\n")

print("All processing complete.")
