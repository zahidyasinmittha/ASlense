import subprocess, json
from pathlib import Path

# ── videos to test ─────────────────────────────────────────────────────────
video1 = Path(r"E:\final_dataset_videos\he_video_5")            # ← not playing
video2 = Path(r"E:\final_dataset_videos\5798535143557326-QM.mp4")  # ← plays
videos = [video1, video2]

# ── ffprobe helper ─────────────────────────────────────────────────────────
def probe(p: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=format_name:stream=index,codec_type,codec_name",
        "-of", "json", str(p)
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    fmt = data["format"]["format_name"].split(",")[0]
    v  = next((s["codec_name"] for s in data["streams"] if s["codec_type"]=="video"), "none")
    a  = next((s["codec_name"] for s in data["streams"] if s["codec_type"]=="audio"), "none")
    return dict(container=fmt, video=v, audio=a)

def playable(info: dict) -> bool:
    return (info["container"] in ("mov", "mp4") and
            info["video"] == "h264" and
            info["audio"] in ("aac", "mp3", "none"))

# ── run check ─────────────────────────────────────────────────────────────
for vid in videos:
    if not vid.exists():
        print(f"[missing] {vid}")
        continue
    info = probe(vid)
    status = "PLAYABLE ✅" if playable(info) else "NEEDS CONVERT ❌"
    print(f"{vid.name:40} {status}  {info}")
