import os
import json
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───── CONFIG ────────────────────────────────────────────────────────────
JSON_FILES  = {
    'train':'MSASL_train.json',
    'val':  'MSASL_val.json',
    'test': 'MSASL_test.json'
}
TOP106_JSON = 'label_map.json'
RAW_DIR     = 'raw_videos'
OUT_ROOT    = 'filtered'
MAX_WORKERS = min(8, os.cpu_count() or 4)
# ──────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def ffprobe_video_info(path):
    """Return a dict with 'width', 'height', 'r_frame_rate' from the first video stream."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json',
        path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    s = info['streams'][0]
    # parse r_frame_rate like "30000/1001"
    num,den = map(int, s['r_frame_rate'].split('/'))
    return {
        'width':  s['width'],
        'height': s['height'],
        'fps':     num/den
    }

def download_video(url, dest):
    if os.path.isfile(dest):
        return True
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        '-o', dest,
        url
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"⚠️  Could not download {url}")
        return False

def clip_and_crop(src, dst, start, end, box):
    """
    - start/end in seconds
    - box = [y0,x0,y1,x1] normalized
    Uses ffmpeg's normalized crop (iw,ih).
    """
    y0, x0, y1, x1 = box
    duration = end - start

    # build a single filter: normalized crop + keep source fps
    crop_filter = f"crop=iw*{x1-x0}:ih*{y1-y0}:iw*{x0}:ih*{y0}"
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{start:.3f}',
        '-i', src,
        '-t', f'{duration:.3f}',
        '-vf', crop_filter,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y', dst
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"⚠️  ffmpeg failed on {os.path.basename(src)} → skipping crop")

def process_entry(entry, split, top_set):
    gloss = entry['text']
    if gloss not in top_set:
        return

    vid_id = entry['url'].split('v=')[-1].split('&')[0]
    raw_mp4 = os.path.join(RAW_DIR, f'{vid_id}.mp4')
    out_dir = os.path.join(OUT_ROOT, split)
    ensure_dir(RAW_DIR); ensure_dir(out_dir)
    out_mp4 = os.path.join(out_dir, f'{gloss}_{vid_id}.mp4')

    # 1) download
    if not download_video(entry['url'], raw_mp4):
        return

    # 2) clip & crop—time‑based and normalized, no width/height/fps assumptions
    clip_and_crop(raw_mp4, out_mp4, entry['start_time'], entry['end_time'], entry['box'])

def process_split(split, json_path, top_set):
    data = load_json(json_path)
    print(f"{split}: {len(data)} entries → filtering to your top106…")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(process_entry, e, split, top_set) for e in data]
        for f in as_completed(futures):
            pass  # errors and skips are printed inline

if __name__ == "__main__":
    # load your gloss whitelist
    top106 = set(load_json(TOP106_JSON))
    print(f"Loaded {len(top106)} target glosses from {TOP106_JSON}")

    # run each split
    for split, jf in JSON_FILES.items():
        if not os.path.isfile(jf):
            raise FileNotFoundError(f"Missing JSON: {jf}")
        process_split(split, jf, top106)

    print(f"\n✅ All done! Your filtered clips are under: {OUT_ROOT}/")
