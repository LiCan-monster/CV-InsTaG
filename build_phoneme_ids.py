#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import numpy as np

SILENCE = {"", "sil", "sp", "spn", "<eps>", "<unk>"}

def read_phone_intervals_from_textgrid(textgrid_path):
    with open(textgrid_path, "r", encoding="utf-8") as f:
        content = f.read()
    tier_pattern = re.compile(
        r'item \[\d+\]:\s*'
        r'class = "IntervalTier"\s*'
        r'name = "([^"]+)"\s*'
        r'xmin = [^\n]+\s*'
        r'xmax = [^\n]+\s*'
        r'intervals: size = \d+\s*'
        r'(.*?)(?=\n\s*item \[\d+\]:|\Z)',
        re.S
    )
    phone_block = None
    tier_name = None
    for match in tier_pattern.finditer(content):
        name = match.group(1)
        if "phone" in name.lower():
            tier_name = name
            phone_block = match.group(2)
            break
    if phone_block is None:
        raise RuntimeError("Cannot find phones tier in TextGrid.")
    interval_pattern = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin = ([0-9.eE+-]+)\s*'
        r'xmax = ([0-9.eE+-]+)\s*'
        r'text = "(.*?)"',
        re.S
    )
    intervals = []
    for match in interval_pattern.finditer(phone_block):
        intervals.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "phone": match.group(3).strip()
        })
    print(f"Using tier: {tier_name}")
    print(f"Loaded {len(intervals)} phone intervals")
    return intervals

def load_num_frames(data_dir: Path) -> int:
    ids = []
    for name in ("transforms_train.json", "transforms_val.json"):
        p = data_dir / name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        ids.extend(int(x["img_id"]) for x in d["frames"])
    if not ids:
        raise FileNotFoundError("Cannot infer frame count.")
    return max(ids) + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cnn")
    parser.add_argument("--textgrid", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tg_path = Path(args.textgrid) if args.textgrid else data_dir / "mfa_output" / "cnn.TextGrid"

    intervals = read_phone_intervals_from_textgrid(str(tg_path))
    num_frames = load_num_frames(data_dir)

    inventory = sorted({
        x["phone"].strip().lower()
        for x in intervals
        if x["phone"].strip().lower() not in SILENCE
    })

    phone_to_id = {p: i + 1 for i, p in enumerate(inventory)}
    id_to_phone = {0: "SIL"}
    for p, i in phone_to_id.items():
        id_to_phone[i] = p

    phone_ids = np.zeros(num_frames, dtype=np.int64)

    j = 0
    for frame_id in range(num_frames):
        t = (frame_id + 0.5) / args.fps
        while j + 1 < len(intervals) and t >= intervals[j]["end"]:
            j += 1
        if j < len(intervals):
            itv = intervals[j]
            if itv["start"] <= t < itv["end"]:
                phone = itv["phone"].strip().lower()
                phone_ids[frame_id] = phone_to_id.get(phone, 0)

    np.save(data_dir / "phoneme_ids.npy", phone_ids)

    meta = {
        "fps": args.fps,
        "num_frames": int(num_frames),
        "num_phonemes_including_silence": len(inventory) + 1,
        "silence_id": 0,
        "phone_to_id": phone_to_id,
        "id_to_phone": {str(k): v for k, v in id_to_phone.items()},
    }
    with (data_dir / "phoneme_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    unique, counts = np.unique(phone_ids, return_counts=True)
    print(f"Frames: {num_frames}")
    print(f"num_phonemes_including_silence: {len(inventory) + 1}")
    for pid, count in zip(unique.tolist(), counts.tolist()):
        print(f"{pid:2d} {id_to_phone.get(pid, 'UNKNOWN'):>8s}: {count}")

    print("Saved:", data_dir / "phoneme_ids.npy")
    print("Saved:", data_dir / "phoneme_meta.json")

if __name__ == "__main__":
    main()
