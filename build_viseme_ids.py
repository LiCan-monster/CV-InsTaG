#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

import re

def read_phone_intervals_from_textgrid(textgrid_path):
    """
    直接解析 MFA 生成的 Praat TextGrid。
    不需要安装 textgrid 包。
    """
    with open(textgrid_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 找 phones tier
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
        raise RuntimeError(
            "没有找到 phones tier，请检查 cnn.TextGrid。"
        )

    interval_pattern = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin = ([0-9.eE+-]+)\s*'
        r'xmax = ([0-9.eE+-]+)\s*'
        r'text = "(.*?)"',
        re.S
    )

    intervals = []

    for match in interval_pattern.finditer(phone_block):
        start = float(match.group(1))
        end = float(match.group(2))
        phone = match.group(3).strip()

        intervals.append({
            "start": start,
            "end": end,
            "phone": phone
        })

    print(f"Using tier: {tier_name}")
    print(f"Loaded {len(intervals)} phone intervals")

    return intervals

# 0 = silence / unknown
VISEME_NAMES = {
    0: "SIL",
    1: "PBM",
    2: "FV",
    3: "ALVEOLAR_DENTAL",
    4: "POSTALVEOLAR_R",
    5: "VELAR_GLOTTAL",
    6: "FRONT_VOWEL",
    7: "OPEN_CENTRAL_VOWEL",
    8: "ROUND_BACK_VOWEL",
}

VISEME_PHONES = {
    1: {
        "p", "pʰ", "pʲ", "b", "bʲ",
        "m", "mʲ", "m̩",
    },
    2: {
        "f", "fʲ", "v", "vʲ", "ɱ",
    },
    3: {
        "t", "tʰ", "tʲ", "t̪",
        "d", "dʲ", "d̪",
        "s", "z", "n", "n̩",
        "l", "ɫ", "ɫ̩", "ɾ",
        "θ", "ð",
    },
    4: {
        "tʃ", "dʒ", "ʃ", "ʒ", "ɹ", "ʎ",
    },
    5: {
        "k", "kʰ", "ɡ", "g", "ŋ",
        "c", "cʰ", "ɟ", "ç",
        "h", "ʔ", "ɲ",
    },
    6: {
        "i", "iː", "ɪ", "e", "ej",
        "ɛ", "ɛː", "æ", "j",
    },
    7: {
        "a", "aː", "aj", "aw",
        "ɑ", "ɑː", "ɐ",
        "ə", "ɚ", "ɜ", "ɜː", "ɝ", "ʌ",
    },
    8: {
        "o", "ow",
        "u", "uː", "ʊ", "ʉ", "ʉː",
        "ɔ", "ɔj", "ɒ", "ɒː", "əw",
        "w",
    },
}

SILENCE = {"", "sil", "sp", "spn", "<eps>", "<unk>"}

PHONE_TO_VISEME = {}
for vid, phones in VISEME_PHONES.items():
    for p in phones:
        PHONE_TO_VISEME[p] = vid


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
        raise FileNotFoundError(
            "Cannot infer frame count: transforms_train.json / transforms_val.json not found."
        )
    return max(ids) + 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cnn")
    parser.add_argument("--textgrid", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tg_path = Path(args.textgrid) if args.textgrid else data_dir / "mfa_output" / "cnn.TextGrid"

    if not tg_path.exists():
        raise FileNotFoundError(tg_path)

    num_frames = load_num_frames(data_dir)

    raw_intervals = read_phone_intervals_from_textgrid(
        str(tg_path)
    )

    intervals = []
    unknown_phones = set()
    phone_inventory = set()

    for itv in raw_intervals:

        phone = itv["phone"].strip().lower()

        if phone in SILENCE:
            vid = 0
        else:
            phone_inventory.add(phone)

            vid = PHONE_TO_VISEME.get(
                phone,
                0
            )

            if vid == 0:
                unknown_phones.add(phone)

        intervals.append(
            (
                itv["start"],
                itv["end"],
                vid,
                phone
            )
        )

    viseme_ids = np.zeros(num_frames, dtype=np.int64)
    phone_labels = [""] * num_frames

    j = 0
    for frame_id in range(num_frames):
        t = (frame_id + 0.5) / args.fps

        while j + 1 < len(intervals) and t >= intervals[j][1]:
            j += 1

        if j < len(intervals):
            t0, t1, vid, phone = intervals[j]
            if t0 <= t < t1:
                viseme_ids[frame_id] = vid
                phone_labels[frame_id] = phone

    np.save(data_dir / "viseme_ids.npy", viseme_ids)

    meta = {
        "fps": args.fps,
        "num_frames": int(num_frames),
        "num_visemes_including_silence": len(VISEME_NAMES),
        "viseme_names": {str(k): v for k, v in VISEME_NAMES.items()},
        "phone_to_viseme": PHONE_TO_VISEME,
        "phone_inventory": sorted(phone_inventory),
        "unmapped_phones": sorted(unknown_phones),
    }
    with (data_dir / "viseme_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    unique, counts = np.unique(viseme_ids, return_counts=True)
    count_dict = {VISEME_NAMES[int(k)]: int(v) for k, v in zip(unique, counts)}

    print(f"Frames: {num_frames}")
    print("Viseme frame counts:", count_dict)
    print("Saved:", data_dir / "viseme_ids.npy")
    print("Saved:", data_dir / "viseme_meta.json")

    if unknown_phones:
        print("\nWARNING: unmapped non-silence phones:")
        print(sorted(unknown_phones))
        print("Please add them to VISEME_PHONES before the final experiment.")
    else:
        print("\nAll non-silence phones were mapped.")


if __name__ == "__main__":
    main()
