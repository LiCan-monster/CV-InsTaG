#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/cnn")
parser.add_argument("--split", required=True)
args = parser.parse_args()

data_dir = Path(args.data_dir)
ids = np.load(data_dir / "phoneme_ids.npy")

with (data_dir / "phoneme_meta.json").open("r", encoding="utf-8") as f:
    meta = json.load(f)

with (data_dir / args.split).open("r", encoding="utf-8") as f:
    split = json.load(f)

frame_ids = [int(x["img_id"]) for x in split["frames"]]
selected = ids[frame_ids]
u, c = np.unique(selected, return_counts=True)

id_to_phone = {int(k): v for k, v in meta["id_to_phone"].items()}

print("split:", args.split)
print("frames:", len(frame_ids))
print("non-silence phonemes:", sum(int(x) != 0 for x in u))
print("silence frames:", int((selected == 0).sum()))

for pid, n in zip(u.tolist(), c.tolist()):
    print(f"{pid:2d} {id_to_phone.get(pid, 'UNKNOWN'):>8s}: {n}")

full = set(range(1, meta["num_phonemes_including_silence"]))
seen = set(int(x) for x in u if int(x) != 0)
missing = sorted(full - seen)

print("missing phonemes:")
print([id_to_phone[x] for x in missing])
