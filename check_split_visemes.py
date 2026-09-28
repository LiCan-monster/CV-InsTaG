#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

NAMES = {
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

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/cnn")
parser.add_argument("--split", required=True, help="e.g. transforms_train_low.json")
args = parser.parse_args()

data_dir = Path(args.data_dir)
ids = np.load(data_dir / "viseme_ids.npy")

with (data_dir / args.split).open("r", encoding="utf-8") as f:
    d = json.load(f)

frame_ids = [int(x["img_id"]) for x in d["frames"]]
v = ids[frame_ids]

u, c = np.unique(v, return_counts=True)
print("split:", args.split)
print("frames:", len(frame_ids))
print("covered non-silence visemes:", sum(int(x) != 0 for x in u))
for k, n in zip(u, c):
    print(f"{int(k)} {NAMES.get(int(k), 'UNKNOWN'):>20}: {int(n)} frames")
