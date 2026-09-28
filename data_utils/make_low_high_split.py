import json
import copy
from pathlib import Path

# =========================
# 配置
# =========================
INPUT_JSON = "transforms_train.json"

# 5 秒 = 125 帧，25 FPS
LOW_START = 35
LOW_END = 159       # inclusive，共 125 帧

HIGH_START = 250
HIGH_END = 374


def make_split(data, start_id, end_id, output_path):
    """
    按 img_id 选择 [start_id, end_id] 范围内的 frame。
    保留 transforms_train.json 中除 frames 外的所有原始信息。
    """

    selected_frames = []

    for frame in data["frames"]:
        if "img_id" not in frame:
            raise KeyError(
                "frame 中没有 'img_id' 字段，请先检查 transforms_train.json 的格式。"
            )

        img_id = int(frame["img_id"])

        if start_id <= img_id <= end_id:
            selected_frames.append(frame)

    # 按 img_id 排序，避免顺序异常
    selected_frames = sorted(
        selected_frames,
        key=lambda x: int(x["img_id"])
    )

    # 深拷贝，避免修改原数据
    new_data = copy.deepcopy(data)
    new_data["frames"] = selected_frames

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    ids = [int(x["img_id"]) for x in selected_frames]

    print("=" * 60)
    print(f"Saved: {output_path}")
    print(f"Requested range: {start_id} ~ {end_id}")
    print(f"Number of frames: {len(selected_frames)}")

    if ids:
        print(f"Actual img_id range: {min(ids)} ~ {max(ids)}")
        print(f"First 5 IDs: {ids[:5]}")
        print(f"Last 5 IDs:  {ids[-5:]}")
    else:
        print("WARNING: 没有找到任何符合条件的 frame!")

    expected = end_id - start_id + 1

    if len(selected_frames) != expected:
        print(
            f"WARNING: 预期 {expected} 帧，"
            f"实际只找到 {len(selected_frames)} 帧。"
        )
    else:
        print(f"OK: 正好 {expected} 帧。")


def main():
    input_path = Path(INPUT_JSON)

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 {INPUT_JSON}，请在 data/cnn 目录下运行脚本。"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "frames" not in data:
        raise KeyError("transforms_train.json 中没有 frames 字段。")

    print(f"Original number of frames: {len(data['frames'])}")

    # Low phoneme coverage
    make_split(
        data,
        LOW_START,
        LOW_END,
        "transforms_train_low.json"
    )

    # High phoneme coverage
    make_split(
        data,
        HIGH_START,
        HIGH_END,
        "transforms_train_high.json"
    )


if __name__ == "__main__":
    main()