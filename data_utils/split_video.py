import os
import argparse
import subprocess
import torch


def extract_val(video_path, fps=25):
    base_dir = os.path.dirname(video_path)

    track_params_path = os.path.join(base_dir, 'track_params.pt')
    aud_path = os.path.join(base_dir, 'aud.wav')

    val_video_path = os.path.join(base_dir, 'val_video.mp4')
    val_audio_path = os.path.join(base_dir, 'val_audio.wav')

    # 读取 track_params.pt，得到总有效帧数
    params_dict = torch.load(
        track_params_path,
        map_location='cpu'
    )

    euler_angle = params_dict['euler']
    valid_num = euler_angle.shape[0]

    # 和原 InsTaG 代码完全一致
    train_val_split = valid_num - 25 * 12 - 1

    if train_val_split < 0:
        raise RuntimeError(
            f'视频太短: valid_num={valid_num}, '
            f'至少需要 301 帧'
        )

    val_start_frame = train_val_split
    val_end_frame = valid_num - 1
    val_frame_num = valid_num - train_val_split

    start_time = val_start_frame / fps
    duration = val_frame_num / fps

    print('================================')
    print(f'valid_num       : {valid_num}')
    print(f'val_start_frame : {val_start_frame}')
    print(f'val_end_frame   : {val_end_frame}')
    print(f'val_frame_num   : {val_frame_num}')
    print(f'start_time      : {start_time:.6f}s')
    print(f'duration        : {duration:.6f}s')
    print('================================')

    # -----------------------------
    # 提取验证集视频
    # -----------------------------
    video_filter = (
        f"select='between(n,{val_start_frame},{val_end_frame})',"
        f"setpts=N/({fps}*TB)"
    )

    cmd_video = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf',
        f"select='between(n,{val_start_frame},{val_end_frame})',"
        f"setpts=N/({fps}*TB)",
        '-an',
        '-r', str(fps),
        '-c:v', 'libopenh264',
        '-b:v', '5M',
        '-pix_fmt', 'yuv420p',
        val_video_path
    ]

    subprocess.run(cmd_video, check=True)

    # -----------------------------
    # 提取对应音频
    # -----------------------------
    # 优先使用 InsTaG 已生成的 aud.wav
    if os.path.exists(aud_path):
        audio_source = aud_path
    else:
        audio_source = video_path

    cmd_audio = [
        'ffmpeg',
        '-y',
        '-ss', f'{start_time:.6f}',
        '-i', audio_source,
        '-t', f'{duration:.6f}',
        '-acodec', 'pcm_s16le',
        val_audio_path
    ]

    subprocess.run(cmd_audio, check=True)

    print()
    print('完成:')
    print(val_video_path)
    print(val_audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'video',
        type=str,
        help='原始视频路径'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=25
    )

    args = parser.parse_args()

    extract_val(
        args.video,
        args.fps
    )