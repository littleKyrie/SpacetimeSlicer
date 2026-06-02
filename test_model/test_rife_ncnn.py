"""Export real RIFE samples for visual inspection of the two production use cases."""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rife_ncnn import RifeNcnnInterpolator


def list_frame_dirs(input_dir):
    return sorted(
        entry for entry in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, entry))
    )


def read_frame(input_dir, frame_dirs, frame_idx, camera_id):
    path = os.path.join(input_dir, frame_dirs[frame_idx], f'{camera_id:03d}.jpg')
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def export_sequence(output_dir, name, first_frame, second_frame, factor, interpolator):
    stage_dir = os.path.join(output_dir, name)
    os.makedirs(stage_dir, exist_ok=True)
    frames = [first_frame]
    for step in range(1, factor):
        frames.append(interpolator.interpolate(first_frame, second_frame, step / factor))
    frames.append(second_frame)

    for index, frame in enumerate(frames):
        cv2.imwrite(os.path.join(stage_dir, f'{index:03d}.png'), frame)

    height, width = frames[0].shape[:2]
    video_path = os.path.join(output_dir, f'{name}.mp4')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f'Exported {name}: {video_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--action_camera', type=int, required=True)
    parser.add_argument('--action_start_frame', type=int, required=True)
    parser.add_argument('--action_end_frame', type=int, required=True)
    parser.add_argument('--freeze_frame', type=int, required=True)
    parser.add_argument('--freeze_camera_start', type=int, required=True)
    parser.add_argument('--freeze_camera_end', type=int, required=True)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--rife_exe')
    parser.add_argument('--rife_model_dir')
    parser.add_argument('--rife_uhd', action='store_true')
    args = parser.parse_args()

    if args.factor < 1:
        raise ValueError('--factor must be at least 1')

    os.makedirs(args.output_dir, exist_ok=True)
    frame_dirs = list_frame_dirs(args.input_dir)
    interpolator = RifeNcnnInterpolator(args.rife_exe, args.rife_model_dir, args.rife_uhd)
    try:
        export_sequence(
            args.output_dir,
            'person_action_slice_generation',
            read_frame(args.input_dir, frame_dirs, args.action_start_frame, args.action_camera),
            read_frame(args.input_dir, frame_dirs, args.action_end_frame, args.action_camera),
            args.factor,
            interpolator,
        )
        export_sequence(
            args.output_dir,
            'global_scene_freeze_orbit',
            read_frame(args.input_dir, frame_dirs, args.freeze_frame, args.freeze_camera_start),
            read_frame(args.input_dir, frame_dirs, args.freeze_frame, args.freeze_camera_end),
            args.factor,
            interpolator,
        )
    finally:
        interpolator.close()


if __name__ == '__main__':
    main()
