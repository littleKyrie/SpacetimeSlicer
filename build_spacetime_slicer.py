import argparse
import time

import cv2
import numpy as np

from models.rife_ncnn import RifeNcnnInterpolator
from models.spacetime_slicer import SpacetimeSlicer


def parse_camera_ids(value):
    """Parse comma-separated camera IDs or an inclusive colon range."""
    if ':' in value:
        start, end = map(int, value.split(':'))
        return list(range(start, end + 1))
    if ',' in value:
        return list(map(int, value.split(',')))
    return [int(value)]


def create_strategy(method, slicer, camera_ids):
    if method == 'RVM':
        from models.rvm import RVMStrategy

        return RVMStrategy(slicer.device)
    if method == 'Hybrid':
        from models.hybrid_rvm import HybridStrategy

        print('>> Building the temporal median background for Hybrid segmentation...')
        start_cam = camera_ids[0]
        gray_frames = [
            cv2.cvtColor(cv2.imread(slicer.frame_paths_dict[start_cam][i]), cv2.COLOR_BGR2GRAY)
            for i in range(0, slicer.total_frames, 5)
        ]
        median_bg = np.median(gray_frames, axis=0).astype(np.uint8)
        return HybridStrategy(slicer.device, median_bg)
    if method == 'SAM2_BBox':
        from models.yolo_sam2 import YOLO_SAM2_Strategy

        return YOLO_SAM2_Strategy(slicer.device)
    if method == 'RMBG2':
        from models.rmbg2 import RMBG2Strategy

        return RMBG2Strategy(slicer.device)
    if method.startswith('rembg-'):
        from models.rembg import RembgStrategy

        return RembgStrategy(slicer.device, method.split('-', 1)[1])
    raise ValueError(f'Unknown method: {method}')


def build_parser():
    parser = argparse.ArgumentParser(description='Spacetime Slicer')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', '--output_root', dest='output_dir', required=True)
    parser.add_argument('--camera_ids', default='0', help='Comma-separated IDs or inclusive range, such as 0:30 (its id corresponds to the actual file id in the subdir)')
    parser.add_argument('--fps', type=int, default=25, help='FPS of the output video')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame of the ghost effects (inclusive and its id should be +1 to corresponds to the actual subdir id)')
    parser.add_argument('--freeze_frame', type=int, required=True, help='Frame to freeze and end all ghost effects (inclusive and its id should be +1 to corresponds to the actual subdir id)')
    parser.add_argument('--end_frame', type=int, required=True, help='End frame of the input video (exclusive and its id should be +1 to corresponds to the actual subdir id)')
    parser.add_argument('--ghost_interval', type=int, default=1)
    parser.add_argument('--edge_feather', type=int, default=0)
    parser.add_argument(
        '--cutout_alpha_threshold',
        type=int,
        default=128,
        help='Alpha threshold for opaque cutout compositing; raise to remove dark fringes, lower to keep more hair/detail.',
    )
    parser.add_argument(
        '--cutout_edge_blur',
        type=int,
        default=1,
        help='Small matte blur radius for cutout anti-aliasing; set 0 for a hard binary edge.',
    )
    parser.add_argument('--fade_duration_frames', type=int)
    parser.add_argument('--ghost_opacity_start', type=float, default=1.0)
    parser.add_argument('--ghost_opacity_end', type=float, default=1.0)
    parser.add_argument('--stretch_head', type=int, default=1)
    parser.add_argument('--stretch_ghost', type=int, default=1)
    parser.add_argument('--stretch_fade', type=int, default=1)
    parser.add_argument('--stretch_freeze', type=int, default=1)
    parser.add_argument(
        '--freeze_interp_mode',
        default='repeat',
        choices=['rife', 'repeat', 'blend'],
        help='Freeze orbit: rife=synthesize views, repeat=hold real frames, blend=crossfade real frames',
    )
    parser.add_argument('--stretch_tail', type=int, default=1)
    parser.add_argument('--background_mode', default='freeze', choices=['median', 'freeze', 'start'])
    parser.add_argument('--initial_subject_patch_mode', default='median', choices=['median', 'freeze'])
    parser.add_argument('--recovery_transition_frames', type=int, default=3)
    parser.add_argument('--rife_exe', help='Path to rife-ncnn-vulkan executable')
    parser.add_argument(
        '--rife_model_dir',
        help='Optional RIFE v4 model directory; defaults to rife-v4.6 next to the executable',
    )
    parser.add_argument('--rife_uhd', action='store_true')
    parser.add_argument('--method', default='RVM',
                        help='RVM, Hybrid, SAM2_BBox, RMBG2, or rembg-<model>')
    return parser


def main():
    args = build_parser().parse_args()
    camera_ids = parse_camera_ids(args.camera_ids)
    print(f'Camera IDs: {camera_ids}')
    start_time = time.time()

    rife_interpolator = None
    if args.stretch_ghost > 1 or (
        args.freeze_interp_mode == 'rife' and args.stretch_freeze > 1
    ):
        rife_interpolator = RifeNcnnInterpolator(
            executable=args.rife_exe,
            model_dir=args.rife_model_dir,
            uhd=args.rife_uhd,
        )

    try:
        slicer = SpacetimeSlicer(
            args.input_dir,
            args.output_dir,
            fps=args.fps,
            camera_ids=camera_ids,
            rife_interpolator=rife_interpolator,
        )
        print(f'Device: {slicer.device}')
        end_frame = args.end_frame if args.end_frame is not None else slicer.total_frames
        strategy = create_strategy(args.method, slicer, camera_ids)

        print(f'Starting spacetime slicer with {args.method}')
        print(f'Frames: {args.start_frame} -> {args.freeze_frame} -> {end_frame}')
        print(f'RIFE factors: slices={args.stretch_ghost}, freeze_orbit={args.stretch_freeze}')
        print(f'Freeze orbit interpolation: {args.freeze_interp_mode}')

        slicer.generate(
            strategy,
            args.start_frame,
            args.freeze_frame,
            end_frame,
            camera_ids=camera_ids,
            ghost_interval=args.ghost_interval,
            edge_feather=args.edge_feather,
            cutout_alpha_threshold=args.cutout_alpha_threshold,
            cutout_edge_blur=args.cutout_edge_blur,
            fade_duration_frames=args.fade_duration_frames,
            ghost_opacity_start=args.ghost_opacity_start,
            ghost_opacity_end=args.ghost_opacity_end,
            stretch_head=args.stretch_head,
            stretch_ghost=args.stretch_ghost,
            stretch_fade=args.stretch_fade,
            stretch_freeze=args.stretch_freeze,
            stretch_tail=args.stretch_tail,
            freeze_interp_mode=args.freeze_interp_mode,
            background_mode=args.background_mode,
            recovery_transition_frames=args.recovery_transition_frames,
            initial_subject_patch_mode=args.initial_subject_patch_mode,
        )
    finally:
        if rife_interpolator is not None:
            rife_interpolator.close()

    print(f'Total time: {time.time() - start_time:.2f}s')


if __name__ == '__main__':
    main()
