import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from models.rife_ncnn import RifeNcnnInterpolator
from models.spacetime_slicer import SpacetimeSlicer


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'spacetime_slicer.json'
REQUIRED_ARGUMENTS = ('input_dir', 'output_dir', 'freeze_frame')


class ConfigArgumentParser(argparse.ArgumentParser):
    """Argument parser whose JSON config values are overridden by CLI values."""

    def _load_config(self, path):
        config_path = Path(path).expanduser()
        try:
            with config_path.open('r', encoding='utf-8') as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            self.error(f'config file not found: {config_path}')
        except json.JSONDecodeError as exc:
            self.error(
                f'invalid JSON in config file {config_path}: '
                f'line {exc.lineno}, column {exc.colno}: {exc.msg}'
            )

        if not isinstance(config, dict):
            self.error(f'config file must contain a JSON object: {config_path}')

        actions = {
            action.dest: action
            for action in self._actions
            if action.dest not in ('help', 'config')
        }
        unknown_keys = sorted(set(config) - set(actions))
        if unknown_keys:
            self.error(
                f'unknown config option(s) in {config_path}: '
                f'{", ".join(unknown_keys)}'
            )

        validated = {}
        for key, value in config.items():
            action = actions[key]
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    self.error(f'config option {key} must be true or false')
                validated[key] = value
                continue

            if value is not None and action.type is not None:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as exc:
                    self.error(f'invalid value for config option {key}: {exc}')

            if value is not None and action.choices is not None and value not in action.choices:
                choices = ', '.join(map(str, action.choices))
                self.error(
                    f'invalid value for config option {key}: {value!r} '
                    f'(choose from {choices})'
                )
            validated[key] = value

        return validated

    def parse_args(self, args=None, namespace=None):
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
        config_args, _ = config_parser.parse_known_args(args)

        config_defaults = self._load_config(config_args.config)
        if namespace is None:
            namespace = argparse.Namespace()
        for key, value in config_defaults.items():
            setattr(namespace, key, value)

        parsed = super().parse_args(args, namespace)
        missing = [
            f'--{name}'
            for name in REQUIRED_ARGUMENTS
            if getattr(parsed, name, None) in (None, '')
        ]
        if missing:
            self.error(
                'the following arguments are required in the config file or '
                f'on the command line: {", ".join(missing)}'
            )
        return parsed


def parse_camera_ids(value):
    """Parse comma-separated camera IDs or an inclusive colon range."""
    if ':' in value:
        start, end = map(int, value.split(':'))
        return list(range(start, end + 1))
    if ',' in value:
        return list(map(int, value.split(',')))
    return [int(value)]


def parse_frame_ids(value):
    """Parse comma-separated frame IDs or an inclusive colon range."""
    return parse_camera_ids(value)


def normalize_cli_frame_args(args):
    """Convert 1-based source frame IDs from the CLI to internal frame indices."""
    frame_values = {
        'start_frame': args.start_frame,
        'freeze_frame': args.freeze_frame,
    }
    if args.end_frame is not None:
        frame_values['end_frame'] = args.end_frame
    if args.initial_subject_patch_frame is not None:
        frame_values['initial_subject_patch_frame'] = args.initial_subject_patch_frame

    for name, value in frame_values.items():
        if value < 1:
            raise ValueError(f'{name} must be a 1-based frame id, got {value}')

    args.source_start_frame = args.start_frame
    args.source_freeze_frame = args.freeze_frame
    args.source_end_frame = args.end_frame
    args.source_initial_subject_patch_frame = args.initial_subject_patch_frame

    args.start_frame -= 1
    args.freeze_frame -= 1
    if args.initial_subject_patch_frame is not None:
        args.initial_subject_patch_frame -= 1
    return args


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


def save_debug_extractions(strategy, slicer, args, camera_ids, end_frame):
    debug_frames = parse_frame_ids(args.debug_extract_frames)
    debug_camera = args.debug_extract_camera
    if debug_camera is None:
        debug_camera = camera_ids[0]

    slice_end_idx, _ = slicer.resolve_effect_schedule(
        args.start_frame,
        args.freeze_frame,
        args.fade_duration_frames,
        args.recovery_timing,
    )
    ghost_frames = list(range(args.start_frame, slice_end_idx + 1, args.ghost_interval))
    ghost_opacities = np.linspace(
        args.ghost_opacity_start,
        args.ghost_opacity_end,
        max(1, len(ghost_frames)),
    )
    ghost_opacity_by_frame = {
        frame_idx: float(ghost_opacities[pos])
        for pos, frame_idx in enumerate(ghost_frames)
    }

    output_dir = os.path.join(
        args.output_dir,
        'debug_extractions',
        f'cam{debug_camera:03d}_s{args.start_frame}_f{args.freeze_frame}_e{end_frame}',
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f'Debug extraction output: {output_dir}')
    print(f'Slice capture frames: {ghost_frames}')

    for frame_idx in debug_frames:
        frame = slicer.read_frame(frame_idx, debug_camera)
        if frame is None:
            raise ValueError(f'Could not read frame {frame_idx} for camera {debug_camera}')

        alpha_mask = strategy.process_frame(frame, frame_idx)
        if args.edge_feather < 0:
            kernel = np.ones((3, 3), np.uint8)
            alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(args.edge_feather))

        ghost_opacity = ghost_opacity_by_frame.get(frame_idx)
        if ghost_opacity is None:
            ghost_opacity = 0.0
            is_slice_frame = False
        else:
            is_slice_frame = True

        ghost_alpha = np.clip(
            alpha_mask.astype(np.float32) * ghost_opacity,
            0,
            255,
        ).astype(np.uint8)
        patch_mask = np.where(
            alpha_mask > args.initial_patch_alpha_threshold,
            255,
            0,
        ).astype(np.uint8)
        if args.initial_patch_dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            patch_mask = cv2.dilate(patch_mask, kernel, iterations=args.initial_patch_dilate)
        prefix = os.path.join(output_dir, f'frame{frame_idx:04d}_cam{debug_camera:03d}')
        cv2.imwrite(f'{prefix}_source.jpg', frame)
        cv2.imwrite(f'{prefix}_rvm_alpha.png', alpha_mask)
        cv2.imwrite(
            f'{prefix}_initial_patch_mask_t{args.initial_patch_alpha_threshold}_d{args.initial_patch_dilate}.png',
            patch_mask,
        )
        raw_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        raw_rgba[:, :, 3] = alpha_mask
        cv2.imwrite(f'{prefix}_rvm_cutout_rgba.png', raw_rgba)

        ghost_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        ghost_rgba[:, :, 3] = ghost_alpha
        cv2.imwrite(f'{prefix}_slice_rgba_opacity{ghost_opacity:.3f}.png', ghost_rgba)

        alpha_3ch = np.repeat((ghost_alpha.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)
        preview_black = (frame.astype(np.float32) * alpha_3ch).astype(np.uint8)
        cv2.imwrite(f'{prefix}_slice_preview_on_black.jpg', preview_black)

        alpha_values = alpha_mask.reshape(-1)
        print(
            f'Frame {frame_idx} cam {debug_camera}: '
            f'is_slice={is_slice_frame}, ghost_opacity={ghost_opacity:.3f}, '
            f'alpha min={int(alpha_values.min())}, max={int(alpha_values.max())}, '
            f'mean={float(alpha_values.mean()):.2f}, '
            f'pixels>240={int((alpha_values > 240).sum())}'
        )


def build_parser():
    parser = ConfigArgumentParser(
        description='Spacetime Slicer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG_PATH),
        help='JSON config file. Command-line arguments override its values.',
    )
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir', '--output_root', dest='output_dir')
    parser.add_argument('--camera_ids', default='0', help='Comma-separated IDs or inclusive range, such as 1:30 (its id corresponds to the actual file id in the subdir)')
    parser.add_argument('--fps', type=int, default=25, help='FPS of the output video')
    parser.add_argument('--start_frame', type=int, help='1-based source frame ID where ghost effects start, inclusive.')
    parser.add_argument('--freeze_frame', type=int, help='1-based source frame ID where slice recovery completes and the multi-camera freeze orbit begins.')
    parser.add_argument('--end_frame', type=int, default=None, help='1-based source frame ID where output ends, inclusive; defaults to the last source frame.')
    parser.add_argument('--ghost_interval', type=int, default=20)
    parser.add_argument('--edge_feather', type=int, default=0)
    parser.add_argument('--fade_duration_frames', type=int, default=10, help='Recovery duration in frames.')
    parser.add_argument('--ghost_opacity_start', type=float, default=0.2)
    parser.add_argument('--ghost_opacity_end', type=float, default=1.0)
    parser.add_argument('--initial_subject_patch_mode', default='freeze', choices=['none', 'median', 'freeze', 'frame'], help='Background source for removing the start-frame subject before overlaying the first translucent slice')
    parser.add_argument('--initial_subject_patch_frame', type=int, help='Manual 1-based source frame ID used when --initial_subject_patch_mode frame; defaults to --freeze_frame')
    parser.add_argument('--initial_canvas_mode', default='patched_start', choices=['patched_start', 'clean'], help='Initial canvas for patched_canvas mode: patched_start uses start_frame with the subject region replaced; clean starts from the replacement frame')
    parser.add_argument('--initial_patch_alpha_threshold', type=int, default=1, help='Alpha threshold used to remove the start-frame subject from the first output frame')
    parser.add_argument('--initial_patch_dilate', type=int, default=1, help='Dilate the start-frame subject patch mask to remove edge residue')
    parser.add_argument('--live_subject_alpha_threshold', type=int, default=16, help='Alpha threshold for keeping the current live subject opaque in patched_canvas mode')
    parser.add_argument('--live_subject_opacity', type=float, default=1.0, help='Opacity of the current live subject in patched_canvas mode')
    parser.add_argument('--effect_base_mode', default='patched_canvas', choices=['patched_canvas', 'source'], help='patched_canvas mattes the current subject every frame; source uses each original frame as the base and only mattes slice frames')
    parser.add_argument('--debug_extract_frames', help='Write RVM alpha/cutout diagnostics for comma-separated frames or an inclusive range, then exit')
    parser.add_argument('--debug_extract_camera', type=int, help='Camera ID for --debug_extract_frames; defaults to the first --camera_ids entry')
    parser.add_argument('--stretch_head', type=int, default=1)
    parser.add_argument('--stretch_ghost', type=int, default=1)
    parser.add_argument('--stretch_fade', type=int, default=1)
    parser.add_argument('--stretch_freeze', type=int, default=1)
    parser.add_argument(
        '--freeze_interp_mode',
        default='rife',
        choices=['rife', 'repeat', 'blend'],
        help='Freeze orbit: rife=synthesize views, repeat=hold real frames, blend=crossfade real frames',
    )
    parser.add_argument('--stretch_tail', type=int, default=1)
    parser.add_argument(
        '--tail_camera_id',
        type=int,
        help='Camera used after the freeze orbit; defaults to the last camera with tail frames, falling back to the first camera for sparse freeze-only views.',
    )
    parser.add_argument('--background_mode', default='freeze', choices=['median', 'freeze', 'start'])
    parser.add_argument('--recovery_transition_frames', type=int, default=3)
    parser.add_argument(
        '--recovery_timing',
        default='after_freeze',
        choices=['after_freeze', 'before_freeze'],
        help='after_freeze inserts recovery frames after the freeze source frame; before_freeze reserves source frames before freeze for recovery.',
    )
    parser.add_argument('--rife_exe', help='Path to rife-ncnn-vulkan executable')
    parser.add_argument(
        '--rife_model_dir',
        help='Optional RIFE v4 model directory; defaults to rife-v4.6 next to the executable',
    )
    parser.add_argument(
        '--rife_uhd',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable UHD mode; use --no-rife_uhd to override a true config value.',
    )
    parser.add_argument('--method', default='RVM',
                        help='RVM, Hybrid, SAM2_BBox, RMBG2, or rembg-<model>')
    return parser


def main(argv=None):
    start_time = time.time()
    args = normalize_cli_frame_args(build_parser().parse_args(argv))
    camera_ids = parse_camera_ids(args.camera_ids)
    print(f'Camera IDs: {camera_ids}')

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
        if args.end_frame is None:
            args.end_frame = slicer.total_frames
            args.source_end_frame = slicer.total_frames
        end_frame = args.end_frame
        strategy = create_strategy(args.method, slicer, camera_ids)

        print(f'Starting spacetime slicer with {args.method}')
        print(
            'Frames: '
            f'slices start at source frame {args.source_start_frame}, '
            f'recovered/frozen at source frame {args.source_freeze_frame}, '
            f'tail ends at source frame {args.source_end_frame}'
        )
        print(f'RIFE factors: slices={args.stretch_ghost}, freeze_orbit={args.stretch_freeze}')
        print(f'Freeze orbit interpolation: {args.freeze_interp_mode}')

        if args.debug_extract_frames:
            save_debug_extractions(strategy, slicer, args, camera_ids, end_frame)
            return

        slicer.generate(
            strategy,
            args.start_frame,
            args.freeze_frame,
            end_frame,
            camera_ids=camera_ids,
            ghost_interval=args.ghost_interval,
            edge_feather=args.edge_feather,
            fade_duration_frames=args.fade_duration_frames,
            ghost_opacity_start=args.ghost_opacity_start,
            ghost_opacity_end=args.ghost_opacity_end,
            stretch_head=args.stretch_head,
            stretch_ghost=args.stretch_ghost,
            stretch_fade=args.stretch_fade,
            stretch_freeze=args.stretch_freeze,
            stretch_tail=args.stretch_tail,
            freeze_interp_mode=args.freeze_interp_mode,
            tail_camera_id=args.tail_camera_id,
            background_mode=args.background_mode,
            recovery_transition_frames=args.recovery_transition_frames,
            recovery_timing=args.recovery_timing,
            initial_canvas_mode=args.initial_canvas_mode,
            initial_subject_patch_mode=args.initial_subject_patch_mode,
            initial_subject_patch_frame=args.initial_subject_patch_frame,
            initial_patch_alpha_threshold=args.initial_patch_alpha_threshold,
            initial_patch_dilate=args.initial_patch_dilate,
            effect_base_mode=args.effect_base_mode,
            live_subject_opacity=args.live_subject_opacity,
            live_subject_alpha_threshold=args.live_subject_alpha_threshold,
        )
    finally:
        if rife_interpolator is not None:
            rife_interpolator.close()

    print(f'Total time: {time.time() - start_time:.2f}s')


if __name__ == '__main__':
    main()
