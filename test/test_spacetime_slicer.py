import json
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from build_spacetime_slicer import build_parser, normalize_cli_frame_args, parse_frame_ids
from models.spacetime_slicer import SpacetimeSlicer, resolve_output_video_path


class FrameCollector:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class ConstantAlphaStrategy:
    def __init__(self, alpha):
        self.alpha = alpha
        self.calls = []

    def process_frame(self, current_frame, current_idx):
        self.calls.append(current_idx)
        return self.alpha.copy()


class SequenceAlphaStrategy:
    def __init__(self, alphas):
        self.alphas = alphas
        self.calls = []

    def process_frame(self, current_frame, current_idx):
        self.calls.append(current_idx)
        return self.alphas[current_idx].copy()


class FakeRifeInterpolator:
    def __init__(self):
        self.calls = []

    def interpolate(self, first_frame, second_frame, timestep):
        self.calls.append(timestep)
        return cv2.addWeighted(first_frame, 1.0 - timestep, second_frame, timestep, 0)


def make_slicer(frames, rife_interpolator=None, source_frames=None):
    slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
    slicer.camera_ids = [0]
    slicer.frame_paths_dict = {0: list(range(len(frames)))}
    slicer.read_frame = lambda idx, camera_id=None: frames[idx].copy()
    source_frames = frames if source_frames is None else source_frames
    slicer.source_image_paths = list(range(len(source_frames)))
    slicer.read_source_image = lambda idx: source_frames[idx].copy()
    slicer.rife_interpolator = rife_interpolator
    slicer.fps = 25
    return slicer


class SpacetimeSlicerTest(unittest.TestCase):
    def test_output_video_filename_matches_output_directory(self):
        output_dir = Path('风暴时刻输出') / 'QPA-2026-07-18-103215'

        self.assertEqual(
            Path(resolve_output_video_path(output_dir)),
            output_dir / 'QPA-2026-07-18-103215.mp4',
        )

    def test_output_video_auto_increment_when_mp4_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'QPA-2026-07-18-103215'
            output_dir.mkdir(parents=True)
            # Create existing non-empty mp4
            base_mp4 = output_dir / 'QPA-2026-07-18-103215.mp4'
            base_mp4.write_bytes(b'existing')

            self.assertEqual(
                Path(resolve_output_video_path(str(output_dir))),
                output_dir / 'QPA-2026-07-18-103215-1.mp4',
            )

    def test_output_video_auto_increment_skips_empty_mp4(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'QPA-2026-07-18-103215'
            output_dir.mkdir(parents=True)
            # Create empty mp4 (should be treated as non-existent)
            base_mp4 = output_dir / 'QPA-2026-07-18-103215.mp4'
            base_mp4.write_bytes(b'')

            self.assertEqual(
                Path(resolve_output_video_path(str(output_dir))),
                output_dir / 'QPA-2026-07-18-103215.mp4',
            )

    def test_output_video_auto_increment_multiple_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'QPA-2026-07-18-103215'
            output_dir.mkdir(parents=True)
            # Create base and -1 mp4
            (output_dir / 'QPA-2026-07-18-103215.mp4').write_bytes(b'existing')
            (output_dir / 'QPA-2026-07-18-103215-1.mp4').write_bytes(b'existing')

            self.assertEqual(
                Path(resolve_output_video_path(str(output_dir))),
                output_dir / 'QPA-2026-07-18-103215-2.mp4',
            )

    def test_cli_frame_ids_are_one_based_source_frame_ids(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--start_frame', '25',
            '--freeze_frame', '125',
            '--end_frame', '149',
            '--initial_subject_patch_frame', '125',
        ])

        normalize_cli_frame_args(args)

        self.assertEqual(args.initial_subject_patch_mode, 'frame')
        self.assertEqual(args.start_frame, 24)
        self.assertEqual(args.freeze_frame, 124)
        self.assertEqual(args.end_frame, 149)
        self.assertEqual(args.initial_subject_patch_frame, 124)
        self.assertEqual(args.source_start_frame, 25)
        self.assertEqual(args.source_freeze_frame, 125)
        self.assertEqual(args.source_end_frame, 149)

    def test_initial_subject_patch_frame_auto_switches_mode_to_frame(self):
        """--initial_subject_patch_frame implies frame mode without needing --initial_subject_patch_mode."""
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--start_frame', '1',
            '--freeze_frame', '75',
            '--initial_subject_patch_frame', '99',
        ])
        # Before normalization: mode should still be default 'freeze'
        self.assertEqual(args.initial_subject_patch_mode, 'freeze')

        normalize_cli_frame_args(args)

        self.assertEqual(args.initial_subject_patch_mode, 'frame')
        self.assertEqual(args.initial_subject_patch_frame, 98)  # 1-based → 0-based

    def test_initial_subject_patch_mode_unchanged_without_frame_arg(self):
        """Without --initial_subject_patch_frame, the mode keeps its explicit value."""
        for mode in ('freeze', 'none', 'median'):
            with self.subTest(mode=mode):
                args = build_parser().parse_args([
                    '--input_dir', 'data',
                    '--output_dir', 'results',
                    '--start_frame', '1',
                    '--freeze_frame', '75',
                    '--initial_subject_patch_mode', mode,
                ])
                normalize_cli_frame_args(args)
                self.assertEqual(args.initial_subject_patch_mode, mode)

    def test_cli_frame_ids_reject_zero(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--start_frame', '0',
            '--freeze_frame', '125',
            '--end_frame', '149',
        ])

        with self.assertRaisesRegex(ValueError, 'start_frame'):
            normalize_cli_frame_args(args)

    def test_end_frame_can_be_omitted_until_input_frames_are_loaded(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--freeze_frame', '125',
        ])

        normalize_cli_frame_args(args)

        self.assertIsNone(args.end_frame)
        self.assertIsNone(args.source_end_frame)

    def test_cli_accepts_tail_camera_id(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--freeze_frame', '125',
            '--end_frame', '149',
            '--tail_camera_id', '1',
        ])

        self.assertEqual(args.tail_camera_id, 1)
        self.assertEqual(args.recovery_timing, 'after_freeze')

    def test_cli_accepts_before_freeze_recovery_timing(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--freeze_frame', '125',
            '--end_frame', '149',
            '--recovery_timing', 'before_freeze',
        ])

        self.assertEqual(args.recovery_timing, 'before_freeze')

    def test_multi_subject_mode_defaults_to_largest_component(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--freeze_frame', '125',
        ])

        self.assertEqual(args.multi_subject_mode, 'largest_component')

    def test_cli_accepts_all_components_multi_subject_mode(self):
        args = build_parser().parse_args([
            '--input_dir', 'data',
            '--output_dir', 'results',
            '--freeze_frame', '125',
            '--multi_subject_mode', 'all_components',
        ])

        self.assertEqual(args.multi_subject_mode, 'all_components')

    def test_config_multi_subject_mode_can_be_overridden_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'slicer.json'
            config_path.write_text(json.dumps({
                'input_dir': 'config-input',
                'output_dir': 'config-output',
                'freeze_frame': 125,
                'multi_subject_mode': 'all_components',
            }), encoding='utf-8')

            args = build_parser().parse_args([
                '--config', str(config_path),
                '--multi_subject_mode', 'largest_component',
            ])

            self.assertEqual(args.multi_subject_mode, 'largest_component')

    def test_config_accepts_all_components_multi_subject_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'slicer.json'
            config_path.write_text(json.dumps({
                'input_dir': 'config-input',
                'output_dir': 'config-output',
                'freeze_frame': 125,
                'multi_subject_mode': 'all_components',
            }), encoding='utf-8')

            args = build_parser().parse_args(['--config', str(config_path)])

            self.assertEqual(args.multi_subject_mode, 'all_components')

    def test_cli_rejects_unknown_multi_subject_mode(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                '--input_dir', 'data',
                '--output_dir', 'results',
                '--freeze_frame', '125',
                '--multi_subject_mode', 'unknown',
            ])

    def test_config_supplies_arguments_and_cli_overrides_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'slicer.json'
            config_path.write_text(json.dumps({
                'input_dir': 'config-input',
                'output_dir': 'config-output',
                'freeze_frame': 125,
                'end_frame': 149,
                'fps': 24,
                'rife_uhd': True,
            }), encoding='utf-8')

            args = build_parser().parse_args([
                '--config', str(config_path),
                '--fps', '30',
                '--no-rife_uhd',
            ])

            self.assertEqual(args.input_dir, 'config-input')
            self.assertEqual(args.output_dir, 'config-output')
            self.assertEqual(args.freeze_frame, 125)
            self.assertEqual(args.end_frame, 149)
            self.assertEqual(args.fps, 30)
            self.assertFalse(args.rife_uhd)

    def test_config_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'slicer.json'
            config_path.write_text(json.dumps({
                'input_dir': 'input',
                'output_dir': 'output',
                'freeze_frame': 125,
                'end_frame': 149,
                'fpss': 24,
            }), encoding='utf-8')

            with self.assertRaises(SystemExit):
                build_parser().parse_args(['--config', str(config_path)])

    def test_sparse_frame_index_ignores_non_frame_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dirname in ['0001', '0002', '0125', '重命名数据']:
                (root / dirname).mkdir()
            (root / '0001' / '001.jpg').write_bytes(b'')
            (root / '0002' / '001.jpg').write_bytes(b'')
            (root / '0125' / '001.jpg').write_bytes(b'')
            (root / '0125' / '002.jpg').write_bytes(b'')
            (root / '重命名数据' / '001.jpg').write_bytes(b'')
            (root / '重命名数据' / '002.jpg').write_bytes(b'')

            slicer = SpacetimeSlicer(str(root), str(root / 'out'), camera_ids=[1, 2])

            self.assertEqual(slicer.total_frames, 125)
            self.assertEqual(len(slicer.frame_paths_dict[1]), 3)
            self.assertEqual(len(slicer.frame_paths_dict[2]), 1)
            self.assertTrue(slicer.has_frame(124, 2))
            self.assertFalse(slicer.has_frame(125, 2))

    def test_tail_camera_defaults_to_first_camera_when_sparse_views_have_no_tail(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.frame_paths_dict = {
            1: list(range(149)),
            90: [124],
        }

        self.assertEqual(slicer.resolve_tail_camera_id([1, 90], 124, 149), 1)

    def test_tail_camera_defaults_to_last_camera_when_it_has_tail_frames(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.frame_paths_dict = {
            1: list(range(149)),
            90: list(range(149)),
        }

        self.assertEqual(slicer.resolve_tail_camera_id([1, 90], 124, 149), 90)

    def test_parse_frame_ids_accepts_ranges_and_lists(self):
        self.assertEqual(parse_frame_ids('115:117'), [115, 116, 117])
        self.assertEqual(parse_frame_ids('115,130'), [115, 130])

    def test_source_clips_ghost_alpha_where_current_subject_is_present(self):
        frames = [
            np.full((1, 2, 3), 100, dtype=np.uint8),
            np.full((1, 2, 3), 200, dtype=np.uint8),
        ]
        first_alpha = np.zeros((1, 2), dtype=np.uint8)
        first_alpha[0, 0] = 255
        second_alpha = np.zeros((1, 2), dtype=np.uint8)
        second_alpha[0, 1] = 255
        alphas = [
            first_alpha,
            second_alpha,
        ]
        slicer = make_slicer(frames)
        all_ghosts = []
        permanent_indices = []

        ghost_count, last_effect_frame = slicer.process_segment(
            SequenceAlphaStrategy(alphas),
            0, 0, 2, 2, 0, all_ghosts, permanent_indices, FrameCollector(),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
            effect_base_mode='source',
            live_subject_protect_dilate=0,
        )

        self.assertEqual(ghost_count, 1)
        self.assertEqual(permanent_indices, [0])
        self.assertEqual(last_effect_frame[0, :, 0].tolist(), [175, 200])

    def test_source_runs_segmentation_on_every_frame_but_only_keeps_slice_frames(self):
        frames = [
            np.full((1, 1, 3), 10, dtype=np.uint8),
            np.full((1, 1, 3), 20, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        strategy = ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8))
        out = FrameCollector()
        all_ghosts = []
        permanent_indices = []

        slicer.process_segment(
            strategy,
            0, 0, 3, 2, 0, all_ghosts, permanent_indices, out,
            ghost_opacity_start=0.5,
            ghost_opacity_end=0.5,
            effect_base_mode='source',
        )

        self.assertEqual(strategy.calls, [0, 1, 2])
        self.assertEqual(len(all_ghosts), 3)
        self.assertEqual(permanent_indices, [0, 2])
        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [10, 20, 30])

    def test_source_non_slice_alpha_does_not_replace_original_frame(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
            np.full((1, 1, 3), 220, dtype=np.uint8),
        ]
        alphas = [
            np.zeros((1, 1), dtype=np.uint8),
            np.full((1, 1), 255, dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        out = FrameCollector()

        slicer.process_segment(
            SequenceAlphaStrategy(alphas),
            0, 0, 3, 2, 0, [], [], out,
            ghost_opacity_start=1.0,
            ghost_opacity_end=1.0,
            effect_base_mode='source',
        )

        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in out.frames],
            [100, 200, 220],
        )

    def test_source_dense_samples_move_recovery_to_final_non_slice_frame(self):
        frames = [
            np.full((1, 5, 3), value, dtype=np.uint8)
            for value in (80, 120, 160, 200)
        ]
        alphas = []
        for position in range(4):
            alpha = np.zeros((1, 5), dtype=np.uint8)
            alpha[0, position] = 255
            alphas.append(alpha)

        slicer = make_slicer(frames)
        all_ghosts = []
        permanent_indices = []
        slicer.process_segment(
            SequenceAlphaStrategy(alphas),
            0, 0, 4, 2, 0, all_ghosts, permanent_indices, FrameCollector(),
            ghost_opacity_start=1.0,
            ghost_opacity_end=1.0,
            effect_base_mode='source',
        )

        self.assertEqual(len(all_ghosts), 4)
        self.assertEqual(permanent_indices, [0, 2])

        recovery_out = FrameCollector()
        slicer.process_fade_out(
            recovery_out,
            all_ghosts,
            permanent_indices,
            np.zeros((1, 5, 3), dtype=np.uint8),
            total_fade_frames=3,
        )

        self.assertEqual(
            recovery_out.frames[-1][0, :, 0].tolist(),
            [0, 0, 0, 200, 0],
        )

    def test_source_tracks_hidden_frames_without_writing_or_capturing_them(self):
        frames = [
            np.full((1, 1, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40)
        ]
        slicer = make_slicer(frames)
        strategy = ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8))
        out = FrameCollector()
        all_ghosts = []
        permanent_indices = []

        _, last_effect_frame = slicer.process_segment(
            strategy,
            0, 0, 2, 1, 0, all_ghosts, permanent_indices, out,
            ghost_opacity_start=0.0,
            ghost_opacity_end=0.0,
            effect_base_mode='source',
            tracking_end_idx=4,
        )

        self.assertEqual(strategy.calls, [0, 1, 2, 3])
        self.assertEqual(len(all_ghosts), 4)
        self.assertEqual(permanent_indices, [0, 1])
        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [10, 20])
        self.assertEqual(int(last_effect_frame[0, 0, 0]), 20)

    def test_patched_canvas_runs_segmentation_on_every_frame(self):
        frames = [
            np.full((1, 1, 3), 10, dtype=np.uint8),
            np.full((1, 1, 3), 20, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        strategy = ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8))

        slicer.process_segment(
            strategy,
            0, 0, 3, 2, 0, [], [], FrameCollector(),
            ghost_opacity_start=0.5,
            ghost_opacity_end=0.5,
        )

        self.assertEqual(strategy.calls, [0, 1, 2])

    def test_source_frame_remains_base_when_first_slice_opacity_is_zero(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        out = FrameCollector()

        slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0, 0, 2, 1, 0, [], [], out,
            ghost_opacity_start=0.0,
            ghost_opacity_end=0.0,
            effect_base_mode='source',
        )

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [100, 200])

    def test_source_ignores_initial_patch_and_keeps_original_start_subject(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        out = FrameCollector()

        slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0, 0, 2, 2, 0, [], [], out,
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
            initial_subject_replacement=np.zeros((1, 1, 3), dtype=np.uint8),
            effect_base_mode='source',
        )

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [100, 200])

    def test_source_subject_protection_dilation_expands_the_no_ghost_region(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        background = np.full((1, 3, 3), 200, dtype=np.uint8)
        ghost_alpha = np.zeros((1, 3), dtype=np.uint8)
        ghost_alpha[0, 0] = 255
        live_alpha = np.zeros((1, 3), dtype=np.uint8)
        live_alpha[0, 1] = 255
        all_ghosts = [{
            'frame': np.full((1, 3, 3), 100, dtype=np.uint8),
            'alpha': ghost_alpha,
            'opacity': 1.0,
        }]

        without_dilation = slicer.compose_static_ghosts(
            background,
            all_ghosts,
            [0],
            live_subject_alpha=live_alpha,
            live_subject_protect_dilate=0,
        )
        with_dilation = slicer.compose_static_ghosts(
            background,
            all_ghosts,
            [0],
            live_subject_alpha=live_alpha,
            live_subject_protect_dilate=1,
        )

        self.assertEqual(without_dilation[0, :, 0].tolist(), [100, 200, 200])
        self.assertEqual(with_dilation[0, :, 0].tolist(), [200, 200, 200])

    def test_manual_patch_frame_defaults_to_freeze_frame(self):
        frames = [
            np.full((1, 1, 3), 10, dtype=np.uint8),
            np.full((1, 1, 3), 20, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)

        replacement = slicer.resolve_initial_subject_replacement('frame', 0, 1)

        self.assertEqual(int(replacement[0, 0, 0]), 20)

    def test_manual_patch_frame_uses_complete_original_image_sequence(self):
        timeline_frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(149)]
        source_frames = [
            np.full((1, 1, 3), image_number, dtype=np.uint8)
            for image_number in range(1, 240)
        ]
        slicer = make_slicer(timeline_frames, source_frames=source_frames)

        replacement = slicer.resolve_initial_subject_replacement(
            'frame',
            camera_id=0,
            freeze_idx=124,
            patch_frame_idx=199,
        )

        self.assertEqual(len(slicer.source_image_paths), 239)
        self.assertEqual(int(replacement[0, 0, 0]), 200)

    def test_manual_patch_frame_reports_original_sequence_range(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.source_image_paths = list(range(239))

        with self.assertRaisesRegex(ValueError, 'between 1 and 239'):
            slicer.read_source_image(239)

    def test_recovery_trajectory_uses_fractional_smoothstep_positions(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)

        trajectories = slicer.build_recovery_trajectories([0], 2, 5)

        self.assertEqual(trajectories[0], [0.0, 0.3125, 1.0, 1.6875, 2.0])

    def test_recovery_keeps_created_ghost_opacity_while_moving(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'opacity': 0.25},
            {'frame': np.full((1, 1, 3), 200, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
            {'frame': np.full((1, 1, 3), 240, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
        ]
        out = FrameCollector()

        slicer.process_fade_out(out, all_ghosts, [0], np.zeros((1, 1, 3), dtype=np.uint8), 3)

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [25, 50, 60])

    def test_recovery_subject_protection_keeps_freeze_person_above_ghosts(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        first_alpha = np.zeros((1, 3), dtype=np.uint8)
        first_alpha[0, 0] = 255
        freeze_alpha = np.zeros((1, 3), dtype=np.uint8)
        freeze_alpha[0, 1] = 255
        all_ghosts = [
            {
                'frame': np.full((1, 3, 3), 100, dtype=np.uint8),
                'alpha': first_alpha,
                'opacity': 1.0,
            },
            {
                'frame': np.full((1, 3, 3), 50, dtype=np.uint8),
                'alpha': freeze_alpha,
                'opacity': 1.0,
            },
        ]
        background = np.full((1, 3, 3), 200, dtype=np.uint8)
        out = FrameCollector()

        slicer.process_fade_out(
            out,
            all_ghosts,
            [0],
            background,
            total_fade_frames=2,
            subject_protection_alpha=freeze_alpha,
            live_subject_protect_dilate=0,
        )

        self.assertEqual(out.frames[0][0, :, 0].tolist(), [100, 200, 200])
        self.assertEqual(out.frames[-1][0, :, 0].tolist(), [200, 200, 200])

    def test_recovery_cutout_aligns_subject_before_blending(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        first_alpha = np.zeros((1, 5), dtype=np.uint8)
        first_alpha[0, 1] = 255
        second_alpha = np.zeros((1, 5), dtype=np.uint8)
        second_alpha[0, 3] = 255
        all_ghosts = [
            {'frame': np.full((1, 5, 3), 100, dtype=np.uint8), 'alpha': first_alpha},
            {'frame': np.full((1, 5, 3), 200, dtype=np.uint8), 'alpha': second_alpha},
        ]

        ghost = slicer.interpolate_ghost(all_ghosts, 0.5)

        self.assertEqual(ghost['alpha'][0].tolist(), [0, 0, 255, 0, 0])
        self.assertEqual(int(ghost['frame'][0, 2, 0]), 150)

    def test_recovery_preserves_each_cutout_aspect_ratio(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        vertical_alpha = np.zeros((9, 9), dtype=np.uint8)
        vertical_alpha[2:7, 4] = 255
        horizontal_alpha = np.zeros((9, 9), dtype=np.uint8)
        horizontal_alpha[4, 1:8] = 255
        all_ghosts = [
            {
                'frame': np.full((9, 9, 3), 100, dtype=np.uint8),
                'alpha': vertical_alpha,
            },
            {
                'frame': np.full((9, 9, 3), 200, dtype=np.uint8),
                'alpha': horizontal_alpha,
            },
        ]

        ghost = slicer.interpolate_ghost(all_ghosts, 0.5)
        x, y, width, height = cv2.boundingRect(
            (ghost['alpha'] > 0).astype(np.uint8)
        )

        self.assertEqual((x, y, width, height), (1, 2, 7, 5))

    def test_recovery_geometry_ignores_disconnected_alpha_noise(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[2:8, 3:7] = 255
        alpha[5, 19] = 255
        ghost = {
            'frame': np.zeros((10, 20, 3), dtype=np.uint8),
            'alpha': alpha,
        }

        geometry = slicer.get_ghost_geometry(ghost)

        np.testing.assert_array_equal(
            geometry,
            np.array([5.0, 5.0, 4.0, 6.0], dtype=np.float32),
        )

    def test_all_components_geometry_uses_union_bbox_and_full_alpha_centroid(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.multi_subject_mode = 'all_components'
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[2:8, 2:6] = 255
        alpha[3:7, 14:17] = 255
        ghost = {
            'frame': np.zeros((10, 20, 3), dtype=np.uint8),
            'alpha': alpha,
        }

        slicer.use_centroid = False
        bbox_geometry = slicer.get_ghost_geometry(ghost)
        np.testing.assert_array_equal(
            bbox_geometry,
            np.array([9.5, 5.0, 15.0, 6.0], dtype=np.float32),
        )

        slicer.use_centroid = True
        centroid_geometry = slicer.get_ghost_geometry(ghost)
        np.testing.assert_allclose(
            centroid_geometry,
            np.array([22.0 / 3.0, 4.5, 15.0, 6.0], dtype=np.float32),
        )

    def test_geometry_cache_is_isolated_by_multi_subject_mode(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.use_centroid = False
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[2:8, 2:6] = 255
        alpha[3:7, 14:17] = 255
        ghost = {
            'frame': np.zeros((10, 20, 3), dtype=np.uint8),
            'alpha': alpha,
        }

        slicer.multi_subject_mode = 'largest_component'
        largest_geometry = slicer.get_ghost_geometry(ghost)
        slicer.multi_subject_mode = 'all_components'
        union_geometry = slicer.get_ghost_geometry(ghost)

        np.testing.assert_array_equal(
            largest_geometry,
            np.array([4.0, 5.0, 4.0, 6.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            union_geometry,
            np.array([9.5, 5.0, 15.0, 6.0], dtype=np.float32),
        )

    def test_single_subject_geometry_is_identical_in_both_modes(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.use_centroid = True
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[2:8, 3:7] = 255
        ghost = {
            'frame': np.zeros((10, 20, 3), dtype=np.uint8),
            'alpha': alpha,
        }

        slicer.multi_subject_mode = 'largest_component'
        largest_layout = slicer.get_ghost_layout(ghost)
        slicer.multi_subject_mode = 'all_components'
        union_layout = slicer.get_ghost_layout(ghost)

        np.testing.assert_array_equal(
            largest_layout['geometry'], union_layout['geometry']
        )
        self.assertEqual(largest_layout['bbox'], union_layout['bbox'])

    def test_all_components_preserves_secondary_subject_during_fractional_recovery(self):
        primary_alpha = np.zeros((10, 20), dtype=np.uint8)
        primary_alpha[2:8, 2:6] = 255
        primary_alpha[3:7, 14:17] = 255
        next_alpha = np.zeros((10, 20), dtype=np.uint8)
        next_alpha[2:8, 3:7] = 255
        next_alpha[3:7, 15:18] = 255
        all_ghosts = [
            {
                'frame': np.full((10, 20, 3), 100, dtype=np.uint8),
                'alpha': primary_alpha,
            },
            {
                'frame': np.full((10, 20, 3), 200, dtype=np.uint8),
                'alpha': next_alpha,
            },
        ]
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.use_centroid = False

        slicer.multi_subject_mode = 'largest_component'
        largest_ghost = slicer.interpolate_ghost(all_ghosts, 0.5)
        largest_count, _ = cv2.connectedComponents(
            (largest_ghost['alpha'] > 0).astype(np.uint8)
        )

        slicer.multi_subject_mode = 'all_components'
        union_ghost = slicer.interpolate_ghost(all_ghosts, 0.5)
        union_count, _ = cv2.connectedComponents(
            (union_ghost['alpha'] > 0).astype(np.uint8)
        )

        self.assertEqual(largest_count, 2)
        self.assertEqual(union_count, 3)
        self.assertGreater(np.count_nonzero(union_ghost['alpha'][:, 12:]), 0)

    def test_temporal_median_background_removes_transient_subject(self):
        frames = [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
        ]

        background = make_slicer(frames).build_temporal_median_background(0)

        self.assertTrue(np.array_equal(background, np.zeros((1, 1, 3), dtype=np.uint8)))

    def test_effect_schedule_defaults_to_recovery_after_freeze_frame(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        slice_end_idx, total_fade_frames = slicer.resolve_effect_schedule(115, 227, None)

        self.assertEqual(slice_end_idx, 227)
        self.assertEqual(total_fade_frames, 12)

    def test_effect_schedule_can_reserve_recovery_before_freeze_frame(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        slice_end_idx, total_fade_frames = slicer.resolve_effect_schedule(
            115, 227, None, recovery_timing='before_freeze'
        )

        self.assertEqual(slice_end_idx, 215)
        self.assertEqual(total_fade_frames, 12)

    def test_effect_schedule_before_freeze_uses_explicit_recovery_duration(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        slice_end_idx, total_fade_frames = slicer.resolve_effect_schedule(
            115, 227, 20, recovery_timing='before_freeze'
        )

        self.assertEqual(slice_end_idx, 207)
        self.assertEqual(total_fade_frames, 20)

    def test_effect_schedule_after_freeze_uses_explicit_recovery_duration(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        slice_end_idx, total_fade_frames = slicer.resolve_effect_schedule(115, 227, 20)

        self.assertEqual(slice_end_idx, 227)
        self.assertEqual(total_fade_frames, 20)

    def test_effect_schedule_before_freeze_rejects_recovery_that_leaves_no_slice_frames(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        with self.assertRaisesRegex(ValueError, 'slice generation'):
            slicer.resolve_effect_schedule(115, 227, 113, recovery_timing='before_freeze')

    def test_slice_generation_uses_rife_for_person_action(self):
        frames = [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.full((1, 1, 3), 90, dtype=np.uint8),
        ]
        rife = FakeRifeInterpolator()
        slicer = make_slicer(frames, rife)
        out = FrameCollector()

        slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0, 0, 2, 1, 0, [], [], out,
            stretch_ghost=3,
        )

        self.assertEqual(rife.calls, [1 / 3, 2 / 3])
        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [0, 30, 60, 90])

    def test_freeze_orbit_uses_rife_for_global_scene(self):
        frames = {
            0: np.zeros((1, 1, 3), dtype=np.uint8),
            1: np.full((1, 1, 3), 100, dtype=np.uint8),
            2: np.full((1, 1, 3), 200, dtype=np.uint8),
        }
        rife = FakeRifeInterpolator()
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.rife_interpolator = rife
        slicer.read_frame = lambda idx, camera_id=None: frames[camera_id].copy()
        out = FrameCollector()

        slicer.process_freeze_transition([0, 1, 2], 0, out, stretch_freeze=2)

        self.assertEqual(rife.calls, [0.5, 0.5])
        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [0, 50, 100, 150, 200])

    def test_freeze_orbit_repeat_uses_only_real_camera_frames(self):
        frames = {
            0: np.zeros((1, 1, 3), dtype=np.uint8),
            1: np.full((1, 1, 3), 100, dtype=np.uint8),
            2: np.full((1, 1, 3), 200, dtype=np.uint8),
        }
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.rife_interpolator = None
        slicer.read_frame = lambda idx, camera_id=None: frames[camera_id].copy()
        out = FrameCollector()

        slicer.process_freeze_transition(
            [0, 1, 2], 0, out, stretch_freeze=2, interpolation_mode='repeat'
        )

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [0, 0, 100, 100, 200, 200])

    def test_freeze_orbit_blend_crossfades_real_camera_frames(self):
        frames = {
            0: np.zeros((1, 1, 3), dtype=np.uint8),
            1: np.full((1, 1, 3), 100, dtype=np.uint8),
            2: np.full((1, 1, 3), 200, dtype=np.uint8),
        }
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.rife_interpolator = None
        slicer.read_frame = lambda idx, camera_id=None: frames[camera_id].copy()
        out = FrameCollector()

        slicer.process_freeze_transition(
            [0, 1, 2], 0, out, stretch_freeze=2, interpolation_mode='blend'
        )

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [0, 50, 100, 150, 200])

    def test_rife_is_required_for_interpolated_stages(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.rife_interpolator = None

        with self.assertRaisesRegex(ValueError, 'RIFE'):
            slicer.create_rife_writer(FrameCollector(), 2, 0, 1)

    def test_generate_rejects_invalid_ghost_interval_before_writing_video(self):
        slicer = make_slicer([np.zeros((1, 1, 3), dtype=np.uint8)])

        with self.assertRaisesRegex(ValueError, 'ghost_interval'):
            slicer.generate(None, 0, 0, 1, ghost_interval=0)

    def test_generate_rejects_unknown_multi_subject_mode_before_writing_video(self):
        slicer = make_slicer([np.zeros((1, 1, 3), dtype=np.uint8)])

        with self.assertRaisesRegex(ValueError, 'multi-subject mode'):
            slicer.generate(None, 0, 0, 1, multi_subject_mode='unknown')


if __name__ == '__main__':
    unittest.main()
