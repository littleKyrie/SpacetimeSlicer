import json
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from build_spacetime_slicer import build_parser, normalize_cli_frame_args, parse_frame_ids
from models.spacetime_slicer import SpacetimeSlicer


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


class FakeRifeInterpolator:
    def __init__(self):
        self.calls = []

    def interpolate(self, first_frame, second_frame, timestep):
        self.calls.append(timestep)
        return cv2.addWeighted(first_frame, 1.0 - timestep, second_frame, timestep, 0)


def make_slicer(frames, rife_interpolator=None):
    slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
    slicer.camera_ids = [0]
    slicer.frame_paths_dict = {0: list(range(len(frames)))}
    slicer.read_frame = lambda idx, camera_id=None: frames[idx].copy()
    slicer.rife_interpolator = rife_interpolator
    slicer.fps = 25
    return slicer


class SpacetimeSlicerTest(unittest.TestCase):
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

        self.assertEqual(args.start_frame, 24)
        self.assertEqual(args.freeze_frame, 124)
        self.assertEqual(args.end_frame, 149)
        self.assertEqual(args.initial_subject_patch_frame, 124)
        self.assertEqual(args.source_start_frame, 25)
        self.assertEqual(args.source_freeze_frame, 125)
        self.assertEqual(args.source_end_frame, 149)

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

    def test_segment_blends_ghost_with_effective_alpha(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        all_ghosts = []
        permanent_indices = []

        ghost_count, last_effect_frame = slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0, 0, 2, 1, 0, all_ghosts, permanent_indices, FrameCollector(),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
            effect_base_mode='source',
        )

        self.assertEqual(ghost_count, 2)
        self.assertEqual(permanent_indices, [0, 1])
        self.assertTrue(np.array_equal(last_effect_frame, np.full((1, 1, 3), 181, dtype=np.uint8)))

    def test_segment_only_runs_segmentation_on_slice_frames(self):
        frames = [
            np.full((1, 1, 3), 10, dtype=np.uint8),
            np.full((1, 1, 3), 20, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        strategy = ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8))
        out = FrameCollector()

        slicer.process_segment(
            strategy,
            0, 0, 3, 2, 0, [], [], out,
            ghost_opacity_start=0.5,
            ghost_opacity_end=0.5,
            effect_base_mode='source',
        )

        self.assertEqual(strategy.calls, [0, 2])
        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [10, 15, 25])

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

    def test_initial_patch_only_replaces_start_frame_base_before_first_slice(self):
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

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [25, 175])

    def test_manual_patch_frame_defaults_to_freeze_frame(self):
        frames = [
            np.full((1, 1, 3), 10, dtype=np.uint8),
            np.full((1, 1, 3), 20, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)

        replacement = slicer.resolve_initial_subject_replacement('frame', 0, 1)

        self.assertEqual(int(replacement[0, 0, 0]), 20)

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


if __name__ == '__main__':
    unittest.main()
