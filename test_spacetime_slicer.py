import unittest
from unittest.mock import patch

import numpy as np

from build_spacetime_slicer import SpacetimeSlicer, StretchedFrameWriter


class FrameCollector:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class ConstantAlphaStrategy:
    def __init__(self, alpha):
        self.alpha = alpha

    def process_frame(self, current_frame, current_idx):
        return self.alpha.copy()


def make_slicer(frames):
    slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
    slicer.camera_ids = [0]
    slicer.frame_paths_dict = {0: list(range(len(frames)))}
    slicer.read_frame = lambda idx, camera_id=None: frames[idx].copy()
    return slicer


class SpacetimeSlicerTest(unittest.TestCase):
    def test_segment_blends_ghost_with_effective_alpha(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        out = FrameCollector()
        all_ghosts = []
        permanent_indices = []

        canvas, ghost_count, last_effect_frame = slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0,
            0,
            2,
            1,
            0,
            all_ghosts,
            permanent_indices,
            out,
            initial_canvas=frames[0],
            initial_subject_replacement=np.zeros((1, 1, 3), dtype=np.uint8),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
        )

        self.assertEqual(ghost_count, 2)
        self.assertEqual(permanent_indices, [0, 1])
        self.assertTrue(np.array_equal(last_effect_frame, np.full((1, 1, 3), 200, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas, np.full((1, 1, 3), 68, dtype=np.uint8)))

    def test_fade_keeps_each_created_ghost_opacity_while_moving(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'opacity': 0.25},
            {'frame': np.full((1, 1, 3), 200, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
            {'frame': np.full((1, 1, 3), 240, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
        ]

        slicer.process_fade_out(
            out,
            all_ghosts,
            [0],
            np.zeros((1, 1, 3), dtype=np.uint8),
            total_fade_frames=3,
        )

        values = [int(frame[0, 0, 0]) for frame in out.frames]
        self.assertEqual(values, [25, 50, 60])

    def test_initial_subject_patch_keeps_start_frame_background_pixels(self):
        frames = [np.array([[[100, 100, 100], [50, 50, 50]]], dtype=np.uint8)]
        slicer = make_slicer(frames)
        out = FrameCollector()

        canvas, _, _ = slicer.process_segment(
            ConstantAlphaStrategy(np.array([[255, 0]], dtype=np.uint8)),
            0,
            0,
            1,
            1,
            0,
            [],
            [],
            out,
            initial_canvas=frames[0],
            initial_subject_replacement=np.zeros((1, 2, 3), dtype=np.uint8),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
        )

        self.assertTrue(np.array_equal(canvas[0, 0], np.full(3, 25, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[0, 1], np.full(3, 50, dtype=np.uint8)))

    def test_fade_keeps_initial_ghost_translucent_during_recovery(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'opacity': 0.25},
            {'frame': np.full((1, 1, 3), 200, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
        ]

        slicer.process_fade_out(
            out,
            all_ghosts,
            [0],
            np.zeros((1, 1, 3), dtype=np.uint8),
            total_fade_frames=2,
        )

        values = [int(frame[0, 0, 0]) for frame in out.frames]
        self.assertEqual(values, [25, 50])

    def test_recovery_canvas_switch_uses_crossfade(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()

        slicer.write_canvas_transition(
            out,
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.full((1, 1, 3), 120, dtype=np.uint8),
            transition_frames=3,
        )

        values = [int(frame[0, 0, 0]) for frame in out.frames]
        self.assertEqual(values, [30, 60, 90])

    def test_recovery_trajectory_uses_fractional_smoothstep_positions(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)

        trajectories = slicer.build_recovery_trajectories([0], 2, 5)

        self.assertEqual(trajectories[0], [0.0, 0.3125, 1.0, 1.6875, 2.0])

    def test_recovery_interpolates_frame_and_alpha_between_source_frames(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        all_ghosts = [
            {
                'frame': np.zeros((1, 1, 3), dtype=np.uint8),
                'alpha': np.zeros((1, 1), dtype=np.uint8),
            },
            {
                'frame': np.full((1, 1, 3), 100, dtype=np.uint8),
                'alpha': np.full((1, 1), 200, dtype=np.uint8),
            },
        ]

        ghost = slicer.interpolate_ghost(all_ghosts, 0.25)

        self.assertTrue(np.array_equal(ghost['frame'], np.full((1, 1, 3), 25, dtype=np.uint8)))
        self.assertTrue(np.array_equal(ghost['alpha'], np.full((1, 1), 50, dtype=np.uint8)))

    def test_default_recovery_duration_is_not_a_two_frame_jump(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.fps = 25

        self.assertEqual(slicer.resolve_fade_duration(4, None), 12)
        self.assertEqual(slicer.resolve_fade_duration(30, None), 30)
        self.assertEqual(slicer.resolve_fade_duration(30, 8), 8)

    def test_fade_stretch_adds_continuous_trajectory_frames_instead_of_repeats(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.zeros((1, 1, 3), dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
        ]

        slicer.process_fade_out(
            out,
            all_ghosts,
            [0],
            np.zeros((1, 1, 3), dtype=np.uint8),
            total_fade_frames=2,
            stretch_fade=3,
        )

        values = [int(frame[0, 0, 0]) for frame in out.frames]
        self.assertEqual(len(values), 6)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 100)
        self.assertGreater(len(set(values)), 2)

    def test_temporal_median_background_removes_transient_subject(self):
        frames = [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
        ]
        slicer = make_slicer(frames)

        background = slicer.build_temporal_median_background(0)

        self.assertTrue(np.array_equal(background, np.zeros((1, 1, 3), dtype=np.uint8)))

    def test_blend_stretch_inserts_intermediate_frames(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()
        writer = StretchedFrameWriter(slicer, out, stretch=4, mode='blend')
        writer.write(np.zeros((1, 1, 3), dtype=np.uint8))
        writer.write(np.full((1, 1, 3), 100, dtype=np.uint8))
        writer.finish()

        values = [int(frame[0, 0, 0]) for frame in out.frames]
        self.assertEqual(values, [0, 25, 50, 75, 100, 100, 100, 100])

    def test_flow_interpolation_keeps_identical_frames_unchanged(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        frame = np.full((8, 8, 3), 100, dtype=np.uint8)

        interpolated = slicer.interpolate_frame(frame, frame, 0.5, 'flow')

        self.assertTrue(np.array_equal(interpolated, frame))

    def test_flow_transition_computes_flow_once_for_all_inserted_frames(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        out = FrameCollector()
        first_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        second_frame = np.full((8, 8, 3), 100, dtype=np.uint8)

        with patch.object(slicer, 'prepare_interpolation', wraps=slicer.prepare_interpolation) as prepare:
            slicer.write_frame_transition(out, first_frame, second_frame, stretch=4, mode='flow')

        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(len(out.frames), 4)

    def test_generate_rejects_invalid_ghost_interval_before_writing_video(self):
        slicer = make_slicer([np.zeros((1, 1, 3), dtype=np.uint8)])

        with self.assertRaisesRegex(ValueError, 'ghost_interval'):
            slicer.generate(None, 0, 0, 1, ghost_interval=0)


if __name__ == '__main__':
    unittest.main()
