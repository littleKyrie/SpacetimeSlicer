import unittest

import cv2
import numpy as np

from models.spacetime_slicer import SpacetimeSlicer


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
    def test_segment_blends_ghost_with_effective_alpha(self):
        frames = [
            np.full((1, 1, 3), 100, dtype=np.uint8),
            np.full((1, 1, 3), 200, dtype=np.uint8),
        ]
        slicer = make_slicer(frames)
        all_ghosts = []
        permanent_indices = []

        canvas, ghost_count, last_effect_frame = slicer.process_segment(
            ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
            0, 0, 2, 1, 0, all_ghosts, permanent_indices, FrameCollector(),
            initial_canvas=frames[0],
            initial_subject_replacement=np.zeros((1, 1, 3), dtype=np.uint8),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
        )

        self.assertEqual(ghost_count, 2)
        self.assertEqual(permanent_indices, [0, 1])
        self.assertTrue(np.array_equal(last_effect_frame, np.full((1, 1, 3), 200, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas, np.full((1, 1, 3), 68, dtype=np.uint8)))

    def test_initial_subject_patch_keeps_background_pixels(self):
        frames = [np.array([[[100, 100, 100], [50, 50, 50]]], dtype=np.uint8)]
        slicer = make_slicer(frames)

        canvas, _, _ = slicer.process_segment(
            ConstantAlphaStrategy(np.array([[255, 0]], dtype=np.uint8)),
            0, 0, 1, 1, 0, [], [], FrameCollector(),
            initial_canvas=frames[0],
            initial_subject_replacement=np.zeros((1, 2, 3), dtype=np.uint8),
            ghost_opacity_start=0.25,
            ghost_opacity_end=0.25,
        )

        self.assertTrue(np.array_equal(canvas[0, 0], np.full(3, 25, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[0, 1], np.full(3, 50, dtype=np.uint8)))

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
            initial_canvas=frames[0],
            initial_subject_replacement=np.zeros((1, 1, 3), dtype=np.uint8),
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
