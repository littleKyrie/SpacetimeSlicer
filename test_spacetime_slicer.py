import unittest
import tempfile
from unittest import mock

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


class CameraAwareAlphaStrategy:
    def __init__(self):
        self.calls = []

    def process_frame(self, current_frame, current_idx):
        first_value = int(current_frame[0, 0, 0])
        self.calls.append((current_idx, first_value))
        if first_value >= 200:
            return np.array([[0, 255]], dtype=np.uint8)
        return np.array([[255, 0]], dtype=np.uint8)


class FakeRifeInterpolator:
    def __init__(self):
        self.calls = []

    def interpolate(self, first_frame, second_frame, timestep):
        self.calls.append(timestep)
        return cv2.addWeighted(first_frame, 1.0 - timestep, second_frame, timestep, 0)


class FakeVideoWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def make_slicer(frames, rife_interpolator=None):
    slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
    slicer.camera_ids = [0]
    slicer.frame_paths_dict = {0: list(range(len(frames)))}
    slicer.read_frame = lambda idx, camera_id=None: frames[idx].copy()
    slicer.rife_interpolator = rife_interpolator
    slicer.fps = 25
    return slicer


class SpacetimeSlicerTest(unittest.TestCase):
    def test_segment_composes_opaque_slices(self):
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
            cutout_edge_blur=0,
        )

        self.assertEqual(ghost_count, 2)
        self.assertEqual(permanent_indices, [0, 1])
        self.assertTrue(np.array_equal(last_effect_frame, np.full((1, 1, 3), 200, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas, np.full((1, 1, 3), 200, dtype=np.uint8)))

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
            cutout_edge_blur=0,
        )

        self.assertTrue(np.array_equal(canvas[0, 0], np.full(3, 100, dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[0, 1], np.full(3, 50, dtype=np.uint8)))

    def test_recovery_trajectory_uses_fractional_smoothstep_positions(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)

        trajectories = slicer.build_recovery_trajectories([0], 2, 5)

        self.assertEqual(trajectories[0], [0.0, 0.3125, 1.0, 1.6875, 2.0])

    def test_recovery_composes_opaque_slices_while_moving(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'opacity': 0.25},
            {'frame': np.full((1, 1, 3), 200, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
            {'frame': np.full((1, 1, 3), 240, dtype=np.uint8), 'alpha': alpha, 'opacity': 1.0},
        ]
        out = FrameCollector()

        slicer.process_fade_out(out, all_ghosts, [0], np.zeros((1, 1, 3), dtype=np.uint8), 3)

        self.assertEqual([int(frame[0, 0, 0]) for frame in out.frames], [100, 200, 240])

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
            cutout_edge_blur=0,
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

    def test_freeze_orbit_carries_created_slices_in_camera_order(self):
        frames = {
            (0, 0): np.array([[[10, 10, 10], [11, 11, 11]]], dtype=np.uint8),
            (0, 1): np.array([[[20, 20, 20], [21, 21, 21]]], dtype=np.uint8),
            (2, 0): np.array([[[100, 100, 100], [101, 101, 101]]], dtype=np.uint8),
            (2, 1): np.array([[[200, 200, 200], [201, 201, 201]]], dtype=np.uint8),
        }
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.rife_interpolator = None
        slicer.read_frame = lambda idx, camera_id=None: frames[(idx, camera_id)].copy()
        out = FrameCollector()
        strategy = CameraAwareAlphaStrategy()
        alpha = np.array([[255, 0]], dtype=np.uint8)
        all_ghosts = [{
            'frame': frames[(2, 0)].copy(),
            'alpha': alpha,
            'source_idx': 2,
            'camera_id': 0,
            'opacity': 1.0,
        }]

        last_frame = slicer.process_freeze_transition(
            [0, 1], 0, out, interpolation_mode='repeat',
            all_ghosts=all_ghosts, permanent_indices=[0], strategy=strategy,
            cutout_edge_blur=0,
        )

        self.assertEqual([frame[0, :, 0].tolist() for frame in out.frames], [[10, 11], [20, 201]])
        self.assertEqual(last_frame[0, :, 0].tolist(), [20, 201])
        self.assertEqual(strategy.calls, [(0, 10), (2, 200), (0, 20)])

    def test_later_slices_are_composited_above_earlier_slices(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        all_ghosts = [
            {'frame': np.full((1, 1, 3), 100, dtype=np.uint8), 'alpha': alpha, 'source_idx': 1},
            {'frame': np.full((1, 1, 3), 200, dtype=np.uint8), 'alpha': alpha, 'source_idx': 2},
        ]

        canvas = slicer.compose_static_ghosts(
            all_ghosts, [1, 0], np.zeros((1, 1, 3), dtype=np.uint8),
            cutout_edge_blur=0,
        )

        self.assertEqual(int(canvas[0, 0, 0]), 200)

    def test_low_alpha_dark_fringe_is_removed_from_cutout(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        canvas = np.full((1, 2, 3), 255, dtype=np.uint8)
        frame = np.array([[[100, 100, 100], [0, 0, 0]]], dtype=np.uint8)
        alpha = np.array([[255, 32]], dtype=np.uint8)

        out = slicer.compose_opaque_cutout(
            canvas, frame, alpha, alpha_threshold=128, edge_blur=0
        )

        self.assertEqual(out[0, :, 0].tolist(), [100, 255])

    def test_generate_orbits_before_recovery_and_forces_opaque_slices(self):
        class RecordingSlicer(SpacetimeSlicer):
            def __init__(self, output_root):
                self.camera_ids = [0, 1]
                self.frame_paths_dict = {0: list(range(4)), 1: list(range(4))}
                self.output_root = output_root
                self.fps = 25
                self.rife_interpolator = None
                self.events = []
                self.segment_opacity = None
                self.freeze_ghosts = None
                self.fade_transition_value = None
                self.fade_background_value = None

            def read_frame(self, idx, camera_id=None):
                if camera_id is None:
                    camera_id = 0
                return np.full((1, 1, 3), camera_id * 100 + idx, dtype=np.uint8)

            def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval,
                                edge_feather, all_ghosts, permanent_indices, out, **kwargs):
                self.events.append('segment')
                self.segment_opacity = (
                    kwargs['ghost_opacity_start'],
                    kwargs['ghost_opacity_end'],
                )
                all_ghosts.append({
                    'frame': np.full((1, 1, 3), 77, dtype=np.uint8),
                    'alpha': np.full((1, 1), 255, dtype=np.uint8),
                    'opacity': kwargs['ghost_opacity_start'],
                })
                permanent_indices.append(0)
                return (
                    np.full((1, 1, 3), 77, dtype=np.uint8),
                    1,
                    np.full((1, 1, 3), 78, dtype=np.uint8),
                )

            def process_freeze_transition(self, camera_ids, freeze_idx, out, **kwargs):
                self.events.append('freeze')
                self.freeze_ghosts = (
                    len(kwargs['all_ghosts']),
                    list(kwargs['permanent_indices']),
                    list(camera_ids),
                )
                return np.full((1, 1, 3), 88, dtype=np.uint8)

            def process_fade_out(self, out, all_ghosts, permanent_indices, background,
                                 total_fade_frames, **kwargs):
                self.events.append('fade')
                self.fade_transition_value = int(kwargs['transition_from'][0, 0, 0])
                self.fade_background_value = int(background[0, 0, 0])

        fake_writer = FakeVideoWriter()
        with tempfile.TemporaryDirectory() as output_root:
            slicer = RecordingSlicer(output_root)
            with mock.patch('models.spacetime_slicer.cv2.VideoWriter', return_value=fake_writer), \
                    mock.patch('models.spacetime_slicer.os.path.isfile', return_value=True), \
                    mock.patch('models.spacetime_slicer.os.path.getsize', return_value=1):
                slicer.generate(
                    ConstantAlphaStrategy(np.full((1, 1), 255, dtype=np.uint8)),
                    1, 2, 4,
                    camera_ids=[0, 1],
                    ghost_opacity_start=0.2,
                    ghost_opacity_end=0.4,
                    initial_subject_patch_mode='freeze',
                    background_mode='freeze',
                    freeze_interp_mode='repeat',
                )

        self.assertEqual(slicer.events, ['segment', 'freeze', 'fade'])
        self.assertEqual(slicer.segment_opacity, (1.0, 1.0))
        self.assertEqual(slicer.freeze_ghosts, (1, [0], [0, 1]))
        self.assertEqual(slicer.fade_transition_value, 88)
        self.assertEqual(slicer.fade_background_value, 102)

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
