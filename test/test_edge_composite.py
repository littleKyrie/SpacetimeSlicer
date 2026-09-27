import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from batch_run import build_slicer_argv, parse_args as parse_batch_args
from build_spacetime_slicer import build_parser, normalize_cli_frame_args, save_debug_extractions
from models.edge_composite import (
    cache_ghost, compose_soft_stack, ghost_alpha, ghost_frame, validate_edge_strategy,
)
from models.rvm import RVMStrategy
from models.seg_strategy import MattingResult, SegmentationStrategy
from models.spacetime_slicer import SpacetimeSlicer
from test.test_spacetime_slicer import FrameCollector, SequenceAlphaStrategy, make_slicer


class EdgeCompositeTest(unittest.TestCase):
    def test_default_and_explicit_legacy_match_original_layer_math(self):
        rng = np.random.default_rng(73)
        background = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
        live = np.zeros((8, 9), np.uint8)
        live[2:5, 3:6] = rng.integers(0, 256, (3, 3), dtype=np.uint8)
        ghosts = [dict(frame=rng.integers(0, 256, background.shape, dtype=np.uint8),
                       alpha=rng.integers(0, 256, live.shape, dtype=np.uint8), opacity=o)
                  for o in (0.2, 0.61, 1.0)]
        protection = cv2.dilate((live > 32).astype(np.uint8), np.ones((3, 3), np.uint8))
        expected = background.copy()
        for ghost in ghosts:
            alpha = ghost['alpha'].astype(np.float32) / 255 * ghost['opacity']
            alpha *= 1 - protection
            expected = (ghost['frame'] * alpha[:, :, None] +
                        expected * (1 - alpha[:, :, None])).astype(np.uint8)
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        for explicit in (False, True):
            if explicit:
                slicer.edge_composite_strategy = 'legacy'
            output = slicer.compose_static_ghosts(background, ghosts, [0, 1, 2], live, 32, 1)
            np.testing.assert_array_equal(output, expected)

    def test_soft_protection_applies_once_to_overlapping_layers(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.edge_composite_strategy = 'soft_alpha'
        background = np.zeros((1, 1, 3), np.uint8)
        ghosts = [dict(frame=np.full_like(background, 255), alpha=np.full((1, 1), 255, np.uint8),
                       opacity=0.5) for _ in range(2)]
        result = slicer.compose_static_ghosts(background, ghosts, [0, 1],
                                             np.array([[127.5]], np.float32), 32, 0)
        # Stack coverage=.75, visibility=.5, result=.375*255, not two gated layers.
        np.testing.assert_array_equal(result, np.full_like(background, 96))
        fully_protected = slicer.compose_static_ghosts(background, ghosts, [0, 1],
                                                      np.array([[255]], np.float32), 32, 0)
        np.testing.assert_array_equal(fully_protected, background)

    def test_smoothstep_and_no_protection(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        alpha = np.array([[0, 12.75, 127.5, 242.25, 255]], np.float32)
        slicer.edge_composite_strategy = 'soft_smoothstep'
        np.testing.assert_allclose(slicer.build_subject_protection_mask(alpha, 32, 0),
                                   [[0, 0, 0.5, 1, 1]], atol=1e-6)
        slicer.edge_composite_strategy = 'diagnostic_no_protection'
        self.assertIsNone(slicer.build_subject_protection_mask(alpha, 32, 2))
        bg = np.zeros((1, 1, 3), np.uint8)
        ghost = dict(frame=np.full_like(bg, 123), alpha=np.full((1, 1), 255, np.uint8))
        np.testing.assert_array_equal(slicer.compose_static_ghosts(bg, [ghost], [0], ghost['alpha']),
                                      ghost['frame'])

    def test_roi_cache_preserves_continuous_alpha_and_bgr(self):
        source = np.zeros((20, 30, 3), np.uint8)
        alpha = np.zeros((20, 30), np.float32)
        alpha[8:12, 14:17] = 0.1234
        fg = np.full((20, 30, 3), [0.9, 0.2, 0.1], np.float32)
        ghost = cache_ghost(source, (alpha * 255).astype(np.uint8), 0.4,
                            MattingResult(alpha, fg), soft=True, use_foreground=True)
        self.assertIsNone(ghost['frame'])
        self.assertLess(ghost['foreground_roi'].nbytes, source.nbytes)
        np.testing.assert_allclose(ghost_alpha(ghost), alpha, atol=4e-5)
        np.testing.assert_allclose(ghost_frame(ghost)[9, 15], fg[9, 15] * 255, atol=0.1)
        np.testing.assert_array_equal(ghost_frame(ghost)[0, 0], [0, 0, 0])

    def test_soft_recovery_uses_premultiplied_interpolation_and_one_opacity(self):
        slicer = SpacetimeSlicer.__new__(SpacetimeSlicer)
        slicer.edge_composite_strategy = 'soft_foreground'
        source = np.zeros((4, 4, 3), np.uint8)
        alpha = np.full((4, 4), 0.4, np.float32)
        ghosts = [cache_ghost(source, (alpha * 255).astype(np.uint8), 0.5,
                              MattingResult(alpha, np.full((4, 4, 3), rgb, np.float32)),
                              soft=True, use_foreground=True)
                  for rgb in ([1, 0, 0], [0, 0, 1])]
        mixed = slicer.interpolate_ghost(ghosts, 0.5)
        self.assertIn('premultiplied', mixed)
        result = slicer.compose_recovery_frame(ghosts, [0], {0: [0.5]}, source, 0,
                                               np.full((4, 4), 0.25, np.float32))
        np.testing.assert_array_equal(result[1, 1], [19, 0, 19])

    def test_rvm_returns_float_alpha_and_bgr_from_one_inference(self):
        rvm = RVMStrategy.__new__(RVMStrategy)
        rvm.device = 'cpu'
        rvm.rec = [None] * 4
        alpha = torch.full((1, 1, 2, 3), 0.2371)
        foreground = torch.tensor([0.1, 0.2, 0.9]).view(1, 3, 1, 1).expand(1, 3, 2, 3)
        states = [torch.tensor(i) for i in range(4)]
        rvm.model = Mock(return_value=(foreground, alpha, *states))
        frame = np.zeros((2, 3, 3), np.uint8)
        matte = rvm.process_matting(frame, 0, include_foreground=True)
        self.assertEqual(rvm.model.call_count, 1)
        np.testing.assert_allclose(matte.foreground[0, 0], [0.9, 0.2, 0.1])
        np.testing.assert_allclose(matte.alpha, 0.2371)
        self.assertEqual(rvm.rec, states)
        np.testing.assert_array_equal(rvm.process_frame(frame, 1), np.full((2, 3), 60, np.uint8))
        self.assertEqual(rvm.model.call_count, 2)

    def test_alpha_only_strategy_has_explicit_foreground_failure(self):
        class AlphaOnly(SegmentationStrategy):
            def process_frame(self, frame, idx):
                return np.full(frame.shape[:2], 128, np.uint8)
        strategy = AlphaOnly()
        result = strategy.process_matting(np.zeros((2, 3, 3), np.uint8), 0)
        self.assertIsNone(result.foreground)
        with self.assertRaisesRegex(ValueError, 'foreground'):
            strategy.process_matting(np.zeros((2, 3, 3), np.uint8), 0, True)

    def test_unsupported_strategies_and_modes_fail_before_rendering(self):
        for name, mode, dilate, foreground in (
            ('refined_soft', 'source', 0, True),
            ('soft_alpha', 'patched_canvas', 0, True),
            ('soft_alpha', 'source', 2, True),
            ('foreground_only', 'source', 0, False),
        ):
            with self.subTest(name=name, mode=mode), self.assertRaises(ValueError):
                validate_edge_strategy(name, mode, dilate, foreground)

    def test_config_default_cli_override_and_batch_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'slicer.json'
            config.write_text(json.dumps(dict(input_dir='frames', output_dir='out', freeze_frame=5)),
                               encoding='utf-8')
            self.assertEqual(build_parser().parse_args(['--config', str(config)]).edge_composite_strategy,
                             'legacy')
            config.write_text(json.dumps(dict(input_dir='frames', output_dir='out', freeze_frame=5,
                                              edge_composite_strategy='soft_foreground')), encoding='utf-8')
            self.assertEqual(build_parser().parse_args(['--config', str(config)]).edge_composite_strategy,
                             'soft_foreground')
            args = build_parser().parse_args(['--config', str(config), '--edge_composite_strategy', 'legacy'])
            self.assertEqual(args.edge_composite_strategy, 'legacy')
        args, forwarded = parse_batch_args(['-s', './data/source', '--edge_composite_strategy', 'soft_alpha'])
        argv = build_slicer_argv(args, forwarded, 'frames', 'out')
        self.assertEqual(build_parser().parse_args(argv).edge_composite_strategy, 'soft_alpha')

    def test_debug_uses_sequential_inputs_and_full_render_opacity_schedule(self):
        frames = [np.full((5, 6, 3), i * 30, np.uint8) for i in range(6)]
        alphas = [np.full((5, 6), 40 + 10 * i, np.uint8) for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps(dict(input_dir='frames', output_dir=tmp, camera_ids='0',
                                              start_frame=1, freeze_frame=5, end_frame=6,
                                              ghost_interval=2, effect_base_mode='source',
                                              initial_subject_patch_mode='none',
                                              edge_composite_strategy='soft_alpha',
                                              live_subject_protect_dilate=0,
                                              debug_extract_frames='2,4')), encoding='utf-8')
            args = normalize_cli_frame_args(build_parser().parse_args(['--config', str(config)]))
            strategy = SequenceAlphaStrategy(alphas)
            save_debug_extractions(strategy, make_slicer(frames), args, [0], 6)
            self.assertEqual(strategy.calls, [0, 1, 2, 3])
            directory = next((Path(tmp) / 'debug_extractions').iterdir())
            manifest = json.loads((directory / 'parameters.json').read_text(encoding='utf-8'))
            self.assertEqual([f['source_frame_id'] for f in manifest['frames']], [2, 4])
            self.assertEqual(manifest['frames'][1]['captured_slice_frame_ids'], [1, 3])
            slicer = make_slicer(frames)
            slicer.edge_composite_strategy = 'soft_alpha'
            collector = FrameCollector()
            slicer.process_segment(SequenceAlphaStrategy(alphas), 0, 0, 5, 2, 0, [], [], collector,
                                   effect_base_mode='source', live_subject_protect_dilate=0)
            for source_id in (2, 4):
                png = cv2.imread(str(directory / f'frame{source_id:04d}_cam000_composite_preencode.png'))
                np.testing.assert_array_equal(png, collector.frames[source_id - 1])

    def test_all_strategies_encode_with_consistent_frame_provenance(self):
        class MatteStrategy(SegmentationStrategy):
            supports_foreground = True

            def process_frame(self, frame, idx):
                return (self.alpha(frame) * 255).astype(np.uint8)

            def alpha(self, frame):
                alpha = np.zeros(frame.shape[:2], np.float32)
                alpha[2:-2, 3:-3] = 0.6
                return alpha

            def process_matting(self, frame, idx, include_foreground=False):
                fg = np.full(frame.shape, [0.8, 0.3, 0.1], np.float32) if include_foreground else None
                return MattingResult(self.alpha(frame), fg)

        from models.edge_composite import EDGE_COMPOSITE_STRATEGIES
        from test.test_spacetime_slicer import FakeRifeInterpolator
        frames = [np.full((12, 16, 3), 20 + i * 30, np.uint8) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for name in EDGE_COMPOSITE_STRATEGIES:
                with self.subTest(strategy=name):
                    slicer = make_slicer(frames, FakeRifeInterpolator())
                    slicer.frame_paths_dict[1] = list(range(len(frames)))
                    slicer.output_root = str(Path(tmp) / name)
                    slicer.generate(
                        MatteStrategy(), 0, 2, 5, camera_ids=[0, 1], ghost_interval=2,
                        effect_base_mode='source', initial_subject_patch_mode='none',
                        live_subject_protect_dilate=0, fade_duration_frames=2,
                        recovery_transition_frames=1, stretch_ghost=2, stretch_fade=2,
                        stretch_freeze=2, freeze_interp_mode='blend', stretch_tail=2,
                        edge_composite_strategy=name,
                    )
                    video = next(Path(slicer.output_root).glob('*.mp4'))
                    metadata = json.loads(video.with_suffix('.json').read_text(encoding='utf-8'))
                    cap = cv2.VideoCapture(str(video))
                    decoded = []
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        decoded.append(frame)
                    cap.release()
                    self.assertEqual(len(decoded), 17)
                    self.assertEqual(metadata['expected_output_frames'], len(decoded))
                    self.assertEqual(metadata['edge_composite_strategy'], name)
                    self.assertEqual(metadata['frame_mapping'][1]['source_frame_ids'], [1, 2])
                    self.assertEqual(metadata['frame_mapping'][-1]['source_frame_ids'], [5])
                    self.assertTrue(all(frame.dtype == np.uint8 for frame in decoded))


if __name__ == '__main__':
    unittest.main()
