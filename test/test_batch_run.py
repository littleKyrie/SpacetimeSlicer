import unittest
from pathlib import Path

from batch_run import parse_args, run_pipeline


class BatchRunTest(unittest.TestCase):
    def test_source_only_uses_same_name_under_current_directory(self):
        args, slicer_args = parse_args(['-s', './data/QP-2026-06-23-175636'])

        self.assertEqual(args.source_dir, './data/QP-2026-06-23-175636')
        self.assertEqual(
            Path(args.output_dir),
            Path.cwd() / 'results' / 'QP-2026-06-23-175636',
        )
        self.assertEqual(slicer_args, [])

    def test_routes_reorganize_and_slicer_overrides(self):
        args, slicer_args = parse_args([
            '-s', './data/source',
            '--output_dir', './custom-output',
            '--pre-frame-count', '100',
            '--camera-count', '30',
            '--fps', '30',
            '--end_frame', '180',
        ])
        calls = {}

        def fake_reorganize(input_dir, **kwargs):
            calls['reorganize'] = (input_dir, kwargs)
            return []

        def fake_slicer(argv):
            calls['slicer'] = argv
            return 0

        result = run_pipeline(
            args,
            slicer_args,
            reorganize_func=fake_reorganize,
            slicer_main=fake_slicer,
        )

        self.assertEqual(result, 0)
        self.assertEqual(calls['reorganize'][1]['pre_frame_count'], 100)
        self.assertEqual(calls['reorganize'][1]['camera_count'], 30)
        self.assertIn('--fps', calls['slicer'])
        self.assertIn('30', calls['slicer'])
        self.assertIn('--end_frame', calls['slicer'])
        self.assertIn('180', calls['slicer'])
        self.assertIn('./custom-output', calls['slicer'])

    def test_invalid_slicer_option_is_rejected_before_reorganization(self):
        args, slicer_args = parse_args([
            '-s', './data/source',
            '--unknown-slicer-option', 'value',
        ])
        calls = []

        def fake_reorganize(input_dir, **kwargs):
            calls.append((input_dir, kwargs))
            return []

        with self.assertRaises(SystemExit):
            run_pipeline(
                args,
                slicer_args,
                reorganize_func=fake_reorganize,
                slicer_main=lambda argv: 0,
            )

        self.assertEqual(calls, [])

    def test_skips_reorganization_when_input_is_already_prepared(self):
        args, slicer_args = parse_args([
            '-s', './data/source',
            '--pre-frame-count', '2',
            '--camera-count', '2',
        ])
        calls = []

        def fake_checker(input_dir, **kwargs):
            self.assertEqual(input_dir, './data/source')
            self.assertEqual(kwargs['pre_frame_count'], 2)
            self.assertEqual(kwargs['camera_count'], 2)
            return True

        def fake_reorganize(input_dir, **kwargs):
            calls.append(('reorganize', input_dir, kwargs))
            return []

        def fake_slicer(argv):
            calls.append(('slicer', argv))
            return 0

        result = run_pipeline(
            args,
            slicer_args,
            reorganize_func=fake_reorganize,
            slicer_main=fake_slicer,
            structure_checker=fake_checker,
        )

        self.assertEqual(result, 0)
        self.assertEqual([call[0] for call in calls], ['slicer'])


if __name__ == '__main__':
    unittest.main()
