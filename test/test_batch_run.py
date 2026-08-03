import json
import os
import tempfile
import unittest
from pathlib import Path

from batch_run import (
    REPO_ROOT,
    dataset_video_path,
    parse_args,
    parse_dataset_pre_frame_count,
    run_pipeline,
)


class BatchRunTest(unittest.TestCase):
    def test_parses_optional_pre_frame_count_from_qp_dataset_name(self):
        self.assertEqual(
            parse_dataset_pre_frame_count('QPA_75-2026-07-19-103215'),
            75,
        )
        self.assertEqual(
            parse_dataset_pre_frame_count('QPB_125-2026-07-19-103215'),
            125,
        )
        self.assertIsNone(
            parse_dataset_pre_frame_count('QPA-2026-07-19-103215')
        )
        self.assertIsNone(
            parse_dataset_pre_frame_count('130_75-2026-07-19-103215')
        )

    def test_rejects_invalid_dataset_pre_frame_count(self):
        for name in (
            'QPA_0-2026-07-19-103215',
            'QPA_-75-2026-07-19-103215',
            'QPA_abc-2026-07-19-103215',
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_dataset_pre_frame_count(name)

    def test_video_filename_matches_dataset_output_directory(self):
        output_dir = Path('风暴时刻输出') / 'QPA-2026-07-18-103215'

        self.assertEqual(
            dataset_video_path(output_dir),
            output_dir / 'QPA-2026-07-18-103215.mp4',
        )

    def make_batch_config(self, temp_dir, data_root, absolute_mode=False):
        config_path = Path(temp_dir) / 'batch.json'
        configured_data_root = str(data_root)
        if not absolute_mode:
            configured_data_root = os.path.relpath(data_root, REPO_ROOT)
        config = {
            'reorganize_config': str(REPO_ROOT / 'configs' / 'reorganize_frame_images.json'),
            'slicer_config': str(REPO_ROOT / 'configs' / 'spacetime_slicer.json'),
            'data_root': configured_data_root,
            'output_dir': None,
        }
        config_path.write_text(json.dumps(config), encoding='utf-8')
        return config_path

    def test_source_only_uses_slicers_dir_next_to_input(self):
        args, slicer_args = parse_args(['-s', './data/QP-2026-06-23-175636'])

        expected_input = REPO_ROOT / 'data' / 'QP-2026-06-23-175636'
        self.assertEqual(Path(args.datasets_to_process[0]), expected_input)
        self.assertEqual(
            Path(args.output_dir),
            REPO_ROOT / 'data' / 'Slicers' / 'QP-2026-06-23-175636',
        )
        self.assertEqual(slicer_args, [])

    def test_sub_dir_selects_all_datasets_in_time_order_regardless_of_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / 'data'
            (data_root / '0630' / '130-2026-06-30-150000').mkdir(parents=True)
            (data_root / '0630' / 'QPA-2026-06-30-161131').mkdir(parents=True)
            (data_root / '0630' / 'QPB-2026-06-30-172000').mkdir(parents=True)
            (data_root / '0630' / 'QPC-2026-06-30-181500').mkdir(parents=True)
            done_dir = data_root / '0630' / 'Slicers' / 'QPB-2026-06-30-172000'
            done_dir.mkdir(parents=True)
            (done_dir / f'{done_dir.name}.mp4').write_bytes(b'done')
            config_path = self.make_batch_config(temp_dir, data_root)

            args, _ = parse_args(['--config', str(config_path), '--sub_dir', '0630'])

            self.assertEqual(
                [Path(path) for path in args.datasets_to_process],
                [
                    (data_root / '0630' / '130-2026-06-30-150000').resolve(),
                    (data_root / '0630' / 'QPA-2026-06-30-161131').resolve(),
                    (data_root / '0630' / 'QPB-2026-06-30-172000').resolve(),
                    (data_root / '0630' / 'QPC-2026-06-30-181500').resolve(),
                ],
            )
            self.assertFalse(args.data_root_is_absolute)
            self.assertEqual(Path(args.batch_input_root), data_root / '0630')
            self.assertEqual(Path(args.batch_output_root), data_root / '0630' / 'Slicers')

    def test_datasets_selects_only_named_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / 'data'
            selected = data_root / '0630' / 'QPA-2026-06-30-161131'
            selected.mkdir(parents=True)
            (data_root / '0630' / 'QPB-2026-06-30-172000').mkdir(parents=True)
            config_path = self.make_batch_config(temp_dir, data_root)

            args, _ = parse_args([
                '--config', str(config_path),
                '--sub_dir', '0630',
                '--datasets', selected.name,
            ])

            self.assertEqual(
                [Path(path) for path in args.datasets_to_process],
                [selected.resolve()],
            )

    def test_force_with_multiple_datasets_selects_only_specified_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / 'data'
            for name in (
                'QPA-2026-06-30-161131',
                'QPB-2026-06-30-172000',
                'QPC-2026-06-30-181500',
            ):
                (data_root / '0630' / name).mkdir(parents=True)
            config_path = self.make_batch_config(temp_dir, data_root)

            args, _ = parse_args([
                '--config', str(config_path),
                '--sub_dir', '0630',
                '--force',
                '--datasets',
                'QPA-2026-06-30-161131',
                'QPC-2026-06-30-181500',
            ])

            self.assertEqual(
                [Path(path) for path in args.datasets_to_process],
                [
                    (data_root / '0630' / 'QPA-2026-06-30-161131').resolve(),
                    (data_root / '0630' / 'QPC-2026-06-30-181500').resolve(),
                ],
            )

    def test_absolute_data_root_replaces_date_and_filters_non_qp_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_root = Path(temp_dir)
            configured_root = shared_root / '0717' / '关键帧'
            qpa = configured_root / 'QPA-2026-07-17-144135'
            qpb = configured_root / 'QPB-2026-07-17-150000'
            qpc = configured_root / 'QPC-2026-07-17-160000'
            ignored = configured_root / '130-2026-07-17-144135'
            ignored_lowercase = configured_root / 'qpd-2026-07-17-170000'
            qpa.mkdir(parents=True)
            qpb.mkdir()
            qpc.mkdir()
            ignored.mkdir()
            ignored_lowercase.mkdir()
            done_dir = (
                shared_root
                / '0717'
                / '风暴时刻输出'
                / qpb.name
            )
            done_dir.mkdir(parents=True)
            (done_dir / f'{done_dir.name}.mp4').write_bytes(b'done')
            empty_output_dir = (
                shared_root
                / '0717'
                / '风暴时刻输出'
                / qpc.name
            )
            empty_output_dir.mkdir(parents=True)
            (empty_output_dir / f'{empty_output_dir.name}.mp4').write_bytes(b'')
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )

            args, _ = parse_args(['--config', str(config_path), '--sub_dir', '0717'])

            self.assertTrue(args.data_root_is_absolute)
            self.assertEqual(Path(args.batch_input_root), configured_root)
            self.assertEqual(
                Path(args.batch_output_root),
                shared_root / '0717' / '风暴时刻输出',
            )
            self.assertEqual(
                [Path(path) for path in args.datasets_to_process],
                [qpa.resolve(), qpb.resolve(), qpc.resolve()],
            )
            self.assertEqual(
                [Path(path) for path in args.output_dirs_to_process],
                [
                    shared_root / '0717' / '风暴时刻输出' / qpa.name,
                    shared_root / '0717' / '风暴时刻输出' / qpb.name,
                    shared_root / '0717' / '风暴时刻输出' / qpc.name,
                ],
            )

    def test_command_line_data_root_override_determines_absolute_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configured_relative_root = Path(temp_dir) / 'relative-data'
            configured_relative_root.mkdir()
            absolute_root = Path(temp_dir) / '0717' / '关键帧'
            selected = absolute_root / 'QPA-2026-07-17-144135'
            selected.mkdir(parents=True)
            config_path = self.make_batch_config(temp_dir, configured_relative_root)

            args, _ = parse_args([
                '--config', str(config_path),
                '--data_root', str(absolute_root),
                '--sub_dir', '0717',
            ])

            self.assertTrue(args.data_root_is_absolute)
            self.assertEqual(Path(args.batch_input_root), absolute_root)
            self.assertEqual(
                Path(args.batch_output_root),
                Path(temp_dir) / '0717' / '风暴时刻输出',
            )
            self.assertEqual(Path(args.datasets_to_process[0]), selected.resolve())

    def test_absolute_data_root_uses_sub_dir_for_both_input_and_output_dates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_root = Path(temp_dir)
            configured_root = shared_root / '0717' / '关键帧'
            selected = shared_root / '0718' / '关键帧' / 'QPA-2026-07-18-144135'
            selected.mkdir(parents=True)
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )

            args, _ = parse_args(['--config', str(config_path), '--sub_dir', '0718'])

            self.assertEqual(
                Path(args.batch_input_root),
                shared_root / '0718' / '关键帧',
            )
            self.assertEqual(
                Path(args.batch_output_root),
                shared_root / '0718' / '风暴时刻输出',
            )
            self.assertEqual(Path(args.datasets_to_process[0]), selected.resolve())
            self.assertEqual(
                Path(args.output_dirs_to_process[0]),
                shared_root / '0718' / '风暴时刻输出' / selected.name,
            )

    def test_absolute_mode_rejects_explicit_non_qp_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configured_root = Path(temp_dir) / '0717' / '关键帧'
            (configured_root / '130-2026-07-17-144135').mkdir(parents=True)
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )

            with self.assertRaises(SystemExit):
                parse_args([
                    '--config', str(config_path),
                    '--sub_dir', '0717',
                    '--datasets', '130-2026-07-17-144135',
                ])

    def test_absolute_mode_routes_multiple_outputs_to_storm_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_root = Path(temp_dir)
            configured_root = shared_root / '0717' / '关键帧'
            names = (
                'QPA-2026-07-17-144135',
                'QPB-2026-07-17-150000',
            )
            for name in names:
                (configured_root / name).mkdir(parents=True)
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )
            args, slicer_args = parse_args([
                '--config', str(config_path),
                '--sub_dir', '0717',
                '--force',
            ])
            slicer_calls = []

            result = run_pipeline(
                args,
                slicer_args,
                slicer_main=lambda argv: slicer_calls.append(argv) or 0,
                structure_checker=lambda *args, **kwargs: True,
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(slicer_calls), 2)
            for name, argv in zip(names, slicer_calls):
                output_index = argv.index('--output_dir') + 1
                self.assertEqual(
                    Path(argv[output_index]),
                    shared_root / '0717' / '风暴时刻输出' / name,
                )

    def test_dataset_name_frame_count_overrides_only_matching_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_root = Path(temp_dir)
            configured_root = shared_root / '0719' / '关键帧'
            names = (
                'QPA_75-2026-07-19-103215',
                'QPB-2026-07-19-113215',
            )
            for name in names:
                (configured_root / name).mkdir(parents=True)
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )
            args, slicer_args = parse_args([
                '--config', str(config_path),
                '--sub_dir', '0719',
                '--force',
            ])
            checked_pre_frame_counts = []
            slicer_calls = []

            def fake_checker(input_dir, **kwargs):
                checked_pre_frame_counts.append(kwargs['pre_frame_count'])
                return True

            result = run_pipeline(
                args,
                slicer_args,
                slicer_main=lambda argv: slicer_calls.append(argv) or 0,
                structure_checker=fake_checker,
            )

            self.assertEqual(result, 0)
            self.assertEqual(checked_pre_frame_counts, [75, 125])
            self.assertEqual(len(slicer_calls), 2)

            def last_option_value(argv, option):
                index = max(i for i, value in enumerate(argv) if value == option)
                return argv[index + 1]

            self.assertEqual(last_option_value(slicer_calls[0], '--start_frame'), '1')
            self.assertEqual(last_option_value(slicer_calls[0], '--freeze_frame'), '75')
            self.assertNotIn('--start_frame', slicer_calls[1])
            self.assertNotIn('--freeze_frame', slicer_calls[1])
            self.assertEqual(
                Path(
                    slicer_calls[0][slicer_calls[0].index('--output_dir') + 1]
                ).name,
                names[0],
            )

    def test_explicit_frame_parameters_override_dataset_name_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_root = Path(temp_dir)
            configured_root = shared_root / '0719' / '关键帧'
            dataset = configured_root / 'QPG_86-2026-07-19-200535'
            dataset.mkdir(parents=True)
            config_path = self.make_batch_config(
                temp_dir,
                configured_root,
                absolute_mode=True,
            )
            args, slicer_args = parse_args([
                '--config', str(config_path),
                '--sub_dir', '0719',
                '--datasets', dataset.name,
                '--force',
                '--pre-frame-count', '75',
                '--start_frame', '1',
                '--freeze_frame', '75',
            ])
            checked_pre_frame_counts = []
            slicer_calls = []

            result = run_pipeline(
                args,
                slicer_args,
                slicer_main=lambda argv: slicer_calls.append(argv) or 0,
                structure_checker=lambda input_dir, **kwargs: (
                    checked_pre_frame_counts.append(kwargs['pre_frame_count'])
                    or True
                ),
            )

            self.assertEqual(result, 0)
            self.assertEqual(checked_pre_frame_counts, [75])
            self.assertEqual(slicer_calls[0].count('--start_frame'), 1)
            self.assertEqual(slicer_calls[0].count('--freeze_frame'), 1)
            self.assertEqual(
                slicer_calls[0][slicer_calls[0].index('--freeze_frame') + 1],
                '75',
            )

    def test_single_explicit_frame_parameter_keeps_pre_and_freeze_in_sync(self):
        args, slicer_args = parse_args([
            '-s', './data/QPG_86-2026-07-19-200535',
            '--pre-frame-count', '75',
        ])
        checked_pre_frame_counts = []
        slicer_calls = []

        result = run_pipeline(
            args,
            slicer_args,
            slicer_main=lambda argv: slicer_calls.append(argv) or 0,
            structure_checker=lambda input_dir, **kwargs: (
                checked_pre_frame_counts.append(kwargs['pre_frame_count']) or True
            ),
        )

        self.assertEqual(result, 0)
        self.assertEqual(checked_pre_frame_counts, [75])
        freeze_index = slicer_calls[0].index('--freeze_frame') + 1
        self.assertEqual(slicer_calls[0][freeze_index], '75')

    def test_rejects_conflicting_explicit_pre_and_freeze_frames(self):
        args, slicer_args = parse_args([
            '-s', './data/QPG_86-2026-07-19-200535',
            '--pre-frame-count', '75',
            '--freeze_frame', '76',
        ])

        with self.assertRaisesRegex(ValueError, 'must match'):
            run_pipeline(
                args,
                slicer_args,
                slicer_main=lambda argv: 0,
                structure_checker=lambda *args, **kwargs: True,
            )

    def test_rejects_sub_dir_containing_path_components(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / 'data'
            config_path = self.make_batch_config(temp_dir, data_root)

            with self.assertRaises(SystemExit):
                parse_args([
                    '--config', str(config_path),
                    '--sub_dir', '../0717',
                ])

    def test_routes_reorganize_and_slicer_overrides(self):
        args, slicer_args = parse_args([
            '-s', './data/source',
            '--output_dir', './custom-output',
            '--pre-frame-count', '100',
            '--camera-count', '30',
            '--fps', '30',
            '--end_frame', '180',
            '--multi_subject_mode', 'all_components',
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
        self.assertIn('--multi_subject_mode', calls['slicer'])
        multi_subject_index = calls['slicer'].index('--multi_subject_mode') + 1
        self.assertEqual(calls['slicer'][multi_subject_index], 'all_components')
        self.assertIn('./custom-output', calls['slicer'])
        self.assertIn('--source_sequence_dir', calls['slicer'])
        source_sequence_index = calls['slicer'].index('--source_sequence_dir') + 1
        self.assertEqual(
            Path(calls['slicer'][source_sequence_index]),
            REPO_ROOT / 'data' / 'source' / '重命名数据',
        )

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
            self.assertEqual(Path(input_dir), REPO_ROOT / 'data' / 'source')
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
