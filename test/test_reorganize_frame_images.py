import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from utils.reorganize_frame_images import (
    ReorganizationError,
    has_reorganized_frame_structure,
    parse_args,
    reorganize_directory,
)


def make_source_images(directory, count):
    for index in range(count):
        path = directory / f'capture_777_frame_{index:03d}.jpg'
        path.write_bytes(f'source-{index}'.encode('ascii'))


class ReorganizeFrameImagesTest(unittest.TestCase):
    def test_detects_existing_reorganized_frame_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame_number in range(1, 4):
                frame_dir = root / f'{frame_number:04d}'
                frame_dir.mkdir()
                (frame_dir / '001.jpg').write_bytes(b'frame')
            (root / '0002' / '002.jpg').write_bytes(b'camera-2')
            (root / '0002' / '003.jpg').write_bytes(b'camera-3')

            self.assertTrue(
                has_reorganized_frame_structure(
                    root,
                    pre_frame_count=2,
                    camera_count=2,
                )
            )

    def test_reorganized_structure_rejects_incomplete_camera_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame_number in range(1, 4):
                frame_dir = root / f'{frame_number:04d}'
                frame_dir.mkdir()
                (frame_dir / '001.jpg').write_bytes(b'frame')
            (root / '0002' / '002.jpg').write_bytes(b'camera-2')

            self.assertFalse(
                has_reorganized_frame_structure(
                    root,
                    pre_frame_count=2,
                    camera_count=2,
                )
            )

    def test_parse_args_accepts_input_dir_option(self):
        args = parse_args(['--input_dir', './data/QP-2026-06-23-174635'])

        self.assertEqual(args.input_dir, './data/QP-2026-06-23-174635')

    def test_parse_args_keeps_positional_input_dir(self):
        args = parse_args(['./data/QP-2026-06-23-174635'])

        self.assertEqual(args.input_dir, './data/QP-2026-06-23-174635')

    def test_config_values_are_overridden_by_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'reorganize.json'
            config_path.write_text(json.dumps({
                'pre_frame_count': 100,
                'camera_count': 30,
                'original_dir_name': 'originals',
                'normalized_dir_name': 'normalized',
                'image_ext': '.png',
                'dry_run': True,
            }), encoding='utf-8')

            args = parse_args([
                '--config', str(config_path),
                '--input_dir', './data/source',
                '--camera-count', '45',
                '--no-dry-run',
            ])

            self.assertEqual(args.pre_frame_count, 100)
            self.assertEqual(args.camera_count, 45)
            self.assertEqual(args.image_ext, '.png')
            self.assertFalse(args.dry_run)

    def test_reorganizes_effect_frame_with_camera_count_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_source_images(root, 220)

            reorganize_directory(root, pre_frame_count=125, camera_count=90)

            self.assertFalse((root / 'capture_777_frame_000.jpg').exists())
            self.assertEqual(
                (root / '原始图片' / 'capture_777_frame_000.jpg').read_bytes(),
                b'source-0',
            )
            self.assertEqual((root / '重命名数据' / '000.jpg').read_bytes(), b'source-0')
            self.assertEqual((root / '0001' / '001.jpg').read_bytes(), b'source-0')
            self.assertEqual((root / '0125' / '001.jpg').read_bytes(), b'source-124')
            self.assertEqual((root / '0125' / '002.jpg').read_bytes(), b'source-125')
            self.assertEqual((root / '0125' / '091.jpg').read_bytes(), b'source-214')
            self.assertEqual((root / '0126' / '001.jpg').read_bytes(), b'source-215')

    def test_uses_last_number_in_filename_for_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names_by_source_index = [
                'camera_010_frame_002.jpg',
                'camera_010_frame_000.jpg',
                'camera_010_frame_001.jpg',
                'camera_010_frame_004.jpg',
                'camera_010_frame_003.jpg',
            ]
            for source_index, name in enumerate(names_by_source_index):
                (root / name).write_bytes(f'source-{source_index}'.encode('ascii'))

            reorganize_directory(root, pre_frame_count=2, camera_count=2)

            self.assertEqual((root / '重命名数据' / '000.jpg').read_bytes(), b'source-1')
            self.assertEqual((root / '重命名数据' / '001.jpg').read_bytes(), b'source-2')
            self.assertEqual((root / '重命名数据' / '002.jpg').read_bytes(), b'source-0')

    def test_rejects_duplicate_numeric_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'left_001.jpg').write_bytes(b'left')
            (root / 'right_001.png').write_bytes(b'right')

            with self.assertRaisesRegex(ReorganizationError, 'Duplicate numeric id'):
                reorganize_directory(root, pre_frame_count=1, camera_count=1)

    def test_rejects_insufficient_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_source_images(root, 215)

            with self.assertRaisesRegex(ReorganizationError, 'Not enough images'):
                reorganize_directory(root, pre_frame_count=125, camera_count=90)

    def test_dry_run_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_source_images(root, 216)

            output = io.StringIO()
            with redirect_stdout(output):
                reorganize_directory(root, pre_frame_count=125, camera_count=90, dry_run=True)

            self.assertIn('COPY', output.getvalue())
            self.assertIn('DELETE', output.getvalue())
            self.assertTrue((root / 'capture_777_frame_000.jpg').exists())
            self.assertFalse((root / '原始图片').exists())
            self.assertFalse((root / '重命名数据').exists())
            self.assertFalse((root / '0001').exists())

    def test_rejects_existing_target_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_source_images(root, 216)
            (root / '0001').mkdir()
            (root / '0001' / '001.jpg').write_bytes(b'existing')

            with self.assertRaisesRegex(ReorganizationError, 'already exists'):
                reorganize_directory(root, pre_frame_count=125, camera_count=90)


if __name__ == '__main__':
    unittest.main()
