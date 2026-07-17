import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.opencv_io import imread_required, imread_unicode


class OpenCvIoTest(unittest.TestCase):
    def test_reads_image_from_unicode_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / '拉流生成' / '0717' / '关键帧'
            image_dir.mkdir(parents=True)
            image_path = image_dir / '测试图片.png'
            expected = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint8,
            )
            success, encoded = cv2.imencode('.png', expected)
            self.assertTrue(success)
            image_path.write_bytes(encoded.tobytes())

            actual = imread_required(image_path)

            np.testing.assert_array_equal(actual, expected)

    def test_supports_grayscale_decode_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / '中文灰度图.png'
            expected = np.array([[0, 64], [128, 255]], dtype=np.uint8)
            success, encoded = cv2.imencode('.png', expected)
            self.assertTrue(success)
            image_path.write_bytes(encoded.tobytes())

            actual = imread_required(image_path, cv2.IMREAD_GRAYSCALE)

            np.testing.assert_array_equal(actual, expected)

    def test_missing_image_matches_imread_none_semantics(self):
        missing_path = Path('不存在的中文目录') / '不存在.jpg'

        self.assertIsNone(imread_unicode(missing_path))

    def test_required_read_reports_original_path(self):
        missing_path = Path('不存在的中文目录') / '不存在.jpg'

        with self.assertRaises(RuntimeError) as context:
            imread_required(missing_path)
        self.assertIn(str(missing_path), str(context.exception))


if __name__ == '__main__':
    unittest.main()
