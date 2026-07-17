import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.ffmpeg_video import FfmpegH264Writer, resolve_ffmpeg_executable


class FakeProcess:
    def __init__(self, command, returncode=0, stderr=b''):
        self.command = command
        self.returncode = returncode
        self.stdin_buffer = io.BytesIO()
        self.stdin = self.stdin_buffer
        self.stderr = io.BytesIO(stderr)
        self.terminated = False

    def poll(self):
        return None if not self.terminated and self.stdin is not None else self.returncode

    def wait(self):
        if self.returncode == 0:
            Path(self.command[-1]).write_bytes(b'h264-output')
        return self.returncode

    def terminate(self):
        self.terminated = True


class FfmpegVideoTest(unittest.TestCase):
    def test_streams_raw_bgr_frames_and_atomically_replaces_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / '中文输出目录' / 'slicer.mp4'
            destination.parent.mkdir()
            destination.write_bytes(b'previous-video')
            processes = []

            def fake_popen(command, **kwargs):
                process = FakeProcess(command)
                processes.append(process)
                return process

            writer = FfmpegH264Writer(
                destination,
                fps=25,
                frame_size=(2, 1),
                executable=Path(__file__),
                popen_factory=fake_popen,
            )
            frame = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
            writer.write(frame)
            written = processes[0].stdin_buffer.getvalue()
            writer.release()

            self.assertEqual(written, frame.tobytes())
            self.assertEqual(destination.read_bytes(), b'h264-output')
            self.assertFalse((destination.parent / 'slicer.h264.tmp.mp4').exists())
            command = processes[0].command
            self.assertIn('rawvideo', command)
            self.assertIn('bgr24', command)
            self.assertIn('libx264', command)
            self.assertNotIn('mp4v', command)

    def test_failed_encoder_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / 'slicer.mp4'
            destination.write_bytes(b'previous-video')

            def fake_popen(command, **kwargs):
                return FakeProcess(command, returncode=1, stderr=b'encoder failed')

            writer = FfmpegH264Writer(
                destination,
                fps=25,
                frame_size=(1, 1),
                executable=Path(__file__),
                popen_factory=fake_popen,
            )
            writer.write(np.zeros((1, 1, 3), dtype=np.uint8))

            with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                writer.release()

            self.assertEqual(destination.read_bytes(), b'previous-video')
            self.assertFalse((Path(temp_dir) / 'slicer.h264.tmp.mp4').exists())

    def test_rejects_frame_with_wrong_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = FfmpegH264Writer(
                Path(temp_dir) / 'slicer.mp4',
                fps=25,
                frame_size=(2, 2),
                executable=Path(__file__),
                popen_factory=lambda command, **kwargs: FakeProcess(command),
            )
            with self.assertRaisesRegex(ValueError, 'shape'):
                writer.write(np.zeros((1, 2, 3), dtype=np.uint8))
            writer.abort()

    def test_accepts_directory_containing_ffmpeg_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable_name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
            executable = Path(temp_dir) / executable_name
            executable.write_bytes(b'fake')

            self.assertEqual(
                Path(resolve_ffmpeg_executable(temp_dir)),
                executable.resolve(),
            )

    def test_real_ffmpeg_output_is_h264_when_ffmpeg_is_available(self):
        try:
            ffmpeg = Path(resolve_ffmpeg_executable())
        except FileNotFoundError:
            self.skipTest('FFmpeg is not available')

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / '中文输出' / 'slicer.mp4'
            destination.parent.mkdir()
            writer = FfmpegH264Writer(
                destination,
                fps=25,
                frame_size=(4, 4),
                executable=ffmpeg,
            )
            for value in (0, 64, 128, 255):
                writer.write(np.full((4, 4, 3), value, dtype=np.uint8))
            writer.release()

            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)
            ffprobe_name = 'ffprobe.exe' if os.name == 'nt' else 'ffprobe'
            ffprobe = ffmpeg.with_name(ffprobe_name)
            if not ffprobe.is_file():
                self.skipTest('ffprobe is not available next to FFmpeg')
            result = subprocess.run(
                [
                    str(ffprobe),
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(destination),
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'h264')


if __name__ == '__main__':
    unittest.main()
