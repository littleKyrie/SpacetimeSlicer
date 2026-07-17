import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def resolve_ffmpeg_executable(executable=None):
    requested = executable or os.environ.get('FFMPEG_EXE') or 'ffmpeg'
    requested_path = Path(requested).expanduser()
    if requested_path.is_dir():
        executable_name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
        requested_path = requested_path / executable_name
    if requested_path.is_file():
        return str(requested_path.resolve())

    discovered = shutil.which(str(requested))
    if discovered:
        return discovered

    raise FileNotFoundError(
        'FFmpeg executable not found. Install FFmpeg and add it to PATH, '
        'set FFMPEG_EXE, or pass --ffmpeg_exe <path>.'
    )


class FfmpegH264Writer:
    """Stream BGR frames to FFmpeg and atomically publish an H.264 MP4."""

    def __init__(
        self,
        output_path,
        fps,
        frame_size,
        executable=None,
        crf=18,
        preset='medium',
        popen_factory=None,
    ):
        if not 0 <= crf <= 51:
            raise ValueError('H.264 CRF must be between 0 and 51')
        width, height = frame_size
        if width < 1 or height < 1:
            raise ValueError('frame_size must contain positive width and height')
        if fps <= 0:
            raise ValueError('fps must be positive')

        self.output_path = Path(output_path)
        self.temporary_output = self.output_path.with_name(
            f'{self.output_path.stem}.h264.tmp{self.output_path.suffix}'
        )
        self.frame_size = (width, height)
        self._released = False
        self.temporary_output.unlink(missing_ok=True)
        ffmpeg = resolve_ffmpeg_executable(executable)
        self.command = [
            ffmpeg,
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-video_size', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', 'pipe:0',
            '-an',
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(self.temporary_output),
        ]
        factory = popen_factory or subprocess.Popen
        self._process = factory(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def isOpened(self):
        return (
            not self._released
            and self._process.poll() is None
            and self._process.stdin is not None
            and not self._process.stdin.closed
        )

    def write(self, frame):
        if not self.isOpened():
            raise RuntimeError('FFmpeg video writer is not open')
        width, height = self.frame_size
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError(
                'Expected a uint8 BGR frame with shape '
                f'({height}, {width}, 3), got {frame.dtype} {frame.shape}'
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('FFmpeg closed the raw-video input pipe unexpectedly') from exc

    def release(self):
        if self._released:
            return
        self._released = True
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        self._process.stdin = None
        stderr = self._process.stderr.read() if self._process.stderr is not None else b''
        returncode = self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()

        if returncode != 0:
            self.temporary_output.unlink(missing_ok=True)
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            details = str(stderr).strip() or 'unknown FFmpeg error'
            raise RuntimeError(f'FFmpeg H.264 encoding failed: {details}')
        if not self.temporary_output.is_file() or self.temporary_output.stat().st_size == 0:
            self.temporary_output.unlink(missing_ok=True)
            raise RuntimeError(
                f'FFmpeg did not create a valid H.264 video: {self.temporary_output}'
            )
        os.replace(self.temporary_output, self.output_path)

    def abort(self):
        if self._released:
            return
        self._released = True
        try:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait()
        finally:
            if self._process.stderr is not None:
                self._process.stderr.close()
            self.temporary_output.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.abort()
        except Exception:
            pass
