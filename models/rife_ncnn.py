import os
import shutil
import subprocess
import tempfile

import cv2


class RifeNcnnInterpolator:
    """Neural frame interpolation through the portable rife-ncnn-vulkan binary."""

    def __init__(self, executable=None, model_dir=None, uhd=False):
        self.executable = self.find_executable(executable)
        self.model_dir = self.find_timestep_model(model_dir)
        self.uhd = uhd
        self.temp_dir = tempfile.TemporaryDirectory(prefix='spacetime_slicer_rife_')
        self.counter = 0

    @staticmethod
    def find_executable(executable=None):
        candidates = [
            executable,
            os.environ.get('RIFE_NCNN_EXE'),
            os.path.join('third_party', 'rife-ncnn-vulkan', 'rife-ncnn-vulkan.exe'),
            os.path.join('third_party', 'rife-ncnn-vulkan', 'rife-ncnn-vulkan'),
            os.path.join('tools', 'rife-ncnn-vulkan', 'rife-ncnn-vulkan.exe'),
            os.path.join('tools', 'rife-ncnn-vulkan', 'rife-ncnn-vulkan'),
            shutil.which('rife-ncnn-vulkan.exe'),
            shutil.which('rife-ncnn-vulkan'),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(
            'rife-ncnn-vulkan executable not found. Pass --rife_exe or set RIFE_NCNN_EXE.'
        )

    def find_timestep_model(self, model_dir=None):
        if model_dir:
            resolved_model_dir = os.path.abspath(model_dir)
            if not os.path.isdir(resolved_model_dir):
                raise FileNotFoundError(f'RIFE model directory not found: {resolved_model_dir}')
            return resolved_model_dir

        executable_dir = os.path.dirname(self.executable)
        for model_name in ('rife-v4.6', 'rife-v4'):
            candidate = os.path.join(executable_dir, model_name)
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            'A RIFE v4 model is required for custom timesteps. '
            'Pass --rife_model_dir pointing to rife-v4.6 or rife-v4.'
        )

    def interpolate(self, first_frame, second_frame, timestep):
        if not 0.0 < timestep < 1.0:
            raise ValueError('RIFE timestep must be between 0 and 1')

        self.counter += 1
        prefix = f'{self.counter:08d}'
        first_path = os.path.join(self.temp_dir.name, f'{prefix}_0.png')
        second_path = os.path.join(self.temp_dir.name, f'{prefix}_1.png')
        output_path = os.path.join(self.temp_dir.name, f'{prefix}_out.png')
        cv2.imwrite(first_path, first_frame)
        cv2.imwrite(second_path, second_frame)

        command = [
            self.executable,
            '-0', first_path,
            '-1', second_path,
            '-o', output_path,
            '-s', f'{timestep:.8f}',
        ]
        if self.model_dir:
            command.extend(['-m', self.model_dir])
        if self.uhd:
            command.append('-u')

        result = subprocess.run(
            command,
            cwd=os.path.dirname(self.executable),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'RIFE inference failed ({result.returncode}) with model {self.model_dir}: '
                f'{result.stderr.strip()}'
            )

        interpolated = cv2.imread(output_path)
        if interpolated is None:
            raise RuntimeError(f'RIFE did not create an output frame: {output_path}')
        return interpolated

    def close(self):
        self.temp_dir.cleanup()


class NeuralSlowMotionWriter:
    """Write source frames and insert RIFE predictions only inside a selected source range."""

    def __init__(self, out, interpolator=None, factor=1, start_idx=None, end_idx=None):
        self.out = out
        self.interpolator = interpolator
        self.factor = factor
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.previous_frame = None
        self.previous_idx = None

    def write(self, frame, source_idx):
        if self.previous_frame is not None and self._should_interpolate(source_idx):
            for step in range(1, self.factor):
                self.out.write(
                    self.interpolator.interpolate(self.previous_frame, frame, step / self.factor)
                )
        self.out.write(frame)
        self.previous_frame = frame.copy()
        self.previous_idx = source_idx

    def _should_interpolate(self, source_idx):
        return (
            self.interpolator is not None and
            self.factor > 1 and
            self.start_idx is not None and
            self.end_idx is not None and
            self.start_idx <= self.previous_idx and
            source_idx <= self.end_idx
        )
