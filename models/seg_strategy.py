# ---------------------------------------------------------
# 分割算法接口 (Strategy Pattern)
# ---------------------------------------------------------
import cv2
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class MattingResult:
    """Straight BGR foreground and alpha, float32 in [0,1], same resolution."""
    alpha: np.ndarray
    foreground: np.ndarray = None
    alpha_origin: str = 'normalized_uint8'


def bgr_frame_to_tensor(frame):
    """Convert a BGR uint8 OpenCV frame to an RGB CHW float tensor in [0, 1]."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb_frame).permute(2, 0, 1).float().div(255.0)


class SegmentationStrategy:
    supports_foreground = False

    def process_frame(self, current_img, current_idx):
        """返回当前帧的 Alpha 通道 (0~255 的 numpy array)"""
        raise NotImplementedError

    def process_matting(self, current_img, current_idx, include_foreground=False):
        if include_foreground:
            raise ValueError(f'{type(self).__name__} does not provide foreground colors')
        alpha = self.process_frame(current_img, current_idx)
        return MattingResult(alpha.astype(np.float32) / 255.0)
