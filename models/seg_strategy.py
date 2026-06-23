# ---------------------------------------------------------
# 分割算法接口 (Strategy Pattern)
# ---------------------------------------------------------
import cv2
import numpy as np
import torch


def bgr_frame_to_tensor(frame):
    """Convert a BGR uint8 OpenCV frame to an RGB CHW float tensor in [0, 1]."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb_frame).permute(2, 0, 1).float().div(255.0)


class SegmentationStrategy:
    def process_frame(self, current_img, current_idx):
        """返回当前帧的 Alpha 通道 (0~255 的 numpy array)"""
        raise NotImplementedError
