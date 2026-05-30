# ---------------------------------------------------------
# 分割算法接口 (Strategy Pattern)
# ---------------------------------------------------------
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class SegmentationStrategy:
    def process_frame(self, current_img, current_idx):
        """返回当前帧的 Alpha 通道 (0~255 的 numpy array)"""
        raise NotImplementedError