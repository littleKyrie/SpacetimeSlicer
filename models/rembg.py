import cv2
import numpy as np
from PIL import Image
from .seg_strategy import SegmentationStrategy


class RembgStrategy(SegmentationStrategy):
    """Rembg 抠图策略"""

    def __init__(self, device="cuda", model_name="bria-rmbg"):
        self.device = device
        self.model_name = model_name
        try:
            from rembg import new_session
            self.session = new_session(model_name)
        except ImportError:
            print("⚠️  Rembg 未安装")
            raise ImportError("rembg not installed")

    def process_frame(self, frame, frame_idx=None):
        """
        处理单帧，返回 alpha 通道 (0-255)

        Args:
            frame: BGR 图像 (H, W, 3)
            frame_idx: 帧索引（可选）

        Returns:
            alpha_mask: 单通道灰度图 (H, W), uint8 (0-255)
        """
        from rembg import remove

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        output = remove(pil_image, session=self.session)

        if output.mode == 'RGBA':
            alpha_channel = np.array(output.split()[-1])
        else:
            alpha_channel = np.ones((pil_image.height, pil_image.width), dtype=np.uint8) * 255

        return alpha_channel

    def name(self):
        return f"Rembg-{self.model_name}"