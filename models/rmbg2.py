import cv2
import numpy as np
import torch
from PIL import Image
from .seg_strategy import SegmentationStrategy


class RMBG2Strategy(SegmentationStrategy):
    """Bria AI RMBG 2.0 抠图策略"""

    def __init__(self, device="cuda"):
        self.device = device
        try:
            from transformers import AutoModelForImageSegmentation
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'briaai/RMBG-2.0',
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            torch.set_float32_matmul_precision(['high', 'highest'][0])
        except Exception as e:
            print(f"⚠️  RMBG-2.0 加载失败: {e}")
            raise e

    def process_frame(self, frame, frame_idx=None):
        """
        处理单帧，返回 alpha 通道 (0-255)

        Args:
            frame: BGR 图像 (H, W, 3)
            frame_idx: 帧索引（可选）

        Returns:
            alpha_mask: 单通道灰度图 (H, W), uint8 (0-255)
        """
        from torchvision import transforms

        h, w = frame.shape[:2]
        image_size = (1024, 1024)

        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform_image(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        mask = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask)
        mask_pil = mask_pil.resize((w, h), Image.BILINEAR)

        mask = np.array(mask_pil)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

        return mask

    def name(self):
        return "RMBG2"