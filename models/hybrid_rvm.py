import os
from pathlib import Path

# Model interface
from models.seg_strategy import *


class HybridStrategy(SegmentationStrategy):
    """RVM + 物理背景差分融合方案"""
    def __init__(self, device, median_bg, diff_threshold=35): # 调大 threshold 以减少背景瑕疵，但可能牺牲部分扇子边缘
        print(f">> 初始化 Hybrid 分割器 (Threshold={diff_threshold})...")
        self.device = device
        self.median_bg = median_bg
        self.diff_threshold = diff_threshold
        repository = Path(__file__).resolve().parents[1] / "third_party" / "RobustVideoMatting"
        if not repository.is_dir():
            raise FileNotFoundError(
                f"RobustVideoMatting repository not found: {repository}. "
                "Run setup.ps1 first."
            )
        self.model = torch.hub.load(
            str(repository), "resnet50", source='local'
        ).to(self.device).eval()
        self.rec = [None] * 4

    def process_frame(self, current_img, current_idx):
        # 1. RVM 核心
        tensor_img = bgr_frame_to_tensor(current_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(tensor_img, *self.rec, downsample_ratio=0.25)
        rvm_mask = (pha[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, 0]
        _, rvm_core = cv2.threshold(rvm_mask, 127, 255, cv2.THRESH_BINARY)

        # 2. 背景差分
        gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_current, self.median_bg)
        _, diff_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.dilate(diff_mask, np.ones((5, 5), np.uint8), iterations=1)

        # 3. 连通域验证
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        validated_diff_mask = np.zeros_like(diff_mask)
        for contour in contours:
            if cv2.contourArea(contour) < 100: continue
            temp_mask = np.zeros_like(diff_mask)
            cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)
            if cv2.countNonZero(cv2.bitwise_and(temp_mask, rvm_core)) > 0:
                validated_diff_mask = cv2.bitwise_or(validated_diff_mask, temp_mask)

        # 4. 融合
        final_alpha = cv2.max(rvm_mask, validated_diff_mask)
        return cv2.GaussianBlur(final_alpha, (3, 3), 0)

    def _get_median_background(self):
        print("计算用于 Hybrid 方案的全局中值背景...")
        frames = [cv2.cvtColor(cv2.imread(os.path.join(self.input_dir, self.image_files[i])), cv2.COLOR_BGR2GRAY) 
                  for i in range(0, self.total_frames, 5)]
        return np.median(frames, axis=0).astype(np.uint8)
