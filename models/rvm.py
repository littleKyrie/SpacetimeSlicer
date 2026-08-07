from pathlib import Path

# Model interface
from models.seg_strategy import *


class RVMStrategy(SegmentationStrategy):
    """纯 RVM 分割方案"""
    def __init__(self, device):
        print(">> 初始化 RVM 分割器...")
        self.device = device
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
        tensor_img = bgr_frame_to_tensor(current_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(tensor_img, *self.rec, downsample_ratio=1.0)
        alpha = (pha[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, 0]
        return alpha
