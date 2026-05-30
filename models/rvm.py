# Model interface
from models.seg_strategy import *


class RVMStrategy(SegmentationStrategy):
    """纯 RVM 分割方案"""
    def __init__(self, device):
        print(">> 初始化 RVM 分割器...")
        self.device = device
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").to(self.device).eval()
        self.rec = [None] * 4
        self.transform = transforms.ToTensor()

    def process_frame(self, current_img, current_idx):
        pil_img = Image.fromarray(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(tensor_img, *self.rec, downsample_ratio=0.25)
        alpha = (pha[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, 0]
        return alpha