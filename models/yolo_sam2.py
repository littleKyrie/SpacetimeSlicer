from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model interface
from models.seg_strategy import *


class YOLO_SAM2_Strategy(SegmentationStrategy):
    """YOLOv8 边界膨胀 + SAM 2 单帧分割方案"""
    def __init__(self, device, expand_ratio=0.4): # 扩大 padding 以包容扇子
        print(f">> 初始化 YOLO+SAM2 分割器 (Padding={expand_ratio})...")
        
        self.device = device
        self.expand_ratio = expand_ratio
        # self.yolo_model = YOLO('yolov8n.pt')
        self.yolo_model = YOLO('./checkpoints/yolo/yolov8n.pt')
        
        # 请确保路径正确
        sam2_checkpoint = "checkpoints/sam2/sam2.1_hiera_large.pt"
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def process_frame(self, current_img, current_idx):
        img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        img_h, img_w = current_img.shape[:2]

        results = self.yolo_model(current_img, classes=[0], verbose=False)
        if len(results[0].boxes) == 0:
            return np.zeros((img_h, img_w), dtype=np.uint8)
            
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * self.expand_ratio / 2
        pad_y = (y2 - y1) * self.expand_ratio / 2
        
        expanded_box = np.array([max(0, x1 - pad_x), max(0, y1 - pad_y), 
                                 min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)], dtype=np.float32)

        self.predictor.set_image(img_rgb)
        masks, _, _ = self.predictor.predict(box=expanded_box, multimask_output=False)
        return (masks[0] * 255).astype(np.uint8)