import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def test_yolo_sam2_segmentation(input_dir, output_dir, sam2_checkpoint, sam2_model_cfg):
    """
    使用 YOLOv8 提取 BBox，结合 SAM 2 进行精细分割
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")

    # 1. 加载 YOLOv8 纯检测模型 (速度最快)
    print("加载 YOLOv8 检测模型...")
    # yolo_model = YOLO('yolov8n.pt') 
    yolo_model = YOLO('./checkpoints/yolo/yolov8n.pt')

    # 2. 加载 SAM 2 模型
    print("加载 SAM 2 模型...")
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # SAM 2 需要 RGB 格式

        # Step A: YOLO 获取边界框
        results = yolo_model(img, classes=[0], verbose=False)
        
        if len(results[0].boxes) == 0:
            print(f"  [-] {img_name} 中未检测到人物，跳过。")
            continue
            
        # 获取置信度最高的人物的 BBox (xyxy格式)
        box = results[0].boxes.xyxy[0].cpu().numpy()

        # Step B: 将图像和 BBox 喂给 SAM 2
        predictor.set_image(img_rgb)
        # multimask_output=False 意味着只让它输出它认为最准的一个 Mask
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False 
        )

        # 处理 SAM 2 输出的 Mask
        mask = masks[0] # shape: (H, W), dtype: bool
        mask_canvas = (mask * 255).astype(np.uint8)

        # 制作带透明通道的 PNG
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask_canvas

        output_name = f"{os.path.splitext(img_name)[0]}_sam2.png"
        cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
        print(f"  [+] 成功处理: {output_name}")

if __name__ == "__main__":
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_sam2_masks"
    
    # SAM2 配置文件和权重路径 (请根据你下载的具体版本修改)
    SAM2_CHECKPOINT = "./checkpoints/sam2/sam2.1_hiera_large.pt"
    SAM2_CFG = "./configs/sam2.1/sam2.1_hiera_l.yaml"
    
    test_yolo_sam2_segmentation(INPUT_FOLDER, OUTPUT_FOLDER, SAM2_CHECKPOINT, SAM2_CFG)