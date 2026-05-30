import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

def test_sam2_pure_body_tracking_yolo(input_dir, output_dir, sam2_checkpoint, sam2_model_cfg):
    """
    极简纯净版 (YOLO 赋能)：YOLO 提取人物框中心点 -> SAM 2 点提示追踪
    确保 100% 过滤背景，牺牲伸出体外的道具（如扇子）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(f"🚀 使用设备: {device}")

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print("❌ 输入目录为空！")
        return

    # =========================================================
    # 第一阶段：预处理数据集为严格格式
    # =========================================================
    print("📦 [阶段 1/3] 正在准备 SAM 2 临时目录...")
    temp_sam2_dir = os.path.join(output_dir, ".temp_sam2_frames")
    os.makedirs(temp_sam2_dir, exist_ok=True)
    frame_mapping = {}
    
    for i, img_name in enumerate(image_files):
        src_path = os.path.join(input_dir, img_name)
        ext = os.path.splitext(img_name)[1]
        temp_name = f"{i:05d}{ext}"
        shutil.copy(src_path, os.path.join(temp_sam2_dir, temp_name))
        frame_mapping[i] = img_name

    # =========================================================
    # 第二阶段：YOLO 锁定人物中心点作为提示点
    # =========================================================
    print("🤖 [阶段 2/3] 使用 YOLOv8 锁定人物身体重心作为提示点...")
    # yolo_model = YOLO('yolov8n.pt')
    yolo_model = YOLO('./checkpoints/yolo/yolov8n.pt')
    
    start_frame_idx = 0
    center_point = None
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        # 强制只检测人 (class 0)
        results = yolo_model(img, classes=[0], verbose=False)
        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # 核心改变：不喂大框，只计算框的几何中心点
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            center_point = np.array([[cx, cy]], dtype=np.float32)
            start_frame_idx = i
            break # 找到第一帧有人的画面，提取中心点后立即停止扫描

    if center_point is None:
        print("❌ YOLO 未能在视频中检测到人物。")
        shutil.rmtree(temp_sam2_dir)
        return
        
    print(f"🎯 成功提取人物重心点坐标: {center_point[0]}, 将从第 {start_frame_idx} 帧开始追踪。")

    # =========================================================
    # 第三阶段：SAM 2 点提示追踪
    # =========================================================
    print("🧠 [阶段 3/3] 启动 SAM 2 纯净模式时序追踪...")
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=temp_sam2_dir)
    predictor.reset_state(inference_state)
    
    # 标签 1 表示这是一个我们需要的“正向提示点” (前景)
    points_labels = np.array([1], np.int32) 
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=start_frame_idx,
        obj_id=1,
        points=center_point,
        labels=points_labels
    )
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        original_img_name = frame_mapping[out_frame_idx]
        current_img_path = os.path.join(input_dir, original_img_name)
        current_img = cv2.imread(current_img_path)

        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        mask_canvas = (mask * 255).astype(np.uint8)

        img_bgra = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask_canvas

        output_name = f"{os.path.splitext(original_img_name)[0]}_purebody.png"
        cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
        
        print(f"  [+] 追踪并保存: {output_name} [{out_frame_idx+1}/{len(image_files)}]", end='\r')

    print("\n🧹 正在清理临时文件...")
    shutil.rmtree(temp_sam2_dir)
    print("\n✅ 纯净版自动化分割完成！")

if __name__ == "__main__":
    # --- 你的路径配置 ---
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_purebody"
    
    SAM2_CHECKPOINT = "./checkpoints/sam2/sam2.1_hiera_large.pt"
    SAM2_CFG = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    
    test_sam2_pure_body_tracking_yolo(INPUT_FOLDER, OUTPUT_FOLDER, SAM2_CHECKPOINT, SAM2_CFG)