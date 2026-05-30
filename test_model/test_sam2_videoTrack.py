import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

def test_ultimate_auto_tracking(input_dir, output_dir, sam2_checkpoint, sam2_model_cfg, expand_ratio=0.3):
    """
    终极方案：YOLO 全局扫描找最大面积帧 -> 边界框膨胀 -> 自动规范化数据集 -> SAM 2 双向时序追踪
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

    # =========================================================
    # 第一阶段：YOLOv8 "先锋侦察" - 寻找黄金关键帧
    # =========================================================
    print("🤖 [阶段 1/3] 正在使用 YOLO 进行全局扫描，寻找动作最大张力帧...")
    # yolo_model = YOLO('yolov8n.pt')
    yolo_model = YOLO('./checkpoints/yolo/yolov8n.pt')
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    max_area = 0
    golden_frame_idx = 0
    golden_box = None
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        results = yolo_model(img, classes=[0], verbose=False)
        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            if area > max_area:
                max_area = area
                golden_frame_idx = i
                golden_box = box

    if golden_box is None:
        print("❌ 整个视频都没有检测到人物！")
        return

    print(f"🎯 锁定黄金关键帧: 第 {golden_frame_idx + 1} 帧 ({image_files[golden_frame_idx]})")
    
    # =========================================================
    # 第二阶段：预处理数据集目录 & 计算膨胀框
    # =========================================================
    print("📦 [阶段 2/3] 正在为 SAM 2 准备标准化的临时目录...")
    
    # 创建一个临时目录用于存放纯数字命名的图片，以绕过 SAM 2 的硬性命名要求
    temp_sam2_dir = os.path.join(output_dir, ".temp_sam2_frames")
    os.makedirs(temp_sam2_dir, exist_ok=True)
    
    frame_mapping = {} # 用于记录: 索引 -> 原始文件名
    
    for i, img_name in enumerate(image_files):
        src_path = os.path.join(input_dir, img_name)
        # 将文件重命名为严格的 5 位数字格式，如 00000.jpg
        ext = os.path.splitext(img_name)[1]
        temp_name = f"{i:05d}{ext}"
        dst_path = os.path.join(temp_sam2_dir, temp_name)
        
        # 复制文件到临时目录
        shutil.copy(src_path, dst_path)
        frame_mapping[i] = img_name

    print("🧠 正在加载 SAM 2 视频追踪器...")
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
    
    sample_img = cv2.imread(os.path.join(input_dir, image_files[0]))
    img_h, img_w = sample_img.shape[:2]

    # 对黄金关键帧的 YOLO 框进行膨胀
    gx1, gy1, gx2, gy2 = golden_box
    box_w = gx2 - gx1
    box_h = gy2 - gy1
    pad_x = box_w * expand_ratio / 2
    pad_y = box_h * expand_ratio / 2
    
    expanded_box = np.array([
        max(0, gx1 - pad_x),
        max(0, gy1 - pad_y),
        min(img_w, gx2 + pad_x),
        min(img_h, gy2 + pad_y)
    ], dtype=np.float32)

    # 核心：将初始化路径改为纯数字的 temp_sam2_dir
    inference_state = predictor.init_state(video_path=temp_sam2_dir)
    predictor.reset_state(inference_state)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=golden_frame_idx,
        obj_id=1,
        box=expanded_box
    )

    # =========================================================
    # 第三阶段：执行全视频时序传播 (双向)
    # =========================================================
    print(f"🎬 [阶段 3/3] 开始双向视频追踪与合成...")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # 通过字典映射找回原始的文件名
        original_img_name = frame_mapping[out_frame_idx]
        current_img_path = os.path.join(input_dir, original_img_name)
        current_img = cv2.imread(current_img_path)

        # 提取 Mask 并转换为 Alpha 通道
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        mask_canvas = (mask * 255).astype(np.uint8)

        # 合成并保存
        img_bgra = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask_canvas

        output_name = f"{os.path.splitext(original_img_name)[0]}_ultimatetrack.png"
        cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
        
        print(f"  [+] 追踪并保存: {output_name} [{out_frame_idx+1}/{len(image_files)}]", end='\r')

    # 自动清理临时文件夹
    print("\n🧹 正在清理临时文件...")
    shutil.rmtree(temp_sam2_dir)

    print("\n✅ 终极自动化分割完成！完美融合 YOLO 自动化与 SAM 2 视频时序记忆！")

# ================= 脚本入口 =================
if __name__ == "__main__":
    # 你的路径配置
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_ultimate"
    
    SAM2_CHECKPOINT = "./checkpoints/sam2/sam2.1_hiera_large.pt"
    SAM2_CFG = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    
    # 执行测试 
    test_ultimate_auto_tracking(INPUT_FOLDER, OUTPUT_FOLDER, SAM2_CHECKPOINT, SAM2_CFG, expand_ratio=0.3)