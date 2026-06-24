import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

def test_ultimate_pose_tracking(input_dir, output_dir, sam2_checkpoint, sam2_model_cfg):
    """
    终极全自动方案：YOLO-Pose 提取躯干+双手腕 -> SAM 2 多点精准追踪
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True

    print(f"🚀 使用设备: {device}")
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # =========================================================
    # 阶段 1：YOLO-Pose 自动扫描黄金帧并提取骨骼点
    # =========================================================
    print("🤖 [阶段 1/3] YOLO-Pose 正在扫描关键帧并提取多重锚点(躯干+手腕)...")
    
    # 核心改变：加载姿态估计模型
    # pose_model = YOLO('yolov8n-pose.pt') 
    pose_model = YOLO('./checkpoints/yolov8n-pose.pt')
    
    max_area = 0
    golden_frame_idx = 0
    golden_keypoints = None
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        results = pose_model(img, verbose=False)
        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            # 找到动作张力最大的帧
            if area > max_area:
                max_area = area
                golden_frame_idx = i
                # 获取该帧的人物 17 个关键点坐标 (x, y, confidence)
                golden_keypoints = results[0].keypoints.data[0].cpu().numpy()

    if golden_keypoints is None:
        print("❌ 未检测到人物骨骼！")
        return

    print(f"🎯 锁定黄金关键帧: 第 {golden_frame_idx + 1} 帧")

    # =========================================================
    # 阶段 2：智能组装 SAM 2 多点提示
    # =========================================================
    prompts = []
    
    # YOLO 骨骼序号：5=左肩, 6=右肩, 11=左胯, 12=右胯
    # 1. 计算躯干中心 (最稳定的主体锚点)
    try:
        torso_x = (golden_keypoints[5][0] + golden_keypoints[6][0] + golden_keypoints[11][0] + golden_keypoints[12][0]) / 4
        torso_y = (golden_keypoints[5][1] + golden_keypoints[6][1] + golden_keypoints[11][1] + golden_keypoints[12][1]) / 4
        if torso_x > 0 and torso_y > 0:
            prompts.append([torso_x, torso_y])
            print("  📍 锁定锚点 1: 躯干中心")
    except:
        pass

    # YOLO 骨骼序号：9=左手腕, 10=右手腕
    # 2. 提取手腕坐标 (强制捕获手中的扇子或水袖)
    l_wrist = golden_keypoints[9]
    if l_wrist[0] > 0 and l_wrist[1] > 0 and l_wrist[2] > 0.5: # 确保置信度大于0.5
        prompts.append([l_wrist[0], l_wrist[1]])
        print("  📍 锁定锚点 2: 左手腕 (抓取左手道具)")
        
    r_wrist = golden_keypoints[10]
    if r_wrist[0] > 0 and r_wrist[1] > 0 and r_wrist[2] > 0.5:
        prompts.append([r_wrist[0], r_wrist[1]])
        print("  📍 锁定锚点 3: 右手腕 (抓取右手道具)")

    points_np = np.array(prompts, dtype=np.float32)
    # 所有提供的点都是正向保留点 (Label=1)
    labels_np = np.ones(len(prompts), dtype=np.int32) 

    # =========================================================
    # 阶段 3：SAM 2 多点双向追踪与视频生成
    # =========================================================
    print("📦 [阶段 2/3] 准备临时目录...")
    temp_sam2_dir = os.path.join(output_dir, ".temp_sam2_frames")
    os.makedirs(temp_sam2_dir, exist_ok=True)
    frame_mapping = {}
    for i, img_name in enumerate(image_files):
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(temp_sam2_dir, f"{i:05d}{os.path.splitext(img_name)[1]}"))
        frame_mapping[i] = img_name

    print("🧠 [阶段 3/3] SAM 2 正在基于多锚点进行双向追踪...")
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=temp_sam2_dir)
    predictor.reset_state(inference_state)
    
    # 将多个锚点一次性喂给 SAM 2
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=golden_frame_idx,
        obj_id=1,
        points=points_np,
        labels=labels_np
    )
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        original_img_name = frame_mapping[out_frame_idx]
        current_img = cv2.imread(os.path.join(input_dir, original_img_name))

        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        mask_canvas = (mask * 255).astype(np.uint8)

        img_bgra = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask_canvas

        output_name = f"{os.path.splitext(original_img_name)[0]}_pose_auto.png"
        cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
        print(f"  [+] 追踪并保存: {output_name} [{out_frame_idx+1}/{len(image_files)}]", end='\r')

    print("\n🧹 清理临时文件...")
    shutil.rmtree(temp_sam2_dir)
    print("\n✅ 全自动完美分割完成！无需任何人工交互！")

if __name__ == "__main__":
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_pose_auto"
    
    SAM2_CHECKPOINT = "./checkpoints/sam2/sam2.1_hiera_large.pt"
    SAM2_CFG = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    
    test_ultimate_pose_tracking(INPUT_FOLDER, OUTPUT_FOLDER, SAM2_CHECKPOINT, SAM2_CFG)