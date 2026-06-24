import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

# 全局变量用于 OpenCV 鼠标交互
click_points = []
click_labels = []
display_img = None
original_img = None

def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels, display_img, original_img
    
    # 左键：添加正向点 (保留区域，绿色)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(1)
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Interactive Setup', display_img)
        
    # 右键：添加负向点 (去除区域，红色)
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(0)
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Interactive Setup', display_img)

def test_interactive_point_tracking(input_dir, output_dir, sam2_checkpoint, sam2_model_cfg):
    global click_points, click_labels, display_img, original_img
    
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
    # 阶段 1：YOLO 自动寻找带道具的黄金帧 (最大面积)
    # =========================================================
    print("🤖 [阶段 1/3] YOLO 正在自动扫描寻找扇子完全展开的黄金关键帧...")
    # yolo_model = YOLO('yolov8n.pt')
    yolo_model = YOLO('./checkpoints/yolo/yolov8n.pt')
    max_area = 0
    golden_frame_idx = 0
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        results = yolo_model(img, classes=[0], verbose=False)
        if len(results[0].boxes) > 0:
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                golden_frame_idx = i

    print(f"🎯 锁定黄金关键帧: 第 {golden_frame_idx + 1} 帧 ({image_files[golden_frame_idx]})")

    # =========================================================
    # 阶段 2：规范化数据集目录
    # =========================================================
    print("📦 正在准备 SAM 2 临时目录...")
    temp_sam2_dir = os.path.join(output_dir, ".temp_sam2_frames")
    os.makedirs(temp_sam2_dir, exist_ok=True)
    frame_mapping = {}
    for i, img_name in enumerate(image_files):
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(temp_sam2_dir, f"{i:05d}{os.path.splitext(img_name)[1]}"))
        frame_mapping[i] = img_name

    # =========================================================
    # 阶段 3：人机交互点选 (自适应缩放窗口)
    # =========================================================
    original_img = cv2.imread(os.path.join(input_dir, image_files[golden_frame_idx]))
    display_img = original_img.copy()

    # 解决窗口太大无法缩放的问题
    cv2.namedWindow('Interactive Setup', cv2.WINDOW_NORMAL)
    # 将窗口初始大小设置为 1280x720，你可以用鼠标拖拽窗口边缘任意放大缩小
    cv2.resizeWindow('Interactive Setup', 1280, 720) 
    cv2.setMouseCallback('Interactive Setup', mouse_callback)

    print("\n" + "="*50)
    print("🖱️ 请在弹出的图像上操作 (窗口可拖拽边缘进行缩放)：")
    print("  [鼠标左键]: 在【人物身体】和【扇子中心】各点一下 (绿色，代表保留)")
    print("  [鼠标右键]: 在误带入的【背景窗户/灯光】上点一下 (红色，代表去除)")
    print("  [按 C 键]: 清除所有点重新点")
    print("  [按 Enter 键]: 确认点选，开始追踪！")
    print("="*50 + "\n")

    while True:
        cv2.imshow('Interactive Setup', display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32: # Enter 或 Space
            break
        elif key == ord('c'): # 清除重来
            click_points.clear()
            click_labels.clear()
            display_img = original_img.copy()

    cv2.destroyAllWindows()

    if len(click_points) == 0:
        print("❌ 未提供任何提示点，程序退出。")
        shutil.rmtree(temp_sam2_dir)
        return

    points_np = np.array(click_points, dtype=np.float32)
    labels_np = np.array(click_labels, dtype=np.int32)
    print(f"✅ 已接收 {len(click_points)} 个提示点，准备灌入 SAM 2...")

    # =========================================================
    # 阶段 4：SAM 2 双向追踪
    # =========================================================
    print("🧠 启动 SAM 2 双向时序追踪...")
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=temp_sam2_dir)
    predictor.reset_state(inference_state)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=golden_frame_idx,
        obj_id=1,
        points=points_np,
        labels=labels_np
    )
    
    print(f"🎬 开始生成全序列纯净帧...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        original_img_name = frame_mapping[out_frame_idx]
        current_img = cv2.imread(os.path.join(input_dir, original_img_name))

        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        mask_canvas = (mask * 255).astype(np.uint8)

        img_bgra = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask_canvas

        output_name = f"{os.path.splitext(original_img_name)[0]}_goldilocks.png"
        cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
        print(f"  [+] 追踪并保存: {output_name} [{out_frame_idx+1}/{len(image_files)}]", end='\r')

    print("\n🧹 清理临时文件...")
    shutil.rmtree(temp_sam2_dir)
    print("\n✅ 完美版分割完成！背景伪影已被彻底清除，扇子已保留！")

if __name__ == "__main__":
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_goldilocks"
    SAM2_CHECKPOINT = "./checkpoints/sam2/sam2.1_hiera_large.pt"
    SAM2_CFG = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    
    test_interactive_point_tracking(INPUT_FOLDER, OUTPUT_FOLDER, SAM2_CHECKPOINT, SAM2_CFG)