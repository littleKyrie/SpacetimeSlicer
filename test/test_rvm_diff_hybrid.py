import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image

def compute_median_background(input_dir, image_files, sample_step=5):
    """
    计算静态背景图 (中值法)
    提取部分帧计算中值，自动过滤掉移动的人物，得到一张纯净的空舞台背景。
    """
    print("🖼️ [阶段 1/4] 正在计算舞台纯净背景 (Median Background)...")
    frames = []
    # 每隔几帧取样一次，节省内存并加快速度
    for i in range(0, len(image_files), sample_step):
        img_path = os.path.join(input_dir, image_files[i])
        img = cv2.imread(img_path)
        # 转为灰度图进行计算，减少计算量，对亮度变化更敏感
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    # 堆叠并计算中值
    median_bg = np.median(frames, axis=0).astype(np.uint8)
    return median_bg

def test_hybrid_matting(input_dir, output_dir, diff_threshold=25):
    """
    RVM + 背景差分连通域融合方案
    diff_threshold: 差分阈值，越小对暗色道具越敏感，但噪点也会增多 (建议 20-35)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files: return

    # 1. 计算背景
    median_bg = compute_median_background(input_dir, image_files)
    
    # 2. 加载 RVM 模型 (负责身体完美边缘)
    print("🧠 [阶段 2/4] 加载 RVM 模型获取高质量人体核心...")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
    model = model.to(device).eval()
    
    rec = [None] * 4 
    transform = transforms.ToTensor()

    print(f"🎬 [阶段 3/4] 开始逐帧进行物理与语义融合推理...")
    
    with torch.no_grad():
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(input_dir, img_name)
            current_img = cv2.imread(img_path)
            
            # --- A. 获取 RVM 核心 Mask ---
            pil_img = Image.open(img_path).convert('RGB')
            tensor_img = transform(pil_img).unsqueeze(0).to(device)
            fgr, pha, *rec = model(tensor_img, *rec, downsample_ratio=0.25)
            
            # rvm_mask: 0~255 的 Alpha 图
            rvm_mask = (pha[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rvm_mask_2d = rvm_mask[:, :, 0]
            
            # 将 RVM 的 Alpha 转为严格的二值核心 (阈值设高点，确保内部扎实)
            _, rvm_core = cv2.threshold(rvm_mask_2d, 127, 255, cv2.THRESH_BINARY)

            # --- B. 计算物理运动差分 Mask (寻找扇子) ---
            gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            # 计算绝对差值
            diff = cv2.absdiff(gray_current, median_bg)
            # 二值化，提取出所有和背景不一样的运动像素 (包括人、扇子、地上的影子)
            _, diff_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            
            # 稍微膨胀差分 Mask，确保细小的扇骨连接不断裂
            kernel = np.ones((5, 5), np.uint8)
            diff_mask = cv2.dilate(diff_mask, kernel, iterations=1)

            # --- C. 连通域融合 (魔法时刻：去留判定) ---
            # 寻找差分 Mask 中的所有独立连通块 (Contours)
            contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建一个空白画布，用来装最终被认可的运动区域
            validated_diff_mask = np.zeros_like(diff_mask)
            
            for contour in contours:
                # 过滤掉极小的噪点碎屑
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # 提取当前连通块的掩码
                temp_mask = np.zeros_like(diff_mask)
                cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)
                
                # 核心判定：这个运动色块，跟 RVM 识别出的人体有交集吗？
                # 如果有交集 (bitwise_and > 0)，说明它是拿在手里的扇子或穿在身上的衣服
                intersection = cv2.bitwise_and(temp_mask, rvm_core)
                if cv2.countNonZero(intersection) > 0:
                    # 将其加入合法画布
                    validated_diff_mask = cv2.bitwise_or(validated_diff_mask, temp_mask)

            # --- D. 最终合成 ---
            # 融合 RVM 柔和的高质量边缘和补回来的道具硬边缘
            # validated_diff_mask 找回了扇子，rvm_mask_2d 提供了发丝/衣摆的 Alpha
            final_alpha = cv2.max(rvm_mask_2d, validated_diff_mask)
            
            # 边缘稍微平滑处理一下
            final_alpha = cv2.GaussianBlur(final_alpha, (3, 3), 0)

            # 提取原图作为前景
            img_bgra = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
            img_bgra[:, :, 3] = final_alpha

            output_name = f"{os.path.splitext(img_name)[0]}_hybrid.png"
            cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
            
            print(f"  [+] 融合处理完毕: {output_name} [{i+1}/{len(image_files)}]", end='\r')

    print("\n✅ 物理差分与语义融合处理完成！")

if __name__ == "__main__":
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_hybrid_matting"
    
    # 执行处理 (如果发现扇子还是没出来，可以把 25 调小到 15；如果有背景噪点，调大到 35)
    test_hybrid_matting(INPUT_FOLDER, OUTPUT_FOLDER, diff_threshold=120)