import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rvm import RVMStrategy
import torch

print("=== 测试抠图和融合逻辑 ===\n")

# 准备测试数据
INPUT_DIR = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"
if not os.path.exists(INPUT_DIR):
    print(f"错误：找不到输入目录 {INPUT_DIR}")
    exit(1)

image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"找到 {len(image_files)} 张图片\n")

if len(image_files) < 3:
    print("图片太少，无法测试")
    exit(1)

# 初始化分割策略
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}\n")
strategy = RVMStrategy(device)

# 读取前3帧进行测试
frame1 = cv2.imread(os.path.join(INPUT_DIR, image_files[0]))
frame2 = cv2.imread(os.path.join(INPUT_DIR, image_files[1]))
frame3 = cv2.imread(os.path.join(INPUT_DIR, image_files[2]))

print(f"帧尺寸：{frame1.shape}\n")

# 抠图
print("开始抠图...")
alpha1 = strategy.process_frame(frame1, 0)
alpha2 = strategy.process_frame(frame2, 1)
alpha3 = strategy.process_frame(frame3, 2)
print("抠图完成\n")

# 保存alpha掩码
cv2.imwrite("alpha_test_1.png", alpha1)
cv2.imwrite("alpha_test_2.png", alpha2)
cv2.imwrite("alpha_test_3.png", alpha3)
print("保存 alpha 掩码到 alpha_test_*.png\n")

# 测试融合逻辑
print("测试融合逻辑...")

# 初始化canvas
canvas = frame1.copy()

# 方式1：当前代码的alpha混合
alpha1_normalized = alpha1[:, :, np.newaxis] / 255.0
print(f"alpha1_normalized shape: {alpha1_normalized.shape}")
frame1_blended = (frame1 * alpha1_normalized + canvas * (1 - alpha1_normalized)).astype(np.uint8)
cv2.imwrite("blend_test_alpha.png", frame1_blended)

# 方式2：np.where
mask_binary = (alpha1 > 0)[:, :, np.newaxis]
mask_binary_3ch = np.repeat(mask_binary, 3, axis=2)
frame1_where = np.where(mask_binary_3ch, frame1, canvas)
cv2.imwrite("blend_test_where.png", frame1_where)

# 方式3：保存原始PNG后读取回来测试
img_bgra = cv2.cvtColor(frame1, cv2.COLOR_BGR2BGRA)
img_bgra[:, :, 3] = alpha1
cv2.imwrite("blend_test_alpha_png.png", img_bgra)

# 测试连续叠加
canvas = frame1.copy()

# 第1帧作为残影
alpha_normalized = alpha1[:, :, np.newaxis] / 255.0
canvas = (frame1 * alpha_normalized + canvas * (1 - alpha_normalized)).astype(np.uint8)

# 第2帧
alpha_normalized = alpha2[:, :, np.newaxis] / 255.0
frame2_output = (frame2 * alpha_normalized + canvas * (1 - alpha_normalized)).astype(np.uint8)
cv2.imwrite("blend_test_frame2.png", frame2_output)

print("测试完成，保存了 blend_test_*.png 文件")
