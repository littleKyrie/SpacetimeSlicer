import cv2
import numpy as np

print("=== 测试 Alpha 混合逻辑 ===")
print("目标：验证 frame_output = current_frame * alpha + canvas * (1 - alpha) 是否正确工作\n")

# 创建测试数据
h, w = 10, 10

# canvas: 红色背景 (0, 0, 255)
canvas = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)
print(f"canvas (红色): shape={canvas.shape}, value={canvas[0, 0]}")

# current_frame: 绿色前景 (0, 255, 0)
current_frame = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)
print(f"current_frame (绿色): shape={current_frame.shape}, value={current_frame[0, 0]}")

# alpha_mask: 50% 透明度 (128)
alpha_mask = np.full((h, w), 128, dtype=np.uint8)
print(f"alpha_mask (50%): shape={alpha_mask.shape}, value={alpha_mask[0, 0]}")

# 扩展成3通道
alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0
print(f"\nalpha_normalized: shape={alpha_normalized.shape}, value={alpha_normalized[0, 0]}")

# 执行混合
result = (current_frame * alpha_normalized + canvas * (1 - alpha_normalized)).astype(np.uint8)
print(f"\n混合结果: shape={result.shape}, value={result[0, 0]}")

# 预期结果
expected = np.array([0, 127, 127], dtype=np.uint8)  # 绿+红混合
print(f"预期结果: {expected}")

# 验证
if np.allclose(result[0, 0], expected, atol=1):
    print("\n✅ Alpha 混合逻辑正确！")
else:
    print(f"\n❌ Alpha 混合逻辑有问题！")
    print(f"   实际: {result[0, 0]}")
    print(f"   预期: {expected}")

# 测试边缘情况
print("\n=== 测试边缘情况 ===")

# 测试 alpha=0 (完全透明)
alpha_mask_0 = np.zeros((h, w), dtype=np.uint8)
alpha_norm_0 = np.repeat(alpha_mask_0[:, :, np.newaxis], 3, axis=2) / 255.0
result_0 = (current_frame * alpha_norm_0 + canvas * (1 - alpha_norm_0)).astype(np.uint8)
print(f"\nalpha=0 (完全透明):")
print(f"  结果: {result_0[0, 0]}")
print(f"  预期: {canvas[0, 0]} (应该等于 canvas)")

# 测试 alpha=255 (完全不透明)
alpha_mask_255 = np.full((h, w), 255, dtype=np.uint8)
alpha_norm_255 = np.repeat(alpha_mask_255[:, :, np.newaxis], 3, axis=2) / 255.0
result_255 = (current_frame * alpha_norm_255 + canvas * (1 - alpha_norm_255)).astype(np.uint8)
print(f"\nalpha=255 (完全不透明):")
print(f"  结果: {result_255[0, 0]}")
print(f"  预期: {current_frame[0, 0]} (应该等于 current_frame)")

# 保存测试图片
cv2.imwrite("test_alpha_50.png", result)
cv2.imwrite("test_alpha_0.png", result_0)
cv2.imwrite("test_alpha_255.png", result_255)
print("\n测试图片已保存，可以用图像软件验证")
