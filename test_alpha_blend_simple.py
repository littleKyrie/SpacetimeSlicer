import cv2
import numpy as np

print("=== 测试 Alpha 混合逻辑 ===\n")

w, h = 200, 200

canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
cv2.rectangle(canvas, (50, 50), (150, 150), (0, 0, 255), -1)

current_frame = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
cv2.rectangle(current_frame, (60, 60), (140, 140), (0, 255, 0), -1)

alpha_mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(alpha_mask, (70, 70), (130, 130), 128, -1)

print(f"canvas (红色方块):\n")
print(f"current_frame (绿色方块):\n")
print(f"alpha_mask (中心128=50%透明度, 周围0=完全透明):\n")

alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0
print(f"alpha_normalized shape: {alpha_normalized.shape}")
print(f"alpha_normalized min: {alpha_normalized.min()}, max: {alpha_normalized.max()}\n")

result = (current_frame * alpha_normalized + canvas * (1 - alpha_normalized)).astype(np.uint8)

cv2.imwrite("test_canvas.png", canvas)
cv2.imwrite("test_current.png", current_frame)
cv2.imwrite("test_alpha_mask.png", alpha_mask)
cv2.imwrite("test_result.png", result)

print("测试图片已保存：")
print("  test_canvas.png - 红色画布")
print("  test_current.png - 绿色前景")
print("  test_alpha_mask.png - Alpha掩码(中心半透明)")
print("  test_result.png - 混合结果")
print("\n如果混合正确，test_result.png 应该显示：")
print("  - 中心区域：红色和绿色的混合（偏黄/橙色）")
print("  - 周围区域：纯红色（因为alpha=0，完全透明）")

print("\n手动验证：用图像软件打开这4张图片对比看是否正确")
