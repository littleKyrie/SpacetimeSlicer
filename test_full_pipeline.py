import cv2
import numpy as np

print("=== 测试完整流程：PNG保存 -> 读取 -> 混合 ===")

# 创建测试数据
h, w = 50, 50

# canvas: 红色背景
canvas = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)
print(f"canvas: 红色背景")

# current_frame: 绿色前景
current_frame = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)
print(f"current_frame: 绿色前景")

# alpha_mask: 中心渐变透明度
alpha_mask = np.zeros((h, w), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        # 计算到中心的距离
        dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        # 中心透明度高，边缘透明度低
        alpha = max(0, min(255, int(255 - dist * 5)))
        alpha_mask[y, x] = alpha

print(f"\nalpha_mask: 中心渐变 (中心255, 边缘0)")
print(f"  alpha_mask min={alpha_mask.min()}, max={alpha_mask.max()}")

# 保存为 PNG
img_bgra = cv2.cvtColor(current_frame, cv2.COLOR_BGR2BGRA)
img_bgra[:, :, 3] = alpha_mask
cv2.imwrite("test_png_output.png", img_bgra)
print("\n已保存 test_png_output.png")

# 读取 PNG
read_png = cv2.imread("test_png_output.png", cv2.IMREAD_UNCHANGED)
print(f"\n读取的 PNG: shape={read_png.shape}")

if read_png.shape[2] == 4:
    read_bgr = read_png[:, :, :3]
    read_alpha = read_png[:, :, 3]
    print(f"  BGR 通道: shape={read_bgr.shape}")
    print(f"  Alpha 通道: shape={read_alpha.shape}")
    print(f"  Alpha 值: min={read_alpha.min()}, max={read_alpha.max()}")
else:
    print("❌ PNG 读取没有 Alpha 通道！")

# 使用读取的 alpha 进行混合
alpha_normalized = np.repeat(read_alpha[:, :, np.newaxis], 3, axis=2) / 255.0
result = (read_bgr * alpha_normalized + canvas * (1 - alpha_normalized)).astype(np.uint8)

cv2.imwrite("test_mixed_result.png", result)
print("\n已保存 test_mixed_result.png")

# 测试不同透明度区域
center_alpha = read_alpha[h//2, w//2]
edge_alpha = read_alpha[0, 0]
print(f"\n中心透明度: {center_alpha}")
print(f"边缘透明度: {edge_alpha}")

center_pixel = result[h//2, w//2]
edge_pixel = result[0, 0]
print(f"\n中心像素: {center_pixel} (应该偏绿)")
print(f"边缘像素: {edge_pixel} (应该偏红)")

# 验证
if center_pixel[1] > center_pixel[2]:  # 绿色分量 > 红色分量
    print("\n✅ 中心区域混合正确 (偏绿)")
else:
    print("\n❌ 中心区域混合有问题")

if edge_pixel[2] > edge_pixel[1]:  # 红色分量 > 绿色分量
    print("✅ 边缘区域混合正确 (偏红)")
else:
    print("❌ 边缘区域混合有问题")

print("\n提示：用图像软件打开 test_png_output.png 和 test_mixed_result.png 对比")
