import cv2
import numpy as np
import os
from ultralytics import YOLO

def test_person_segmentation(input_dir, output_dir, model_weight='yolov8n-seg.pt'):
    """
    测试 YOLOv8 实例分割效果，并将提取的人物保存为透明背景的 PNG 图片。
    
    参数:
        input_dir: 测试图片所在文件夹
        output_dir: 提取结果保存的文件夹
        model_weight: YOLOv8 模型权重。可选 'yolov8n-seg.pt' (最快), 'yolov8s-seg.pt' (较均衡), 'yolov8m-seg.pt' (更精确)
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    # 加载模型 (首次运行会自动下载权重)
    print(f"🚀 正在加载模型: {model_weight} ...")
    model = YOLO(model_weight)

    # 获取输入目录下所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中没有找到图片文件。")
        return

    print(f"🔍 找到 {len(image_files)} 张图片，开始处理...")

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ 无法读取图片: {img_name}，已跳过。")
            continue

        # 执行推理：classes=[0] 强制指定只提取 COCO 数据集中的第 0 类（Person）
        # retina_masks=True 可以在后处理时获得更高分辨率的 mask
        results = model(img, classes=[0], retina_masks=True, verbose=False)
        
        # 创建一个全黑的掩码画布（单通道）
        mask_canvas = np.zeros(img.shape[:2], dtype=np.uint8)

        # 检查是否检测到了人物
        if results[0].masks is not None:
            # YOLOv8 提供了多边形坐标 (masks.xy)，使用 cv2.fillPoly 绘制 mask 会比直接 resize tensor 边缘更锐利
            for contour in results[0].masks.xy:
                # 将坐标转为 int32 格式供 OpenCV 使用
                contour = np.array(contour, dtype=np.int32)
                cv2.fillPoly(mask_canvas, [contour], 255)
        else:
            print(f"  [-] {img_name} 中未检测到人物。")
            # 即使没检测到人，也保存一张空图以便核对
        
        # --- 制作带有透明通道 (Alpha) 的 PNG ---
        # 1. 将原图的 BGR 转换为 BGRA (增加透明通道)
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # 2. 将 mask_canvas 作为 Alpha 通道赋值给图片
        # 掩码中为 255 的地方（人物）完全不透明，为 0 的地方（背景）完全透明
        img_bgra[:, :, 3] = mask_canvas

        # 生成输出路径，强制保存为 .png
        base_name = os.path.splitext(img_name)[0]
        output_name = f"{base_name}_seg.png"
        output_path = os.path.join(output_dir, output_name)

        # 保存结果
        cv2.imwrite(output_path, img_bgra)
        print(f"  [+] 成功处理并保存: {output_name}")

    print("✅ 所有图片测试完成！")

# ================= 脚本入口 =================
if __name__ == "__main__":
    # --- 请在这里修改你的测试路径 ---
    # 支持相对路径或绝对路径
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_output_masks"
    
    # 建议测试策略：
    # 1. 先用 'yolov8n-seg.pt' 测算力（它计算极快）。
    # 2. 如果觉得边缘细节（比如手部、设备边缘）不够好，换用 'yolov8s-seg.pt' 或 'yolov8m-seg.pt'。
    # 影视级应用通常需要牺牲一点速度换取边缘质量。
    MODEL_VERSION = './checkpoints/yolo/yolov8s-seg.pt' 
    
    # 自动创建输入测试文件夹（防止第一次运行报错）
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"💡 已自动创建测试输入目录 '{INPUT_FOLDER}'，请放入几张图片后重新运行此脚本。")
    else:
        test_person_segmentation(INPUT_FOLDER, OUTPUT_FOLDER, model_weight=MODEL_VERSION)