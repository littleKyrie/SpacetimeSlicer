import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def test_rvm_matting(input_dir, output_dir):
    """
    使用 Robust Video Matting (RVM) 提取带 Alpha 通道的精细人像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")

    # 直接从 PyTorch Hub 加载 RVM 模型
    # resnet50 版本精度最高，mobilenetv3 版本速度最快
    print("正在从 TorchHub 加载 RVM (ResNet50) 模型，首次运行需下载...")
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
    model = model.to(device).eval()

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # RVM 的核心：循环记忆状态，初始化为 None
    rec = [None] * 4 
    
    # 定义图像预处理
    transform = transforms.ToTensor()

    print(f"🔍 开始连续处理 {len(image_files)} 帧...")
    
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            
            # 使用 PIL 读取 RGB，并转换为 Tensor
            pil_img = Image.open(img_path).convert('RGB')
            tensor_img = transform(pil_img).unsqueeze(0).to(device) # Shape: (1, 3, H, W)

            # downsample_ratio=0.25 (1080p推荐使用0.25，4K推荐0.125) 
            # 帮助模型更好地把握全局上下文，同时提升速度
            fgr, pha, *rec = model(tensor_img, *rec, downsample_ratio=0.25)

            # 取出 Alpha 通道 (pha) 和 前景颜色 (fgr)
            # RVM 牛逼之处在于它不仅能预测透明度，还能预测“被背景颜色污染前的前景颜色”
            alpha = pha[0].cpu().numpy().transpose(1, 2, 0) # Shape: (H, W, 1)
            foreground = fgr[0].cpu().numpy().transpose(1, 2, 0) # Shape: (H, W, 3)

            # 将前处理转回 OpenCV 的 BGR 格式
            foreground_bgr = cv2.cvtColor((foreground * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            alpha_uint8 = (alpha * 255).astype(np.uint8)

            # 合成最终的 BGRA 图片
            # 注意：这里我们使用 RVM 预测出的 pure foreground，而不是原图，可以解决边缘反光问题
            img_bgra = np.concatenate([foreground_bgr, alpha_uint8], axis=2)

            output_name = f"{os.path.splitext(img_name)[0]}_rvm.png"
            cv2.imwrite(os.path.join(output_dir, output_name), img_bgra)
            print(f"  [+] 成功处理 (融合Alpha): {output_name}")

if __name__ == "__main__":
    INPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"   
    OUTPUT_FOLDER = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\test_rvm_alpha"
    
    test_rvm_matting(INPUT_FOLDER, OUTPUT_FOLDER)