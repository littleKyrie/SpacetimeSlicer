import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import time

# 分割算法接口 (Strategy Pattern)
from models.rvm import RVMStrategy
from models.hybrid_rvm import HybridStrategy
from models.yolo_sam2 import YOLO_SAM2_Strategy


# 时空切片合成系统引擎
class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25, ghost_interval=1):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.ghost_interval = ghost_interval
        self.image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.total_frames = len(self.image_files)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True

    def read_frame(self, idx):
        return cv2.imread(os.path.join(self.input_dir, self.image_files[idx]))

    def generate(self, method_name, effect_start_idx, effect_end_idx, ghost_interval=1):
        """生成时空切片视频主循环，并保存带有透明通道的中间提取帧

        Args:
            method_name: 分割方案 ('RVM', 'Hybrid', 'SAM2_BBox')
            effect_start_idx: 特效起始帧 (包含)
            effect_end_idx: 特效结束帧 (不包含)
            ghost_interval: 每隔几帧保留一次残影 (默认1表示每帧都保留, 2表示每隔1帧保留, 3表示每隔2帧保留, 以此类推)
        """
        print(f"\n{'='*50}\n🚀 开始制作时空切片: [{method_name}] ({effect_start_idx} -> {effect_end_idx}), 残影间隔={ghost_interval}\n{'='*50}")

        if effect_start_idx < 0 or effect_end_idx >= self.total_frames or effect_start_idx >= effect_end_idx:
            raise ValueError("帧范围参数错误！")
        if ghost_interval < 1:
            raise ValueError("ghost_interval 必须 >= 1")

        # 1. 初始化具体的策略
        if method_name == 'RVM':
            strategy = RVMStrategy(self.device)
        elif method_name == 'Hybrid':
            bg = self._get_median_background()
            strategy = HybridStrategy(self.device, bg, diff_threshold=35) 
        elif method_name == 'SAM2_BBox':
            strategy = YOLO_SAM2_Strategy(self.device)
        else:
            raise ValueError(f"未知的方案: {method_name}")

        # 2. 准备输出目录
        run_name = f"{method_name}_{effect_start_idx}-{effect_end_idx}"
        output_dir = os.path.join(self.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # ====== 🌟 修改点 1：创建保存纯净提取人物 PNG 的子目录 ======
        extracted_pngs_dir = os.path.join(output_dir, "extracted_pngs")
        os.makedirs(extracted_pngs_dir, exist_ok=True)
        
        # (可选) 如果你还想保留累加后的画布效果图片，可以取消下面这两行的注释
        # canvas_frames_dir = os.path.join(output_dir, "cumulative_canvas")
        # os.makedirs(canvas_frames_dir, exist_ok=True)
        
        video_path = os.path.join(output_dir, f"slicer_{run_name}.mp4")
        
        # 3. 初始化视频写入器
        sample_frame = self.read_frame(0)
        h, w = sample_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        # 写入片头正常视频
        print("🎞️ 写入片头...")
        for i in range(0, effect_start_idx):
            out.write(self.read_frame(i))

        # =======================================================
        # 4. 核心：制作时空切片累加效果，并保存纯净 PNG
        # =======================================================
        print(f"✨ 制作时空切片特效段并保存资产 (残影间隔={ghost_interval})...")
        background = self.read_frame(0).copy()
        canvas_ghosts = background.copy()
        ghost_count = 0

        for i in range(effect_start_idx, effect_end_idx):
            current_frame = self.read_frame(i)

            alpha_mask = strategy.process_frame(current_frame, i)

            img_bgra = cv2.cvtColor(current_frame, cv2.COLOR_BGR2BGRA)
            img_bgra[:, :, 3] = alpha_mask
            png_filename = os.path.join(extracted_pngs_dir, f"extracted_{i:05d}.png")
            cv2.imwrite(png_filename, img_bgra)

            should_be_ghost = (i == effect_start_idx) or ((i - effect_start_idx) % ghost_interval == 0)

            alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0
            mask_3d = np.repeat((alpha_mask > 0)[:, :, np.newaxis], 3, axis=2)

            frame_output = canvas_ghosts.copy()
            frame_output = np.where(mask_3d, current_frame, frame_output)

            out.write(frame_output)

            if should_be_ghost:
                canvas_ghosts = np.where(mask_3d, current_frame, canvas_ghosts)
                ghost_count += 1

            print(f"  > 渲染特效帧 & 保存 PNG: {i}/{effect_end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        # 5. 写入片尾正常视频
        print("\n🎞️ 写入片尾...")
        for i in range(effect_end_idx, self.total_frames):
            current_frame = self.read_frame(i)
            out.write(current_frame)

        out.release()
        print(f"✅ 视频及纯净 PNG 资产已输出！保存在: {output_dir}")
        print(f"   残影参数: ghost_interval={ghost_interval}, 实际添加了 {ghost_count} 个残影")


if __name__ == "__main__":
    start_time = time.time()

    INPUT_DIR = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"
    OUTPUT_ROOT = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\results_slicer"

    START_FRAME = 25
    END_FRAME = 220
    GHOST_INTERVAL = 3

    slicer = SpacetimeSlicer(INPUT_DIR, OUTPUT_ROOT, fps=25, ghost_interval=GHOST_INTERVAL)

    slicer.generate('RVM', START_FRAME, END_FRAME, ghost_interval=GHOST_INTERVAL)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"total generation time: {total_time}s")