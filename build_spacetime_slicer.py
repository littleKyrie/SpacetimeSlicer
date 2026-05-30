import cv2
import os
import time
import numpy as np
import torch
from models.rvm import RVMStrategy
from models.hybrid_rvm import HybridStrategy
from models.yolo_sam2 import YOLO_SAM2_Strategy

class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.frame_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.total_frames = len(self.frame_paths)

    def read_frame(self, idx):
        return cv2.imread(self.frame_paths[idx])

    def generate(self, method_name, effect_start_idx, effect_end_idx, ghost_interval=1, edge_feather=0, fade_duration_frames=None):
        """
        生成时空切片视频
        :param method_name: 分割方法名 ('RVM', 'Hybrid', 'SAM2_BBox')
        :param effect_start_idx: 特效开始帧
        :param effect_end_idx: 特效结束帧 (不包含)
        :param ghost_interval: 每隔几帧保留一次残影 (默认1表示每帧都保留, 2表示每隔1帧保留, 以此类推)
        :param edge_feather: 边缘处理 (正值=羽化, 负值=腐蚀)
        :param fade_duration_frames: 片尾残影淡出持续帧数 (默认None表示使用 ghost_interval * 2)
        """
        run_name = f"{method_name}_{effect_start_idx}_{effect_end_idx}"
        output_dir = os.path.join(self.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_pngs_dir = os.path.join(output_dir, "extracted_pngs")
        os.makedirs(extracted_pngs_dir, exist_ok=True)
        
        video_path = os.path.join(output_dir, f"slicer_{run_name}.mp4")
        
        sample_frame = self.read_frame(0)
        h, w = sample_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        print("🎞️ 写入片头...")
        for i in range(0, effect_start_idx):
            out.write(self.read_frame(i))

        # =======================================================
        # 4. 核心：制作时空切片累加效果，并保存纯净 PNG
        # =======================================================
        print(f"✨ 制作时空切片特效段并保存资产 (残影间隔={ghost_interval}, 边缘处理={edge_feather})...")
        background = self.read_frame(0).copy()
        canvas_ghosts = background.copy()
        ghost_count = 0
        
        # 保存所有残影信息（用于片尾按顺序消失）
        ghost_list = []

        for i in range(effect_start_idx, effect_end_idx):
            current_frame = self.read_frame(i)

            alpha_mask = strategy.process_frame(current_frame, i)

            if edge_feather < 0:
                kernel = np.ones((3, 3), np.uint8)
                alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

            # PNG 保存已禁用，需要时取消注释
            # img_bgra = cv2.cvtColor(current_frame, cv2.COLOR_BGR2BGRA)
            # img_bgra[:, :, 3] = alpha_mask
            # png_filename = os.path.join(extracted_pngs_dir, f"extracted_{i:05d}.png")
            # cv2.imwrite(png_filename, img_bgra)

            should_be_ghost = (i == effect_start_idx) or ((i - effect_start_idx) % ghost_interval == 0)

            alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0
            mask_binary = np.repeat((alpha_mask > 0)[:, :, np.newaxis], 3, axis=2)

            frame_output = (current_frame * alpha_normalized + canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)

            out.write(frame_output)

            if should_be_ghost:
                canvas_ghosts = frame_output#np.where(mask_binary, current_frame, canvas_ghosts)#这里也要用alpha_normalized来做融合，而不是直接把current frame覆盖上去
                ghost_list.append({
                    'frame': current_frame.copy(),
                    'alpha': alpha_mask.copy()
                })
                ghost_count += 1

            print(f"  > 渲染特效帧 & 保存 PNG: {i}/{effect_end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        # 5. 写入片尾正常视频（带残影淡出效果）
        print("\n🎞️ 写入片尾（定格淡出 + 继续播放）...")
        
        # 获取最后一帧画面（用于定格）
        last_frame = self.read_frame(effect_end_idx - 1)
        background = last_frame
        
        # 第一步：定格画面，残影按顺序逐个消失（先产生的先消失）
        # 每个残影保持完全不透明的帧数
        hold_frames_per_ghost = max(2, (fade_duration_frames if fade_duration_frames is not None else ghost_interval * 2) // max(len(ghost_list), 1))
        
        active_ghost_indices = list(range(len(ghost_list)))  # 当前活跃的残影索引（按生成顺序）
        
        # 逐个移除残影：先保持完全不透明，最后几帧淡出
        fade_out_frames = max(1, hold_frames_per_ghost // 3)  # 淡出占用的帧数
        hold_frames = hold_frames_per_ghost - fade_out_frames  # 保持不透明的帧数
        
        # 按生成顺序逐个消失：先生成的先消失，最后生成的最后消失
        for ghost_idx_to_remove in range(len(ghost_list)):
            # 阶段1：当前要消失的残影保持完全不透明，显示几帧
            for _ in range(hold_frames):
                current_canvas = background.copy()
                for idx in active_ghost_indices:
                    ghost = ghost_list[idx]
                    ghost_alpha_3ch = np.repeat(ghost['alpha'][:, :, np.newaxis], 3, axis=2) / 255.0
                    current_canvas = (ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)).astype(np.uint8)
                
                frame_output = current_canvas
                out.write(frame_output)
            
            # 阶段2：要消失的残影逐渐变透明然后消失
            for step in range(fade_out_frames):
                fade_ratio = 1.0 - (step + 1) / fade_out_frames  # 从1.0到0.0
                
                current_canvas = background.copy()
                for idx in active_ghost_indices:
                    ghost = ghost_list[idx]
                    ghost_alpha = ghost['alpha'].copy()
                    
                    # 只有当前要消失的残影（按生成顺序）才变透明
                    if idx == ghost_idx_to_remove:
                        ghost_alpha = (ghost_alpha * fade_ratio).astype(np.uint8)
                    
                    ghost_alpha_3ch = np.repeat(ghost_alpha[:, :, np.newaxis], 3, axis=2) / 255.0
                    current_canvas = (ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)).astype(np.uint8)
                
                frame_output = current_canvas
                out.write(frame_output)
            
            # 移除已消失的残影（按生成顺序）
            active_ghost_indices.remove(ghost_idx_to_remove)
            
            print(f"  > 残影消失: {ghost_idx_to_remove+1}/{len(ghost_list)} (剩余 {len(active_ghost_indices)} 个)", end='\r')
        
        # 第二步：残影完全消失后，继续播放正常视频
        for i in range(effect_end_idx, self.total_frames):
            frame_output = self.read_frame(i)
            out.write(frame_output)
            print(f"  > 继续播放: {i}/{self.total_frames-1}", end='\r')

        out.release()
        print(f"\n✅ 视频及纯净 PNG 资产已输出！保存在: {output_dir}")
        print(f"   残影参数: ghost_interval={ghost_interval}, 实际添加了 {ghost_count} 个残影, 边缘处理={edge_feather}")


if __name__ == "__main__":
    start_time = time.time()

    INPUT_DIR = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204"
    OUTPUT_ROOT = "E:\\3dgs-exp\\datasets\\spacetimeSlice\\E-2026-04-22-132204\\results_slicer"
    
    START_FRAME = 25
    END_FRAME = 220
    GHOST_INTERVAL = 45
    EDGE_FEATHER = -5
    FADE_DURATION_FRAMES = 25  # 片尾定格淡出持续帧数

    print(f"\n{'='*50}")
    print(f"🚀 开始制作时空切片: [RVM] ({START_FRAME} -> {END_FRAME}), 残影间隔={GHOST_INTERVAL}, 边缘处理={EDGE_FEATHER}, 淡出帧数={FADE_DURATION_FRAMES}")
    print(f"{'='*50}")

    slicer = SpacetimeSlicer(INPUT_DIR, OUTPUT_ROOT, fps=25)
    print(f"使用设备: {slicer.device}")
    
    METHOD = 'RVM'
    if METHOD == 'RVM':
        strategy = RVMStrategy(slicer.device)
    elif METHOD == 'Hybrid':
        strategy = HybridStrategy(slicer.device)
    elif METHOD == 'SAM2_BBox':
        strategy = YOLO_SAM2_Strategy(slicer.device)
    else:
        raise ValueError(f"Unknown method: {METHOD}")
    
    slicer.generate(METHOD, START_FRAME, END_FRAME, ghost_interval=GHOST_INTERVAL, edge_feather=EDGE_FEATHER, fade_duration_frames=FADE_DURATION_FRAMES)

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")