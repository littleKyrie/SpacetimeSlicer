import cv2
import os
import time
import numpy as np
import torch
import argparse
from models.rvm import RVMStrategy
from models.hybrid_rvm import HybridStrategy
from models.yolo_sam2 import YOLO_SAM2_Strategy
from models.rmbg2 import RMBG2Strategy
from models.rembg import RembgStrategy

class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25, camera_id=0):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.camera_id = camera_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        self.frame_paths = []
        for subdir in subdirs:
            frame_filename = f"{subdir} {camera_id:03d}.jpg"
            frame_path = os.path.join(input_dir, subdir, frame_filename)
            if os.path.exists(frame_path):
                self.frame_paths.append(frame_path)
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
        all_ghosts = []      # 所有残影（永久+临时）
        permanent_indices = []  # 永久残影的索引

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

            # 存储所有残影
            all_ghosts.append({
                'frame': current_frame.copy(),
                'alpha': alpha_mask.copy()
            })

            if should_be_ghost:
                canvas_ghosts = (current_frame * alpha_normalized + canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)
                ghost_count += 1
                permanent_indices.append(len(all_ghosts) - 1)  # 记录当前永久残影的索引

            print(f"  > 渲染特效帧: {i}/{effect_end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        # 5. 写入片尾（所有永久残影同时开始、同时结束消失）
        print("\n🎞️ 写入片尾（定格 + 残影消失 + 继续播放）...")

        # 获取最后一帧画面（用于定格）
        last_frame = self.read_frame(effect_end_idx - 1)
        background = last_frame

        # 计算每个永久残影的消失参数
        total_fade_frames = fade_duration_frames if fade_duration_frames is not None else ghost_interval * 2

        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, 总时长 {total_fade_frames} 帧")

        # 生成每帧的消失状态
        # 对于每个永久残影，它需要"穿越"从自己到最后一个残影之间的所有残影
        last_ghost_idx = len(all_ghosts) - 1

        # 5. 写入片尾（所有永久残影同时开启消失旅程，同时结束）
        print("\n🎞️ 写入片尾（定格 + 残影消失 + 继续播放）...")

        # 获取最后一帧画面（用于定格）
        last_frame = self.read_frame(effect_end_idx - 1)
        background = last_frame

        # 消失总时长
        total_fade_frames = fade_duration_frames if fade_duration_frames is not None else ghost_interval * 2

        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, 总时长 {total_fade_frames} 帧")

        # 计算每个永久残影的播放序列
        last_ghost_idx = len(all_ghosts) - 1

        ghost_sequences = {}  # permanent_idx -> list of ghost indices to display

        for p_idx in permanent_indices:
            # 计算这个永久残影需要穿越的残影范围
            ghosts_to_traverse = list(range(p_idx, last_ghost_idx + 1))

            # 采样：如果数量超过 total_fade_frames，均匀采样
            if len(ghosts_to_traverse) > total_fade_frames:
                # 均匀采样
                indices = np.linspace(0, len(ghosts_to_traverse) - 1, total_fade_frames, dtype=int)
                sequence = [ghosts_to_traverse[i] for i in indices]
            else:
                # 直接使用全部，按顺序重复以填充 total_fade_frames
                sequence = []
                repeats = total_fade_frames // len(ghosts_to_traverse)
                remainder = total_fade_frames % len(ghosts_to_traverse)
                for i, g_idx in enumerate(ghosts_to_traverse):
                    sequence.extend([g_idx] * repeats)
                    if i < remainder:
                        sequence.append(g_idx)
                sequence = sequence[:total_fade_frames]

            ghost_sequences[p_idx] = sequence

        # 生成每帧的消失状态
        for frame_offset in range(total_fade_frames):
            # 当前帧叠放的所有残影（先生成在下，后生成在上）
            current_canvas = background.copy()

            # 遍历所有永久残影，按生成顺序叠加
            for p_idx in permanent_indices:
                sequence = ghost_sequences[p_idx]
                ghost_idx = sequence[frame_offset]

                ghost = all_ghosts[ghost_idx]
                ghost_alpha_3ch = np.repeat(ghost['alpha'][:, :, np.newaxis], 3, axis=2) / 255.0
                current_canvas = (ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)).astype(np.uint8)

            frame_output = current_canvas
            out.write(frame_output)

            print(f"  > 消失进度: {frame_offset+1}/{total_fade_frames}", end='\r')

            frame_output = current_canvas
            out.write(frame_output)

            print(f"  > 消失进度: {frame_offset+1}/{total_fade_frames}", end='\r')

        # 第二步：残影完全消失后，继续播放正常视频
        for i in range(effect_end_idx, self.total_frames):
            frame_output = self.read_frame(i)
            out.write(frame_output)
            print(f"  > 继续播放: {i}/{self.total_frames-1}", end='\r')

        out.release()
        print(f"\n✅ 视频及纯净 PNG 资产已输出！保存在: {output_dir}")
        print(f"   残影参数: ghost_interval={ghost_interval}, 实际添加了 {ghost_count} 个残影, 边缘处理={edge_feather}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spacetime Slicer')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_root', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--camera_id', type=int, default=0, help='机位号 (默认: 0)')
    parser.add_argument('--fps', type=int, default=25, help='输出视频帧率 (默认: 25)')
    parser.add_argument('--start_frame', type=int, default=0, help='起始帧 (默认: 0)')
    parser.add_argument('--end_frame', type=int, default=None, help='结束帧 (默认: 最后一帧)')
    parser.add_argument('--ghost_interval', type=int, default=1, help='残影间隔 (默认: 1)')
    parser.add_argument('--edge_feather', type=int, default=0, help='边缘处理 (默认: 0)')
    parser.add_argument('--fade_duration_frames', type=int, default=None, help='片尾残影淡出持续帧数 (默认: ghost_interval * 2)')
    parser.add_argument('--method', type=str, default='RVM', choices=['RVM', 'Hybrid', 'SAM2_BBox', 'RMBG2'], help='分割方法 (默认: RVM)')
    args = parser.parse_args()

    start_time = time.time()

    slicer = SpacetimeSlicer(args.input_dir, args.output_root, fps=args.fps, camera_id=args.camera_id)
    print(f"使用设备: {slicer.device}")
    print(f"找到 {slicer.total_frames} 帧 (机位 {args.camera_id})")

    end_frame = args.end_frame if args.end_frame is not None else slicer.total_frames

    if args.method == 'RVM':
        strategy = RVMStrategy(slicer.device)
    elif args.method == 'Hybrid':
        print(">> 计算中值背景用于 Hybrid 方案...")
        median_bg_frames = [cv2.cvtColor(cv2.imread(slicer.frame_paths[i]), cv2.COLOR_BGR2GRAY)
                          for i in range(0, slicer.total_frames, 5)]
        median_bg = np.median(median_bg_frames, axis=0).astype(np.uint8)
        strategy = HybridStrategy(slicer.device, median_bg)
    elif args.method == 'SAM2_BBox':
        strategy = YOLO_SAM2_Strategy(slicer.device)
    elif args.method.startswith('rembg-'):
        model_name = args.method.split('-', 1)[1]
        strategy = RembgStrategy(slicer.device, model_name)
    elif args.method == 'RMBG2':
        strategy = RMBG2Strategy(slicer.device)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    fade_duration = args.fade_duration_frames if args.fade_duration_frames is not None else args.ghost_interval * 2

    print(f"\n{'='*50}")
    print(f"🚀 开始制作时空切片: [Method: {args.method}] ({args.start_frame} -> {end_frame}), 残影间隔={args.ghost_interval}, 边缘处理={args.edge_feather}, 淡出帧数={fade_duration}")
    print(f"{'='*50}")

    slicer.generate(args.method, args.start_frame, end_frame, ghost_interval=args.ghost_interval, edge_feather=args.edge_feather, fade_duration_frames=fade_duration)

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")