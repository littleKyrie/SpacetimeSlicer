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


def parse_camera_ids(s):
    """解析机位序列，支持逗号分割或冒号范围"""
    if ':' in s:
        start, end = map(int, s.split(':'))
        return list(range(start, end + 1))
    elif ',' in s:
        return list(map(int, s.split(',')))
    else:
        return [int(s)]


class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25, camera_ids=None):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.camera_ids = camera_ids if camera_ids is not None else [0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 扫描所有子文件夹（帧）
        self.subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

        # 为每个机位存储帧路径
        self.frame_paths_dict = {}
        for cam_id in self.camera_ids:
            frame_paths = []
            for subdir in self.subdirs:
                frame_filename = f"{subdir} {cam_id:03d}.jpg"
                frame_path = os.path.join(input_dir, subdir, frame_filename)
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
            self.frame_paths_dict[cam_id] = frame_paths

        # 总帧数取第一个机位的帧数
        self.total_frames = len(self.frame_paths_dict[self.camera_ids[0]])
        print(f"找到 {self.total_frames} 帧")
        for cam_id in self.camera_ids:
            print(f"  机位 {cam_id}: {len(self.frame_paths_dict[cam_id])} 帧")

    def read_frame(self, idx, camera_id=None):
        if camera_id is None:
            camera_id = self.camera_ids[0]
        return cv2.imread(self.frame_paths_dict[camera_id][idx])

    def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval, edge_feather,
                        all_ghosts, permanent_indices, out, initial_canvas=None):
        """处理一段特效"""
        if initial_canvas is None:
            background = self.read_frame(0, camera_id).copy()
            canvas_ghosts = background.copy()
        else:
            canvas_ghosts = initial_canvas.copy()
            background = initial_canvas.copy()

        ghost_count = len(permanent_indices)

        for i in range(start_idx, end_idx):
            current_frame = self.read_frame(i, camera_id)

            alpha_mask = strategy.process_frame(current_frame, i)

            if edge_feather < 0:
                kernel = np.ones((3, 3), np.uint8)
                alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

            should_be_ghost = (i == start_idx) or ((i - start_idx) % ghost_interval == 0)

            alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0

            if should_be_ghost:
                canvas_ghosts = (current_frame * alpha_normalized + canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)
                ghost_count += 1
                permanent_indices.append(len(all_ghosts))

            frame_output = (current_frame * alpha_normalized + canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)

            out.write(frame_output)

            all_ghosts.append({
                'frame': current_frame.copy(),
                'alpha': alpha_mask.copy()
            })

            print(f"  > 机位 {camera_id} 渲染特效帧: {i}/{end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        return canvas_ghosts, ghost_count

    def process_freeze_transition(self, strategy, camera_ids, effect_start_idx, freeze_idx,
                                  ghost_interval, edge_feather, out):
        """处理凝结转场阶段"""
        print(f"\n🔄 处理凝结转场: 机位 {','.join(map(str, camera_ids))}")

        freeze_ghosts_list = []
        freeze_canvas_list = []

        # 先为每个机位生成凝结状态
        for cam_id in camera_ids:
            print(f"\n  预计算机位 {cam_id} 的凝结状态...")
            background = self.read_frame(0, cam_id).copy()
            canvas_ghosts = background.copy()
            cam_ghosts = []
            cam_permanent = []

            for i in range(effect_start_idx, freeze_idx + 1):
                current_frame = self.read_frame(i, cam_id)
                alpha_mask = strategy.process_frame(current_frame, i)

                if edge_feather < 0:
                    kernel = np.ones((3, 3), np.uint8)
                    alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

                should_be_ghost = (i == effect_start_idx) or ((i - effect_start_idx) % ghost_interval == 0)
                alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0

                cam_ghosts.append({
                    'frame': current_frame.copy(),
                    'alpha': alpha_mask.copy()
                })

                if should_be_ghost:
                    canvas_ghosts = (current_frame * alpha_normalized + canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)
                    cam_permanent.append(len(cam_ghosts) - 1)

                print(f"    帧 {i}/{freeze_idx}", end='\r')

            freeze_ghosts_list.append(cam_ghosts)
            freeze_canvas_list.append(canvas_ghosts)

        # 现在写入转场帧
        print(f"\n  写入转场帧...")
        for i, cam_id in enumerate(camera_ids):
            freeze_frame = self.read_frame(freeze_idx, cam_id)
            canvas_ghosts = freeze_canvas_list[i]

            out.write(canvas_ghosts)
            print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')

        return freeze_ghosts_list, freeze_canvas_list

    def process_fade_out(self, out, all_ghosts, permanent_indices, background, total_fade_frames):
        """处理片尾淡出"""
        print(f"\n🎞️ 写入片尾（定格 + 残影消失 + 继续播放）...")
        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, 总时长 {total_fade_frames} 帧")

        last_ghost_idx = len(all_ghosts) - 1

        ghost_sequences = {}
        for p_idx in permanent_indices:
            ghosts_to_traverse = list(range(p_idx, last_ghost_idx + 1))
            if len(ghosts_to_traverse) > total_fade_frames:
                indices = np.linspace(0, len(ghosts_to_traverse) - 1, total_fade_frames, dtype=int)
                sequence = [ghosts_to_traverse[i] for i in indices]
            else:
                sequence = []
                repeats = total_fade_frames // len(ghosts_to_traverse)
                remainder = total_fade_frames % len(ghosts_to_traverse)
                for i, g_idx in enumerate(ghosts_to_traverse):
                    sequence.extend([g_idx] * repeats)
                    if i < remainder:
                        sequence.append(g_idx)
                sequence = sequence[:total_fade_frames]
            ghost_sequences[p_idx] = sequence

        for frame_offset in range(total_fade_frames):
            current_canvas = background.copy()
            for p_idx in permanent_indices:
                sequence = ghost_sequences[p_idx]
                ghost_idx = sequence[frame_offset]
                ghost = all_ghosts[ghost_idx]
                ghost_alpha_3ch = np.repeat(ghost['alpha'][:, :, np.newaxis], 3, axis=2) / 255.0
                current_canvas = (ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)).astype(np.uint8)
            frame_output = current_canvas
            out.write(frame_output)
            print(f"  > 消失进度: {frame_offset+1}/{total_fade_frames}", end='\r')

    def generate(self, strategy, effect_start_idx, freeze_idx, effect_end_idx, camera_ids=None,
                 ghost_interval=1, edge_feather=0, fade_duration_frames=None):
        """
        生成时空切片视频（多机位凝结版）
        :param strategy: 分割策略对象
        :param effect_start_idx: 特效开始帧
        :param freeze_idx: 凝结帧
        :param effect_end_idx: 特效结束帧 (不包含)
        :param camera_ids: 机位序列
        :param ghost_interval: 残影间隔
        :param edge_feather: 边缘处理
        :param fade_duration_frames: 片尾淡出帧数
        """
        if camera_ids is None:
            camera_ids = self.camera_ids

        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]

        run_name = f"freeze_{start_cam}_to_{end_cam}_{effect_start_idx}_{freeze_idx}_{effect_end_idx}"
        output_dir = os.path.join(self.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, f"slicer_{run_name}.mp4")

        sample_frame = self.read_frame(0, start_cam)
        h, w = sample_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        all_ghosts = []
        permanent_indices = []

        # 1. 写入片头
        print("🎞️ 写入片头...")
        for i in range(0, effect_start_idx):
            out.write(self.read_frame(i, start_cam))

        # 2. 特效段1：起始机位，起始帧 -> 凝结帧
        print(f"\n✨ 特效段1: 机位 {start_cam} ({effect_start_idx} -> {freeze_idx})")
        canvas_ghosts, _ = self.process_segment(
            strategy, start_cam, effect_start_idx, freeze_idx + 1,
            ghost_interval, edge_feather, all_ghosts, permanent_indices, out
        )

        # 3. 凝结转场：机位序列
        freeze_ghosts_list, freeze_canvas_list = self.process_freeze_transition(
            strategy, camera_ids, effect_start_idx, freeze_idx,
            ghost_interval, edge_feather, out
        )

        # 4. 特效段2：终止机位，凝结帧+1 -> 结束帧
        # 清空残影数据，只保留终止机位的残影用于片尾消失
        all_ghosts.clear()
        permanent_indices.clear()

        print(f"\n✨ 特效段2: 机位 {end_cam} ({freeze_idx + 1} -> {effect_end_idx})")
        final_canvas, ghost_count = self.process_segment(
            strategy, end_cam, freeze_idx + 1, effect_end_idx,
            ghost_interval, edge_feather, all_ghosts, permanent_indices, out,
            initial_canvas=freeze_canvas_list[-1]
        )

        # 5. 片尾淡出
        total_fade_frames = fade_duration_frames if fade_duration_frames is not None else ghost_interval * 2
        last_frame = self.read_frame(effect_end_idx - 1, end_cam)
        self.process_fade_out(out, all_ghosts, permanent_indices, last_frame, total_fade_frames)

        # 6. 继续播放到结束
        for i in range(effect_end_idx, self.total_frames):
            frame_output = self.read_frame(i, end_cam)
            out.write(frame_output)
            print(f"  > 继续播放: {i}/{self.total_frames-1}", end='\r')

        out.release()
        print(f"\n✅ 视频已输出！保存在: {output_dir}")
        print(f"   残影参数: ghost_interval={ghost_interval}, 实际添加了 {len(permanent_indices)} 个残影, 边缘处理={edge_feather}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spacetime Slicer (Multi-Camera Freeze Version)')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_root', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--camera_ids', type=str, default='0', help='机位序列，逗号分割(0,1,2)或冒号范围(0:3)，默认:0')
    parser.add_argument('--fps', type=int, default=25, help='输出视频帧率，默认:25')
    parser.add_argument('--start_frame', type=int, default=0, help='特效起始帧，默认:0')
    parser.add_argument('--freeze_frame', type=int, required=True, help='凝结帧')
    parser.add_argument('--end_frame', type=int, default=None, help='特效结束帧，默认:最后一帧')
    parser.add_argument('--ghost_interval', type=int, default=1, help='残影间隔，默认:1')
    parser.add_argument('--edge_feather', type=int, default=0, help='边缘处理，默认:0')
    parser.add_argument('--fade_duration_frames', type=int, default=None, help='片尾淡出帧数，默认:ghost_interval*2')
    parser.add_argument('--method', type=str, default='RVM', choices=['RVM', 'Hybrid', 'SAM2_BBox', 'RMBG2'], help='分割方法，默认:RVM')
    args = parser.parse_args()

    camera_ids = parse_camera_ids(args.camera_ids)
    print(f"机位序列: {camera_ids}")

    start_time = time.time()

    slicer = SpacetimeSlicer(args.input_dir, args.output_root, fps=args.fps, camera_ids=camera_ids)
    print(f"使用设备: {slicer.device}")

    end_frame = args.end_frame if args.end_frame is not None else slicer.total_frames

    if args.method == 'RVM':
        strategy = RVMStrategy(slicer.device)
    elif args.method == 'Hybrid':
        print(">> 计算中值背景用于 Hybrid 方案...")
        start_cam = camera_ids[0]
        median_bg_frames = [cv2.cvtColor(cv2.imread(slicer.frame_paths_dict[start_cam][i]), cv2.COLOR_BGR2GRAY)
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
    print(f"🚀 开始制作时空切片: [Method: {args.method}]")
    print(f"   机位: {camera_ids}")
    print(f"   时间: 起始帧={args.start_frame} -> 凝结帧={args.freeze_frame} -> 结束帧={end_frame}")
    print(f"   参数: ghost_interval={args.ghost_interval}, edge_feather={args.edge_feather}, fade_duration={fade_duration}")
    print(f"{'='*50}")

    slicer.generate(
        strategy,
        args.start_frame,
        args.freeze_frame,
        end_frame,
        camera_ids=camera_ids,
        ghost_interval=args.ghost_interval,
        edge_feather=args.edge_feather,
        fade_duration_frames=fade_duration
    )

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")
