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
                frame_filename = f"{cam_id:03d}.jpg"
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

    def write_frame_repeat(self, out, frame, stretch=1):
        """写入帧，stretch 控制重复次数（默认 1 = 不重复）"""
        for _ in range(max(1, stretch)):
            out.write(frame)

    def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval, edge_feather,
                        all_ghosts, permanent_indices, out, initial_canvas=None,
                        ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                        stretch_ghost=1):
        """处理一段特效（支持残影透明度渐变）

        - 所有帧在循环内处理（含第一帧），输出始终显示完整人物，无亮度跳变
        - ghost 帧用"覆盖"公式烧入画布：canvas = frame·α·opacity + canvas·(1-α)
          确保 ghost #0 覆盖掉初始画布中的 100% 人物
        - 淡出背景单独生成（模糊去人），保证回收时所有残影一致
        """
        num_ghosts_expected = max(1, ((end_idx - 1 - start_idx) // ghost_interval) + 1)
        ghost_opacities = np.linspace(ghost_opacity_start, ghost_opacity_end, num_ghosts_expected)

        # 初始画布：直接用起始帧（含 100% 人物，将在 ghost #0 时被覆盖）
        if initial_canvas is not None:
            canvas_ghosts = initial_canvas.copy()
        else:
            canvas_ghosts = self.read_frame(start_idx, camera_id).copy()

        ghost_count = len(permanent_indices)
        clean_background = None  # 循环结束后用 ghost #0 的 alpha 生成

        for i in range(start_idx, end_idx):
            current_frame = self.read_frame(i, camera_id)
            alpha_mask = strategy.process_frame(current_frame, i)

            if edge_feather < 0:
                kernel = np.ones((3, 3), np.uint8)
                alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

            should_be_ghost = ((i - start_idx) % ghost_interval == 0)
            alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0

            if should_be_ghost:
                g_opacity = ghost_opacities[min(ghost_count, len(ghost_opacities) - 1)]
                # 覆盖公式：人物区 = 当前帧×opacity，背景区 = 原有画布
                canvas_ghosts = (current_frame * alpha_normalized * g_opacity +
                                 canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)
                ghost_count += 1
                permanent_indices.append(len(all_ghosts))

            # 输出帧：完整人物叠加到残影画布上（所有帧人物都 100% 显示，无闪烁）
            frame_output = (current_frame * alpha_normalized +
                            canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)

            self.write_frame_repeat(out, frame_output, stretch_ghost)

            all_ghosts.append({
                'frame': current_frame.copy(),
                'alpha': alpha_mask.copy(),
                'opacity': ghost_opacities[min(ghost_count - 1, len(ghost_opacities) - 1)] if should_be_ghost else 1.0
            })

            print(f"  > 机位 {camera_id} 渲染特效帧: {i}/{end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        # 循环后生成淡出背景：复用 ghost #0 的 alpha（避免 RVM 双重调用）
        if clean_background is None and len(all_ghosts) > 0:
            ghost0 = all_ghosts[0]
            alpha_bg = ghost0['alpha']
            alpha_bg_3ch = np.repeat(alpha_bg[:, :, np.newaxis], 3, axis=2) / 255.0
            blurred_bg = cv2.GaussianBlur(ghost0['frame'], (51, 51), 0)
            clean_background = (ghost0['frame'] * (1 - alpha_bg_3ch) +
                                blurred_bg * alpha_bg_3ch).astype(np.uint8)

        return canvas_ghosts, ghost_count, clean_background

    def process_freeze_transition(self, camera_ids, freeze_idx, out, stretch_freeze=1):
        """处理凝结转场阶段（回收残影后的多视角环绕）
        按输入机位顺序输出每个机位在 freeze_idx 的帧，产生多视角效果
        """
        print(f"\n🔄 处理凝结转场: {len(camera_ids)} 个机位 ({camera_ids[0]} -> {camera_ids[-1]})")

        for i, cam_id in enumerate(camera_ids):
            frame = self.read_frame(freeze_idx, cam_id)
            self.write_frame_repeat(out, frame, stretch_freeze)
            print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')

        print(f"\n  ✅ 凝结转场完成: 共 {len(camera_ids)} 个视角 × {stretch_freeze}")

    def process_fade_out(self, out, all_ghosts, permanent_indices, background, total_fade_frames,
                         stretch_fade=1):
        """处理片尾淡出（应用残影透明度渐变，保留原残影的透明度）"""
        print(f"\n🎞️ 写入片尾（定格 + 残影回收消失）...")
        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, 总时长 {total_fade_frames} 帧")

        if len(all_ghosts) == 0 or len(permanent_indices) == 0:
            print("  ⚠️ 没有残影需要回收，直接写入定格帧")
            for _ in range(total_fade_frames):
                self.write_frame_repeat(out, background, stretch_fade)
            return

        last_ghost_idx = len(all_ghosts) - 1

        ghost_sequences = {}
        for p_idx in permanent_indices:
            ghosts_to_traverse = list(range(p_idx, last_ghost_idx + 1))
            if len(ghosts_to_traverse) == 0:
                continue
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

        if len(ghost_sequences) == 0:
            print("  ⚠️ 无法构建残影序列，直接写入定格帧")
            for _ in range(total_fade_frames):
                self.write_frame_repeat(out, background, stretch_fade)
            return

        for frame_offset in range(total_fade_frames):
            current_canvas = background.copy()
            for p_idx in permanent_indices:
                sequence = ghost_sequences.get(p_idx)
                if sequence is None:
                    continue
                ghost_idx = sequence[frame_offset]
                ghost = all_ghosts[ghost_idx]
                # 保留原残影的透明度
                ghost_opacity = ghost.get('opacity', 1.0)
                ghost_alpha_3ch = np.repeat(ghost['alpha'][:, :, np.newaxis], 3, axis=2) / 255.0 * ghost_opacity
                current_canvas = (ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)).astype(np.uint8)
            self.write_frame_repeat(out, current_canvas, stretch_fade)
            print(f"  > 消失进度: {frame_offset+1}/{total_fade_frames}", end='\r')
        print()  # 换行

    def generate(self, strategy, effect_start_idx, freeze_idx, effect_end_idx, camera_ids=None,
                 ghost_interval=1, edge_feather=0, fade_duration_frames=None,
                 ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                 stretch_head=1, stretch_ghost=1, stretch_fade=1,
                 stretch_freeze=1, stretch_tail=1):
        """
        生成时空切片视频（残影渐变 → 回收 → 多视角凝结 → 继续播放）

        流程:
          1. 片头: 0 -> effect_start_idx（固定机位原样播放）
          2. 特效段: effect_start_idx -> freeze_idx（残影+透明度渐变）
             - 第一帧人物 = ghost #0, 直接应用 opacity 作为初始画布
             - 后续残影线性叠加
          3. 片尾淡出: 回收所有残影
          4. 凝结转场: 按 camera_ids 顺序输出各机位 freeze_idx 帧
          5. 继续播放: freeze_idx+1 -> effect_end_idx（终止机位原样播放）
        """
        if camera_ids is None:
            camera_ids = self.camera_ids

        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]

        stretch_suffix = f"_sh{stretch_head}_sg{stretch_ghost}_sfd{stretch_fade}_sfz{stretch_freeze}_st{stretch_tail}"
        run_name = f"freeze_{start_cam}_to_{end_cam}_seq{len(camera_ids)}_s{effect_start_idx}_f{freeze_idx}_e{effect_end_idx}{stretch_suffix}"
        output_dir = os.path.join(self.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, f"slicer_{run_name}.mp4")

        sample_frame = self.read_frame(0, start_cam)
        h, w = sample_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        all_ghosts = []
        permanent_indices = []

        total_fade_frames = fade_duration_frames if fade_duration_frames is not None else ghost_interval * 2

        # ============ 1. 写入片头 ============
        print("🎞️ 写入片头...")
        for i in range(0, effect_start_idx):
            self.write_frame_repeat(out, self.read_frame(i, start_cam), stretch_head)
        if effect_start_idx > 0:
            print(f"  片头完成: 0 -> {effect_start_idx-1} ({effect_start_idx} 帧, ×{stretch_head})")

        # ============ 2. 特效段: 固定机位 + 残影 + 透明度渐变 ============
        print(f"\n✨ 特效段: 机位 {start_cam} ({effect_start_idx} -> {freeze_idx})")
        print(f"   残影透明度: {ghost_opacity_start:.0%} -> {ghost_opacity_end:.0%}, 插帧 ×{stretch_ghost}")
        canvas_ghosts, ghost_count, clean_background = self.process_segment(
            strategy, start_cam, effect_start_idx, freeze_idx + 1,
            ghost_interval, edge_feather, all_ghosts, permanent_indices, out,
            ghost_opacity_start=ghost_opacity_start,
            ghost_opacity_end=ghost_opacity_end,
            stretch_ghost=stretch_ghost
        )

        # ============ 3. 片尾淡出: 回收所有残影 ============
        # 使用特效段第一帧的去人物背景，保证回收时透明度与生成一致
        fade_background = clean_background if clean_background is not None else self.read_frame(freeze_idx, start_cam)
        self.process_fade_out(out, all_ghosts, permanent_indices, fade_background, total_fade_frames,
                              stretch_fade=stretch_fade)

        # ============ 4. 凝结转场: 多机位环绕 ============
        print(f"\n💫 进入凝结状态，多机位环绕...")
        self.process_freeze_transition(camera_ids, freeze_idx, out, stretch_freeze=stretch_freeze)

        # ============ 5. 继续播放: 凝结帧之后 -> 结束帧 ============
        if freeze_idx + 1 < effect_end_idx:
            print(f"\n▶️  继续播放: 机位 {end_cam} ({freeze_idx + 1} -> {effect_end_idx - 1}), 插帧 ×{stretch_tail}")
            for i in range(freeze_idx + 1, effect_end_idx):
                self.write_frame_repeat(out, self.read_frame(i, end_cam), stretch_tail)
                print(f"  > 播放帧 {i}/{effect_end_idx - 1}", end='\r')
            print()

        # ============ 6. 收尾 ============
        out.release()
        print(f"\n✅ 视频已输出！保存在: {output_dir}")
        print(f"   残影: ghost_interval={ghost_interval}, 共 {ghost_count} 个, "
              f"透明度 {ghost_opacity_start:.0%}->{ghost_opacity_end:.0%}")
        print(f"   片尾淡出: {total_fade_frames} 帧 ×{stretch_fade}")
        print(f"   凝结转场: {len(camera_ids)} 个机位视角 ×{stretch_freeze}")
        if freeze_idx + 1 < effect_end_idx:
            print(f"   继续播放: {freeze_idx + 1} -> {effect_end_idx - 1} ({effect_end_idx - freeze_idx - 1} 帧, 机位 {end_cam}, ×{stretch_tail})")


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
    parser.add_argument('--ghost_opacity_start', type=float, default=0.2, help='最早残影透明度 (0~1, 越小越透明), 默认:0.2')
    parser.add_argument('--stretch_head', type=int, default=1, help='片头每帧重复次数, 默认:1')
    parser.add_argument('--stretch_ghost', type=int, default=1, help='特效段每帧重复次数, 默认:1')
    parser.add_argument('--stretch_fade', type=int, default=1, help='淡出段每帧重复次数, 默认:1')
    parser.add_argument('--stretch_freeze', type=int, default=1, help='凝结转场每帧重复次数, 默认:1')
    parser.add_argument('--stretch_tail', type=int, default=1, help='尾帧每帧重复次数, 默认:1')
    parser.add_argument('--method', type=str, default='RVM', choices=['RVM', 'Hybrid', 'SAM2_BBox', 'RMBG2'], help='分割方法，默认:RVM')
    args = parser.parse_args()

    camera_ids = parse_camera_ids(args.camera_ids)
    print(f"机位序列: {camera_ids}")

    start_time = time.time()

    slicer = SpacetimeSlicer(args.input_dir, args.output_root, fps=args.fps, camera_ids=camera_ids)
    # slicer.device = "cpu"
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
    print(f"   参数: ghost_interval={args.ghost_interval}, edge_feather={args.edge_feather}, fade_duration={fade_duration}, opacity_start={args.ghost_opacity_start}")
    print(f"   插帧: head={args.stretch_head}, ghost={args.stretch_ghost}, fade={args.stretch_fade}, freeze={args.stretch_freeze}, tail={args.stretch_tail}")
    print(f"{'='*50}")

    slicer.generate(
        strategy,
        args.start_frame,
        args.freeze_frame,
        end_frame,
        camera_ids=camera_ids,
        ghost_interval=args.ghost_interval,
        edge_feather=args.edge_feather,
        fade_duration_frames=fade_duration,
        ghost_opacity_start=args.ghost_opacity_start,
        stretch_head=args.stretch_head,
        stretch_ghost=args.stretch_ghost,
        stretch_fade=args.stretch_fade,
        stretch_freeze=args.stretch_freeze,
        stretch_tail=args.stretch_tail
    )

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")
