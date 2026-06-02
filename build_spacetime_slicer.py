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


class StretchedFrameWriter:
    """Write a stage with optional interpolation between adjacent frames."""

    def __init__(self, slicer, out, stretch=1, mode='repeat'):
        self.slicer = slicer
        self.out = out
        self.stretch = max(1, stretch)
        self.mode = mode
        self.pending_frame = None

    def write(self, frame):
        if self.mode == 'repeat' or self.stretch == 1:
            self.slicer.write_frame_repeat(self.out, frame, self.stretch)
            return

        if self.pending_frame is not None:
            self.slicer.write_frame_transition(
                self.out, self.pending_frame, frame, self.stretch, self.mode
            )
        self.pending_frame = frame.copy()

    def finish(self):
        if self.pending_frame is not None:
            self.slicer.write_frame_repeat(self.out, self.pending_frame, self.stretch)
            self.pending_frame = None


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
    def __init__(self, input_dir, output_root, fps=25, camera_ids=None, flow_scale=0.5):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.camera_ids = camera_ids if camera_ids is not None else [0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flow_scale = flow_scale

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

    def prepare_interpolation(self, first_frame, second_frame, mode):
        if mode != 'flow':
            return None

        scale = min(1.0, max(0.05, getattr(self, 'flow_scale', 0.5)))
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            first_gray = cv2.resize(first_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            second_gray = cv2.resize(second_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        forward_flow = cv2.calcOpticalFlowFarneback(
            first_gray, second_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        backward_flow = cv2.calcOpticalFlowFarneback(
            second_gray, first_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = first_frame.shape[:2]
        if scale < 1.0:
            forward_flow = cv2.resize(forward_flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
            backward_flow = cv2.resize(backward_flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        return forward_flow, backward_flow, grid_x, grid_y

    def interpolate_frame(self, first_frame, second_frame, ratio, mode, prepared=None):
        if mode == 'blend':
            return cv2.addWeighted(first_frame, 1.0 - ratio, second_frame, ratio, 0)
        if mode != 'flow':
            raise ValueError(f"Unknown stretch mode: {mode}")

        forward_flow, backward_flow, grid_x, grid_y = prepared or self.prepare_interpolation(
            first_frame, second_frame, mode
        )
        first_warped = cv2.remap(
            first_frame,
            grid_x + backward_flow[:, :, 0] * ratio,
            grid_y + backward_flow[:, :, 1] * ratio,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        second_warped = cv2.remap(
            second_frame,
            grid_x + forward_flow[:, :, 0] * (1.0 - ratio),
            grid_y + forward_flow[:, :, 1] * (1.0 - ratio),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return cv2.addWeighted(first_warped, 1.0 - ratio, second_warped, ratio, 0)

    def write_frame_transition(self, out, first_frame, second_frame, stretch, mode):
        prepared = self.prepare_interpolation(first_frame, second_frame, mode)
        for step in range(max(1, stretch)):
            ratio = step / stretch
            frame = first_frame if step == 0 else self.interpolate_frame(
                first_frame, second_frame, ratio, mode, prepared
            )
            out.write(frame)

    def build_temporal_median_background(self, camera_id, max_samples=31):
        """Build a clean plate for a static camera without keeping one source-frame subject."""
        frame_count = len(self.frame_paths_dict[camera_id])
        if frame_count == 0:
            raise ValueError(f"Camera {camera_id} has no frames")
        sample_count = min(max_samples, frame_count)
        sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        sampled_frames = [self.read_frame(i, camera_id) for i in sample_indices]
        return np.median(np.stack(sampled_frames, axis=0), axis=0).astype(np.uint8)

    def resolve_fade_duration(self, effect_frame_count, fade_duration_frames):
        if fade_duration_frames is not None:
            return fade_duration_frames
        return max(effect_frame_count, max(2, self.fps // 2))

    def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval, edge_feather,
                        all_ghosts, permanent_indices, out, initial_canvas=None,
                        initial_subject_replacement=None,
                        ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                        stretch_ghost=1, stretch_mode='repeat'):
        """Accumulate translucent ghosts while keeping the current subject fully visible."""
        num_ghosts_expected = max(1, ((end_idx - 1 - start_idx) // ghost_interval) + 1)
        ghost_opacities = np.linspace(ghost_opacity_start, ghost_opacity_end, num_ghosts_expected)

        # A clean initial canvas prevents the first source-frame subject from surviving recovery.
        if initial_canvas is not None:
            canvas_ghosts = initial_canvas.copy()
        else:
            canvas_ghosts = self.read_frame(start_idx, camera_id).copy()

        ghost_count = len(permanent_indices)
        stage_writer = StretchedFrameWriter(self, out, stretch_ghost, stretch_mode)
        last_frame_output = canvas_ghosts.copy()

        for i in range(start_idx, end_idx):
            current_frame = self.read_frame(i, camera_id)
            alpha_mask = strategy.process_frame(current_frame, i)

            if edge_feather < 0:
                kernel = np.ones((3, 3), np.uint8)
                alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

            should_be_ghost = ((i - start_idx) % ghost_interval == 0)
            alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0

            if i == start_idx and initial_subject_replacement is not None:
                canvas_ghosts = (
                    canvas_ghosts * (1 - alpha_normalized) +
                    initial_subject_replacement * alpha_normalized
                ).astype(np.uint8)

            if should_be_ghost:
                g_opacity = ghost_opacities[min(ghost_count, len(ghost_opacities) - 1)]
                # Apply opacity to the complete alpha-over operation.
                ghost_alpha = alpha_normalized * g_opacity
                canvas_ghosts = (current_frame * ghost_alpha +
                                 canvas_ghosts * (1 - ghost_alpha)).astype(np.uint8)
                ghost_count += 1
                permanent_indices.append(len(all_ghosts))

            # 输出帧：完整人物叠加到残影画布上（所有帧人物都 100% 显示，无闪烁）
            frame_output = (current_frame * alpha_normalized +
                            canvas_ghosts * (1 - alpha_normalized)).astype(np.uint8)
            last_frame_output = frame_output

            stage_writer.write(frame_output)

            all_ghosts.append({
                'frame': current_frame.copy(),
                'alpha': alpha_mask.copy(),
                'opacity': ghost_opacities[min(ghost_count - 1, len(ghost_opacities) - 1)] if should_be_ghost else 1.0
            })

            print(f"  > 机位 {camera_id} 渲染特效帧: {i}/{end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        stage_writer.finish()

        return canvas_ghosts, ghost_count, last_frame_output

    def process_freeze_transition(self, camera_ids, freeze_idx, out, stretch_freeze=1,
                                  stretch_mode='repeat'):
        """处理凝结转场阶段（回收残影后的多视角环绕）
        按输入机位顺序输出每个机位在 freeze_idx 的帧，产生多视角效果
        """
        print(f"\n🔄 处理凝结转场: {len(camera_ids)} 个机位 ({camera_ids[0]} -> {camera_ids[-1]})")

        stage_writer = StretchedFrameWriter(self, out, stretch_freeze, stretch_mode)
        for i, cam_id in enumerate(camera_ids):
            frame = self.read_frame(freeze_idx, cam_id)
            stage_writer.write(frame)
            print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        stage_writer.finish()

        print(f"\n  ✅ 凝结转场完成: 共 {len(camera_ids)} 个视角 × {stretch_freeze}")

    def interpolate_ghost(self, all_ghosts, position):
        lower_idx = int(np.floor(position))
        upper_idx = min(lower_idx + 1, len(all_ghosts) - 1)
        ratio = position - lower_idx
        if upper_idx == lower_idx or ratio <= 0:
            return all_ghosts[lower_idx]

        lower_ghost = all_ghosts[lower_idx]
        upper_ghost = all_ghosts[upper_idx]
        return {
            'frame': cv2.addWeighted(lower_ghost['frame'], 1.0 - ratio,
                                     upper_ghost['frame'], ratio, 0),
            'alpha': cv2.addWeighted(lower_ghost['alpha'], 1.0 - ratio,
                                     upper_ghost['alpha'], ratio, 0),
        }

    def build_recovery_trajectories(self, permanent_indices, last_ghost_idx, total_fade_frames):
        trajectories = {}
        for p_idx in permanent_indices:
            if total_fade_frames == 1:
                trajectories[p_idx] = [float(last_ghost_idx)]
                continue

            positions = []
            for frame_offset in range(total_fade_frames):
                progress = frame_offset / (total_fade_frames - 1)
                eased_progress = progress * progress * (3.0 - 2.0 * progress)
                positions.append(p_idx + (last_ghost_idx - p_idx) * eased_progress)
            trajectories[p_idx] = positions
        return trajectories

    def compose_recovery_frame(self, all_ghosts, permanent_indices, ghost_trajectories, background,
                               frame_offset):
        current_canvas = background.copy()
        for p_idx in permanent_indices:
            trajectory = ghost_trajectories.get(p_idx)
            if trajectory is None:
                continue
            ghost = self.interpolate_ghost(all_ghosts, trajectory[frame_offset])
            ghost_opacity = all_ghosts[p_idx].get('opacity', 1.0)
            ghost_alpha_3ch = np.repeat(
                ghost['alpha'][:, :, np.newaxis], 3, axis=2
            ) / 255.0 * ghost_opacity
            current_canvas = (
                ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)
            ).astype(np.uint8)
        return current_canvas

    def write_canvas_transition(self, out, first_frame, second_frame, transition_frames):
        for step in range(1, transition_frames + 1):
            ratio = step / (transition_frames + 1)
            out.write(cv2.addWeighted(first_frame, 1.0 - ratio, second_frame, ratio, 0))

    def process_fade_out(self, out, all_ghosts, permanent_indices, background, total_fade_frames,
                         stretch_fade=1, stretch_mode='repeat', transition_from=None,
                         recovery_transition_frames=0):
        """处理片尾淡出（应用残影透明度渐变，保留原残影的透明度）"""
        output_fade_frames = total_fade_frames * max(1, stretch_fade)
        print(f"\n🎞️ 写入片尾（定格 + 残影回收消失）...")
        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, "
              f"{total_fade_frames} 逻辑帧 × {stretch_fade} = {output_fade_frames} 输出帧")

        if len(all_ghosts) == 0 or len(permanent_indices) == 0:
            print("  ⚠️ 没有残影需要回收，直接写入定格帧")
            if transition_from is not None:
                self.write_canvas_transition(out, transition_from, background, recovery_transition_frames)
            for _ in range(output_fade_frames):
                out.write(background)
            return

        last_ghost_idx = len(all_ghosts) - 1

        ghost_trajectories = self.build_recovery_trajectories(
            permanent_indices, last_ghost_idx, output_fade_frames
        )

        if len(ghost_trajectories) == 0:
            print("  ⚠️ 无法构建残影序列，直接写入定格帧")
            if transition_from is not None:
                self.write_canvas_transition(out, transition_from, background, recovery_transition_frames)
            for _ in range(output_fade_frames):
                out.write(background)
            return

        first_recovery_frame = self.compose_recovery_frame(
            all_ghosts, permanent_indices, ghost_trajectories, background, 0
        )
        if transition_from is not None:
            self.write_canvas_transition(
                out, transition_from, first_recovery_frame, recovery_transition_frames
            )

        for frame_offset in range(output_fade_frames):
            current_canvas = self.compose_recovery_frame(
                all_ghosts, permanent_indices, ghost_trajectories, background, frame_offset
            )
            out.write(current_canvas)
            print(f"  > 消失进度: {frame_offset+1}/{output_fade_frames}", end='\r')
        print()  # 换行

    def generate(self, strategy, effect_start_idx, freeze_idx, effect_end_idx, camera_ids=None,
                 ghost_interval=1, edge_feather=0, fade_duration_frames=None,
                 ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                 stretch_head=1, stretch_ghost=1, stretch_fade=1,
                 stretch_freeze=1, stretch_tail=1, stretch_mode='repeat',
                 background_mode='freeze', recovery_transition_frames=3,
                 initial_subject_patch_mode='median'):
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

        if not camera_ids:
            raise ValueError("camera_ids must not be empty")
        if ghost_interval < 1:
            raise ValueError("ghost_interval must be at least 1")
        if not 0.0 <= ghost_opacity_start <= 1.0 or not 0.0 <= ghost_opacity_end <= 1.0:
            raise ValueError("ghost opacity must be between 0 and 1")
        if not effect_start_idx <= freeze_idx < effect_end_idx:
            raise ValueError("Expected effect_start_idx <= freeze_idx < effect_end_idx")
        if fade_duration_frames is not None and fade_duration_frames < 1:
            raise ValueError("fade_duration_frames must be at least 1")
        if recovery_transition_frames < 0:
            raise ValueError("recovery_transition_frames must not be negative")
        if min(stretch_head, stretch_ghost, stretch_fade, stretch_freeze, stretch_tail) < 1:
            raise ValueError("stretch values must be at least 1")

        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]
        if freeze_idx >= len(self.frame_paths_dict[start_cam]):
            raise ValueError(f"Camera {start_cam} does not contain freeze frame {freeze_idx}")
        for camera_id in camera_ids:
            if freeze_idx >= len(self.frame_paths_dict[camera_id]):
                raise ValueError(f"Camera {camera_id} does not contain freeze frame {freeze_idx}")
        if effect_end_idx > len(self.frame_paths_dict[end_cam]):
            raise ValueError(f"Camera {end_cam} does not contain frames up to {effect_end_idx - 1}")

        stretch_suffix = f"_sh{stretch_head}_sg{stretch_ghost}_sfd{stretch_fade}_sfz{stretch_freeze}_st{stretch_tail}_{stretch_mode}_patch{initial_subject_patch_mode}_recoverybg{background_mode}_rt{recovery_transition_frames}"
        run_name = f"freeze_{start_cam}_to_{end_cam}_seq{len(camera_ids)}_s{effect_start_idx}_f{freeze_idx}_e{effect_end_idx}{stretch_suffix}"
        output_dir = os.path.join(self.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, "slicer.mp4")

        sample_frame = self.read_frame(0, start_cam)
        h, w = sample_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        if not out.isOpened():
            out.release()
            raise RuntimeError(
                f"Failed to open video writer: {video_path} "
                f"(path length: {len(os.path.abspath(video_path))})"
            )

        all_ghosts = []
        permanent_indices = []

        effect_frame_count = freeze_idx - effect_start_idx + 1
        total_fade_frames = self.resolve_fade_duration(effect_frame_count, fade_duration_frames)

        # ============ 1. 写入片头 ============
        print("🎞️ 写入片头...")
        stage_writer = StretchedFrameWriter(self, out, stretch_head, stretch_mode)
        for i in range(0, effect_start_idx):
            stage_writer.write(self.read_frame(i, start_cam))
        stage_writer.finish()
        if effect_start_idx > 0:
            print(f"  片头完成: 0 -> {effect_start_idx-1} ({effect_start_idx} 帧, ×{stretch_head})")

        # ============ 2. 特效段: 固定机位 + 残影 + 透明度渐变 ============
        print(f"\n✨ 特效段: 机位 {start_cam} ({effect_start_idx} -> {freeze_idx})")
        print(f"   残影透明度: {ghost_opacity_start:.0%} -> {ghost_opacity_end:.0%}, 插帧 ×{stretch_ghost}")
        generation_canvas = self.read_frame(effect_start_idx, start_cam).copy()
        if initial_subject_patch_mode == 'median':
            initial_subject_replacement = self.build_temporal_median_background(start_cam)
        elif initial_subject_patch_mode == 'freeze':
            initial_subject_replacement = self.read_frame(freeze_idx, start_cam).copy()
        else:
            raise ValueError(f"Unknown initial subject patch mode: {initial_subject_patch_mode}")
        canvas_ghosts, ghost_count, last_effect_frame = self.process_segment(
            strategy, start_cam, effect_start_idx, freeze_idx + 1,
            ghost_interval, edge_feather, all_ghosts, permanent_indices, out,
            initial_canvas=generation_canvas,
            initial_subject_replacement=initial_subject_replacement,
            ghost_opacity_start=ghost_opacity_start,
            ghost_opacity_end=ghost_opacity_end,
            stretch_ghost=stretch_ghost,
            stretch_mode=stretch_mode
        )

        # ============ 3. 片尾淡出: 回收所有残影 ============
        if background_mode == 'median':
            fade_background = self.build_temporal_median_background(start_cam)
        elif background_mode == 'freeze':
            fade_background = self.read_frame(freeze_idx, start_cam).copy()
        elif background_mode == 'start':
            fade_background = generation_canvas
        else:
            raise ValueError(f"Unknown background mode: {background_mode}")
        self.process_fade_out(out, all_ghosts, permanent_indices, fade_background, total_fade_frames,
                              stretch_fade=stretch_fade, stretch_mode=stretch_mode,
                              transition_from=last_effect_frame,
                              recovery_transition_frames=recovery_transition_frames)

        # ============ 4. 凝结转场: 多机位环绕 ============
        print(f"\n💫 进入凝结状态，多机位环绕...")
        self.process_freeze_transition(camera_ids, freeze_idx, out, stretch_freeze=stretch_freeze,
                                       stretch_mode=stretch_mode)

        # ============ 5. 继续播放: 凝结帧之后 -> 结束帧 ============
        if freeze_idx + 1 < effect_end_idx:
            print(f"\n▶️  继续播放: 机位 {end_cam} ({freeze_idx + 1} -> {effect_end_idx - 1}), 插帧 ×{stretch_tail}")
            stage_writer = StretchedFrameWriter(self, out, stretch_tail, stretch_mode)
            for i in range(freeze_idx + 1, effect_end_idx):
                stage_writer.write(self.read_frame(i, end_cam))
                print(f"  > 播放帧 {i}/{effect_end_idx - 1}", end='\r')
            stage_writer.finish()
            print()

        # ============ 6. 收尾 ============
        out.release()
        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"Video writer did not create a valid output file: {video_path}")
        print(f"\n✅ 视频已输出！保存在: {video_path}")
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
    parser.add_argument('--camera_ids', type=str, default='0', help='机位序列，逗号分割(0,1,2) 或 冒号范围(0:3)，默认:0')
    parser.add_argument('--fps', type=int, default=25, help='输出视频帧率，默认:25')
    parser.add_argument('--start_frame', type=int, default=0, help='特效起始帧，默认:0')
    parser.add_argument('--freeze_frame', type=int, required=True, help='凝结帧，该帧回收特效切片并进入多机位环绕')
    parser.add_argument('--end_frame', type=int, default=None, help='视频结束帧，默认:最后一帧，即与原视频保持一致不提前退出')
    parser.add_argument('--ghost_interval', type=int, default=1, help='残影间隔，默认:1')
    parser.add_argument('--edge_feather', type=int, default=0, help='边缘处理，默认:0')
    parser.add_argument('--fade_duration_frames', type=int, default=None,
                        help='切片回收逻辑帧数，默认:max(特效段帧数, fps//2)')
    parser.add_argument('--ghost_opacity_start', type=float, default=0.2, help='最早残影透明度, 默认:0.2')
    parser.add_argument('--ghost_opacity_end', type=float, default=1.0, help='最晚残影透明度, 默认:1.0')
    parser.add_argument('--stretch_head', type=int, default=1, help='片头每帧重复次数, 默认:1')
    parser.add_argument('--stretch_ghost', type=int, default=1, help='生成切片段每帧重复次数, 默认:1')
    parser.add_argument('--stretch_fade', type=int, default=1, help='切片回收段每帧重复次数, 默认:1')
    parser.add_argument('--stretch_freeze', type=int, default=1, help='凝结转场每帧重复次数, 默认:1')
    parser.add_argument('--stretch_tail', type=int, default=1, help='片尾每帧重复次数, 默认:1')
    parser.add_argument('--stretch_mode', type=str, default='repeat', choices=['repeat', 'blend', 'flow'],
                        help='延时模式: repeat=重复帧, blend=线性混合, flow=光流插帧')
    parser.add_argument('--background_mode', type=str, default='freeze', choices=['median', 'freeze', 'start'],
                        help='回收画布背景: median=时间中位数干净背景, freeze=凝结帧, start=起始帧')
    parser.add_argument('--initial_subject_patch_mode', type=str, default='median', choices=['median', 'freeze'],
                        help='起始人物区域补洞来源: median=时间中位数干净背景, freeze=凝结帧')
    parser.add_argument('--recovery_transition_frames', type=int, default=3,
                        help='生成画布切换到回收画布时的交叉过渡帧数, 默认:3')
    parser.add_argument('--flow_scale', type=float, default=0.5,
                        help='光流计算缩放比例 (0.05~1.0), 越小越快, 默认:0.5')
    parser.add_argument('--method', type=str, default='RVM', choices=['RVM', 'Hybrid', 'SAM2_BBox', 'RMBG2'], help='分割方法，默认:RVM')
    args = parser.parse_args()

    camera_ids = parse_camera_ids(args.camera_ids)
    print(f"机位序列: {camera_ids}")

    start_time = time.time()

    slicer = SpacetimeSlicer(args.input_dir, args.output_root, fps=args.fps, camera_ids=camera_ids,
                             flow_scale=args.flow_scale)
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

    fade_duration = args.fade_duration_frames if args.fade_duration_frames is not None else 'auto'

    print(f"\n{'='*50}")
    print(f"🚀 开始制作时空切片: [Method: {args.method}]")
    print(f"   机位: {camera_ids}")
    print(f"   时间: 起始帧={args.start_frame} -> 凝结帧={args.freeze_frame} -> 结束帧={end_frame}")
    print(f"   参数: ghost_interval={args.ghost_interval}, edge_feather={args.edge_feather}, fade_duration={fade_duration}, opacity={args.ghost_opacity_start}->{args.ghost_opacity_end}")
    print(f"   插帧: mode={args.stretch_mode}, head={args.stretch_head}, ghost={args.stretch_ghost}, fade={args.stretch_fade}, freeze={args.stretch_freeze}, tail={args.stretch_tail}")
    print(f"   起始人物补洞: {args.initial_subject_patch_mode}")
    print(f"   回收画布: background={args.background_mode}, transition={args.recovery_transition_frames}帧")
    print(f"{'='*50}")

    slicer.generate(
        strategy,
        args.start_frame,
        args.freeze_frame,
        end_frame,
        camera_ids=camera_ids,
        ghost_interval=args.ghost_interval,
        edge_feather=args.edge_feather,
        fade_duration_frames=args.fade_duration_frames,
        ghost_opacity_start=args.ghost_opacity_start,
        ghost_opacity_end=args.ghost_opacity_end,
        stretch_head=args.stretch_head,
        stretch_ghost=args.stretch_ghost,
        stretch_fade=args.stretch_fade,
        stretch_freeze=args.stretch_freeze,
        stretch_tail=args.stretch_tail,
        stretch_mode=args.stretch_mode,
        background_mode=args.background_mode,
        recovery_transition_frames=args.recovery_transition_frames,
        initial_subject_patch_mode=args.initial_subject_patch_mode
    )

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f}秒")
