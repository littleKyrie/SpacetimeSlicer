import cv2
import os
import re
import numpy as np
import torch
from models.rife_ncnn import NeuralSlowMotionWriter
from utils.ffmpeg_video import FfmpegH264Writer, resolve_ffmpeg_executable
from utils.opencv_io import imread_required


FRAME_DIR_PATTERN = re.compile(r'^\d+$')


def resolve_output_video_path(output_dir):
    output_name = os.path.basename(os.path.normpath(output_dir))
    if not output_name:
        raise ValueError(f"Cannot derive video name from output directory: {output_dir}")
    return os.path.join(output_dir, f"{output_name}.mp4")


class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25, camera_ids=None,
                 rife_interpolator=None):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.camera_ids = camera_ids if camera_ids is not None else [0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rife_interpolator = rife_interpolator

        # Only numeric subdirectories are frame directories. Generated helper
        # folders such as 重命名数据/ and 原始图片/ must not be indexed as frames.
        self.subdirs = sorted(
            [
                d for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d)) and FRAME_DIR_PATTERN.fullmatch(d)
            ],
            key=lambda value: int(value),
        )

        # 为每个机位存储帧路径
        self.frame_paths_dict = {}
        self.frame_paths_by_frame_dict = {}
        for cam_id in self.camera_ids:
            frame_paths = []
            frame_paths_by_frame = {}
            for subdir in self.subdirs:
                frame_filename = f"{cam_id:03d}.jpg"
                frame_path = os.path.join(input_dir, subdir, frame_filename)
                if os.path.exists(frame_path):
                    frame_idx = int(subdir) - 1
                    frame_paths_by_frame[frame_idx] = frame_path
                    frame_paths.append(frame_path)
            self.frame_paths_dict[cam_id] = frame_paths
            self.frame_paths_by_frame_dict[cam_id] = frame_paths_by_frame

        # 总帧数取第一个机位覆盖的实际帧号范围
        start_cam_frames = self.frame_paths_by_frame_dict[self.camera_ids[0]]
        self.total_frames = max(start_cam_frames.keys()) + 1 if start_cam_frames else 0
        print(f"找到 {self.total_frames} 帧")
        for cam_id in self.camera_ids:
            print(f"  机位 {cam_id}: {len(self.frame_paths_dict[cam_id])} 帧")

    def has_frame(self, idx, camera_id):
        frame_paths_by_frame = getattr(self, 'frame_paths_by_frame_dict', None)
        if frame_paths_by_frame is not None and camera_id in frame_paths_by_frame:
            return idx in frame_paths_by_frame[camera_id]
        return idx < len(self.frame_paths_dict[camera_id])

    def has_frame_range(self, start_idx, end_idx, camera_id):
        return all(self.has_frame(frame_idx, camera_id) for frame_idx in range(start_idx, end_idx))

    def resolve_tail_camera_id(self, camera_ids, freeze_idx, effect_end_idx, requested_tail_camera_id=None):
        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]
        tail_start_idx = freeze_idx + 1
        if tail_start_idx >= effect_end_idx:
            return end_cam if requested_tail_camera_id is None else requested_tail_camera_id

        if requested_tail_camera_id is not None:
            return requested_tail_camera_id

        if self.has_frame_range(tail_start_idx, effect_end_idx, end_cam):
            return end_cam
        if self.has_frame_range(tail_start_idx, effect_end_idx, start_cam):
            return start_cam
        return end_cam

    def read_frame(self, idx, camera_id=None):
        if camera_id is None:
            camera_id = self.camera_ids[0]
        frame_paths_by_frame = getattr(self, 'frame_paths_by_frame_dict', None)
        if frame_paths_by_frame is not None and camera_id in frame_paths_by_frame:
            if idx not in frame_paths_by_frame[camera_id]:
                raise ValueError(f"Camera {camera_id} does not contain frame {idx}")
            return imread_required(frame_paths_by_frame[camera_id][idx])
        return imread_required(self.frame_paths_dict[camera_id][idx])

    def write_frame_repeat(self, out, frame, stretch=1):
        """写入帧，stretch 控制重复次数（默认 1 = 不重复）"""
        for _ in range(max(1, stretch)):
            out.write(frame)

    def create_rife_writer(self, out, factor, start_idx, end_idx):
        if factor > 1 and self.rife_interpolator is None:
            raise ValueError(
                "RIFE interpolator is required when stretch_ghost or stretch_freeze is greater than 1"
            )
        return NeuralSlowMotionWriter(
            out,
            interpolator=self.rife_interpolator,
            factor=factor,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def interpolate_cubic_values(self, p0, p1, p2, p3, ratio):
        ratio_squared = ratio * ratio
        ratio_cubed = ratio_squared * ratio
        interpolated = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * ratio +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * ratio_squared +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * ratio_cubed
        )
        return np.clip(interpolated, np.minimum(p1, p2), np.maximum(p1, p2))

    def build_temporal_median_background(self, camera_id, max_samples=31):
        """Build a clean plate for a static camera without keeping one source-frame subject."""
        frame_count = len(self.frame_paths_dict[camera_id])
        if frame_count == 0:
            raise ValueError(f"Camera {camera_id} has no frames")
        sample_count = min(max_samples, frame_count)
        sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        sampled_frames = [self.read_frame(i, camera_id) for i in sample_indices]
        return np.median(np.stack(sampled_frames, axis=0), axis=0).astype(np.uint8)

    def resolve_initial_subject_replacement(self, mode, camera_id, freeze_idx, patch_frame_idx=None):
        if mode == 'none':
            return None
        if mode == 'median':
            return self.build_temporal_median_background(camera_id)
        if mode == 'freeze':
            return self.read_frame(freeze_idx, camera_id).copy()
        if mode == 'frame':
            resolved_patch_idx = freeze_idx if patch_frame_idx is None else patch_frame_idx
            if resolved_patch_idx >= len(self.frame_paths_dict[camera_id]):
                raise ValueError(f"Camera {camera_id} does not contain patch frame {resolved_patch_idx}")
            return self.read_frame(resolved_patch_idx, camera_id).copy()
        raise ValueError(f"Unknown initial subject patch mode: {mode}")

    def patch_initial_subject_region(self, base_frame, replacement_frame, alpha_mask,
                                     alpha_threshold=1, dilate_iterations=1):
        patch_mask = (alpha_mask > alpha_threshold).astype(np.uint8)
        if dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            patch_mask = cv2.dilate(patch_mask, kernel, iterations=dilate_iterations)
        patch_alpha = np.repeat(patch_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32)
        return (
            base_frame * (1 - patch_alpha) +
            replacement_frame * patch_alpha
        ).astype(np.uint8)

    def resolve_fade_duration(self, effect_frame_count, fade_duration_frames):
        if fade_duration_frames is not None:
            return fade_duration_frames
        return min(effect_frame_count, max(2, self.fps // 2))

    def resolve_effect_schedule(self, effect_start_idx, freeze_idx, fade_duration_frames,
                                recovery_timing='after_freeze'):
        effect_frame_count = freeze_idx - effect_start_idx + 1
        if effect_frame_count < 2:
            raise ValueError("Expected at least two frames between effect_start_idx and freeze_idx")
        if recovery_timing not in ('after_freeze', 'before_freeze'):
            raise ValueError(f"Unknown recovery timing: {recovery_timing}")

        total_fade_frames = self.resolve_fade_duration(effect_frame_count, fade_duration_frames)
        if recovery_timing == 'after_freeze':
            return freeze_idx, total_fade_frames

        max_fade_frames = effect_frame_count - 1
        if total_fade_frames > max_fade_frames:
            if fade_duration_frames is not None:
                raise ValueError(
                    "fade_duration_frames must leave at least one frame for slice generation"
                )
            total_fade_frames = max_fade_frames

        slice_end_idx = freeze_idx - total_fade_frames
        return slice_end_idx, total_fade_frames

    def get_ghost_geometry(self, ghost):
        if 'geometry' in ghost:
            return ghost['geometry']

        binary_alpha = (ghost['alpha'] > 8).astype(np.uint8)
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            binary_alpha, connectivity=8
        )
        if component_count <= 1:
            h, w = ghost['alpha'].shape
            geometry = np.array([w / 2.0, h / 2.0, float(w), float(h)], dtype=np.float32)
        else:
            primary_label = 1 + int(
                np.argmax(component_stats[1:, cv2.CC_STAT_AREA])
            )
            x = component_stats[primary_label, cv2.CC_STAT_LEFT]
            y = component_stats[primary_label, cv2.CC_STAT_TOP]
            w = component_stats[primary_label, cv2.CC_STAT_WIDTH]
            h = component_stats[primary_label, cv2.CC_STAT_HEIGHT]
            geometry = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h)], dtype=np.float32)
        ghost['geometry'] = geometry
        return geometry

    def align_ghost_to_center(self, ghost, target_center):
        """Translate a cutout to the interpolated center without changing its body proportions."""
        frame_h, frame_w = ghost['alpha'].shape
        source_geometry = self.get_ghost_geometry(ghost)
        source_w = max(1, int(round(source_geometry[2])))
        source_h = max(1, int(round(source_geometry[3])))
        source_x = int(round(source_geometry[0] - source_w / 2.0))
        source_y = int(round(source_geometry[1] - source_h / 2.0))
        source_x = min(max(0, source_x), frame_w - 1)
        source_y = min(max(0, source_y), frame_h - 1)
        source_w = min(source_w, frame_w - source_x)
        source_h = min(source_h, frame_h - source_y)

        target_x = int(round(target_center[0] - source_w / 2.0))
        target_y = int(round(target_center[1] - source_h / 2.0))

        frame_crop = ghost['frame'][source_y:source_y + source_h, source_x:source_x + source_w]
        alpha_crop = ghost['alpha'][source_y:source_y + source_h, source_x:source_x + source_w]

        aligned_frame = np.zeros_like(ghost['frame'])
        aligned_alpha = np.zeros_like(ghost['alpha'])
        dst_x0 = max(0, target_x)
        dst_y0 = max(0, target_y)
        dst_x1 = min(frame_w, target_x + source_w)
        dst_y1 = min(frame_h, target_y + source_h)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return aligned_frame, aligned_alpha

        src_x0 = dst_x0 - target_x
        src_y0 = dst_y0 - target_y
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        aligned_frame[dst_y0:dst_y1, dst_x0:dst_x1] = frame_crop[src_y0:src_y1, src_x0:src_x1]
        aligned_alpha[dst_y0:dst_y1, dst_x0:dst_x1] = alpha_crop[src_y0:src_y1, src_x0:src_x1]
        return aligned_frame, aligned_alpha

    def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval, edge_feather,
                        all_ghosts, permanent_indices, out, initial_canvas=None,
                        ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                        stretch_ghost=1,
                        initial_subject_replacement=None,
                        initial_patch_alpha_threshold=1,
                        initial_patch_dilate=1,
                        effect_base_mode='patched_canvas',
                        live_subject_opacity=1.0,
                        live_subject_alpha_threshold=16):
        """Generate the slice segment using either source-frame or patched-canvas compositing."""
        if effect_base_mode not in ('source', 'patched_canvas'):
            raise ValueError(f"Unknown effect base mode: {effect_base_mode}")

        num_ghosts_expected = max(1, ((end_idx - 1 - start_idx) // ghost_interval) + 1)
        ghost_opacities = np.linspace(ghost_opacity_start, ghost_opacity_end, num_ghosts_expected)

        if initial_canvas is not None:
            canvas_ghosts = initial_canvas.copy()
        else:
            canvas_ghosts = self.read_frame(start_idx, camera_id).copy()

        ghost_count = len(permanent_indices)
        stage_writer = self.create_rife_writer(out, stretch_ghost, start_idx, end_idx - 1)
        last_frame_output = None

        for i in range(start_idx, end_idx):
            current_frame = self.read_frame(i, camera_id)
            base_frame = current_frame
            should_be_ghost = ((i - start_idx) % ghost_interval == 0)
            needs_alpha = should_be_ghost or effect_base_mode == 'patched_canvas'
            alpha_mask = None
            if needs_alpha:
                alpha_mask = strategy.process_frame(current_frame, i)

                if edge_feather < 0:
                    kernel = np.ones((3, 3), np.uint8)
                    alpha_mask = cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))

            if i == start_idx and initial_subject_replacement is not None and alpha_mask is not None:
                patched_start_frame = self.patch_initial_subject_region(
                    current_frame,
                    initial_subject_replacement,
                    alpha_mask,
                    alpha_threshold=initial_patch_alpha_threshold,
                    dilate_iterations=initial_patch_dilate,
                )
                base_frame = patched_start_frame
                if effect_base_mode == 'patched_canvas':
                    canvas_ghosts = patched_start_frame.copy()

            if should_be_ghost:
                g_opacity = ghost_opacities[min(ghost_count, len(ghost_opacities) - 1)]
                if effect_base_mode == 'patched_canvas':
                    alpha_normalized = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) / 255.0
                    ghost_alpha = alpha_normalized * g_opacity
                    canvas_ghosts = (
                        current_frame * ghost_alpha +
                        canvas_ghosts * (1 - ghost_alpha)
                    ).astype(np.uint8)

                permanent_indices.append(len(all_ghosts))
                all_ghosts.append({
                    'frame': current_frame.copy(),
                    'alpha': alpha_mask.copy(),
                    'opacity': g_opacity,
                })
                ghost_count += 1

            if effect_base_mode == 'source':
                frame_output = self.compose_static_ghosts(
                    base_frame, all_ghosts, permanent_indices
                )
            else:
                live_mask = (alpha_mask > live_subject_alpha_threshold).astype(np.float32)
                live_alpha = np.repeat(live_mask[:, :, np.newaxis], 3, axis=2) * live_subject_opacity
                frame_output = (
                    current_frame * live_alpha +
                    canvas_ghosts * (1 - live_alpha)
                ).astype(np.uint8)
            last_frame_output = frame_output

            stage_writer.write(frame_output, i)

            if effect_base_mode == 'patched_canvas' and not should_be_ghost:
                all_ghosts.append({
                    'frame': current_frame.copy(),
                    'alpha': alpha_mask.copy(),
                    'opacity': 1.0,
                })

            print(f"  > 机位 {camera_id} 渲染特效帧: {i}/{end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        return ghost_count, last_frame_output

    def process_freeze_transition(self, camera_ids, freeze_idx, out, stretch_freeze=1,
                                  interpolation_mode='rife'):
        """处理凝结转场阶段（回收残影后的多视角环绕）
        按输入机位顺序输出每个机位在 freeze_idx 的帧，产生多视角效果
        """
        print(f"\n处理凝结转场: {len(camera_ids)} 个机位 ({camera_ids[0]} -> {camera_ids[-1]})")

        if interpolation_mode == 'rife':
            stage_writer = self.create_rife_writer(out, stretch_freeze, 0, len(camera_ids) - 1)
            for i, cam_id in enumerate(camera_ids):
                frame = self.read_frame(freeze_idx, cam_id)
                stage_writer.write(frame, i)
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        elif interpolation_mode == 'repeat':
            for i, cam_id in enumerate(camera_ids):
                frame = self.read_frame(freeze_idx, cam_id)
                self.write_frame_repeat(out, frame, stretch_freeze)
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        elif interpolation_mode == 'blend':
            previous_frame = None
            for i, cam_id in enumerate(camera_ids):
                frame = self.read_frame(freeze_idx, cam_id)
                if previous_frame is not None:
                    for step in range(1, stretch_freeze):
                        ratio = step / stretch_freeze
                        out.write(cv2.addWeighted(previous_frame, 1.0 - ratio, frame, ratio, 0))
                out.write(frame)
                previous_frame = frame
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        else:
            raise ValueError(f"Unknown freeze interpolation mode: {interpolation_mode}")

        print(f"\n  凝结转场完成: 共 {len(camera_ids)} 个视角 × {stretch_freeze}")

    def interpolate_ghost(self, all_ghosts, position):
        lower_idx = int(np.floor(position))
        upper_idx = min(lower_idx + 1, len(all_ghosts) - 1)
        ratio = position - lower_idx
        if upper_idx == lower_idx or ratio <= 0:
            return all_ghosts[lower_idx]

        p0_idx = max(0, lower_idx - 1)
        p3_idx = min(len(all_ghosts) - 1, upper_idx + 1)
        target_geometry = self.interpolate_cubic_values(
            self.get_ghost_geometry(all_ghosts[p0_idx]),
            self.get_ghost_geometry(all_ghosts[lower_idx]),
            self.get_ghost_geometry(all_ghosts[upper_idx]),
            self.get_ghost_geometry(all_ghosts[p3_idx]),
            ratio,
        )
        target_center = target_geometry[:2]
        lower_frame, lower_alpha = self.align_ghost_to_center(
            all_ghosts[lower_idx], target_center
        )
        upper_frame, upper_alpha = self.align_ghost_to_center(
            all_ghosts[upper_idx], target_center
        )
        lower_alpha_f = lower_alpha.astype(np.float32) / 255.0
        upper_alpha_f = upper_alpha.astype(np.float32) / 255.0
        mixed_alpha_f = lower_alpha_f * (1.0 - ratio) + upper_alpha_f * ratio
        mixed_premultiplied = (
            lower_frame.astype(np.float32) * lower_alpha_f[:, :, np.newaxis] * (1.0 - ratio) +
            upper_frame.astype(np.float32) * upper_alpha_f[:, :, np.newaxis] * ratio
        )
        mixed_frame = np.divide(
            mixed_premultiplied,
            mixed_alpha_f[:, :, np.newaxis],
            out=np.zeros_like(mixed_premultiplied),
            where=mixed_alpha_f[:, :, np.newaxis] > 1e-6,
        )
        return {
            'frame': np.clip(mixed_frame, 0, 255).astype(np.uint8),
            'alpha': np.clip(mixed_alpha_f * 255.0, 0, 255).astype(np.uint8),
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

    def compose_static_ghosts(self, background, all_ghosts, permanent_indices):
        current_canvas = background.copy()
        for p_idx in permanent_indices:
            ghost = all_ghosts[p_idx]
            ghost_opacity = ghost.get('opacity', 1.0)
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
            ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            out.write(cv2.addWeighted(first_frame, 1.0 - ratio, second_frame, ratio, 0))

    def process_fade_out(self, out, all_ghosts, permanent_indices, background, total_fade_frames,
                         stretch_fade=1, transition_from=None, recovery_transition_frames=0):
        """处理片尾淡出（应用残影透明度渐变，保留原残影的透明度）"""
        output_fade_frames = total_fade_frames * max(1, stretch_fade)
        print(f"\n写入片尾（定格 + 残影回收消失）...")
        print(f"  > 残影消失过程: {len(permanent_indices)} 个永久残影, "
              f"{total_fade_frames} 逻辑帧 × {stretch_fade} = {output_fade_frames} 输出帧")

        if len(all_ghosts) == 0 or len(permanent_indices) == 0:
            print("  没有残影需要回收，直接写入定格帧")
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
            print("  无法构建残影序列，直接写入定格帧")
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
                 stretch_freeze=1, stretch_tail=1,
                 background_mode='freeze', recovery_transition_frames=3,
                 recovery_timing='after_freeze',
                 freeze_interp_mode='rife',
                 tail_camera_id=None,
                 initial_canvas_mode='patched_start',
                 initial_subject_patch_mode='freeze',
                 initial_subject_patch_frame=None,
                 initial_patch_alpha_threshold=1,
                 initial_patch_dilate=1,
                 effect_base_mode='patched_canvas',
                 live_subject_opacity=1.0,
                 live_subject_alpha_threshold=16,
                 ffmpeg_executable=None,
                 h264_crf=18,
                 h264_preset='medium'):
        """
        生成时空切片视频（残影渐变 → 回收 → 多视角凝结 → 继续播放）

        流程:
          1. 片头: 0 -> effect_start_idx（固定机位原样播放）
          2. 特效段: effect_start_idx -> slice_end_idx（残影+透明度渐变）
             - 每个输出帧以原始视频帧为底
             - 只在 ghost_interval 命中的帧分割人物并捕获为透明切片
             - 已捕获切片按各自 opacity 叠加到当前原始帧上
          3. 回收窗口: slice_end_idx+1 -> freeze_idx（回收所有残影）
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
        if initial_canvas_mode not in ('patched_start', 'clean'):
            raise ValueError(f"Unknown initial canvas mode: {initial_canvas_mode}")
        if initial_subject_patch_mode not in ('none', 'median', 'freeze', 'frame'):
            raise ValueError(f"Unknown initial subject patch mode: {initial_subject_patch_mode}")
        if not 0 <= initial_patch_alpha_threshold <= 255:
            raise ValueError("initial_patch_alpha_threshold must be between 0 and 255")
        if initial_patch_dilate < 0:
            raise ValueError("initial_patch_dilate must not be negative")
        if effect_base_mode not in ('source', 'patched_canvas'):
            raise ValueError(f"Unknown effect base mode: {effect_base_mode}")
        if not 0.0 <= live_subject_opacity <= 1.0:
            raise ValueError("live_subject_opacity must be between 0 and 1")
        if not 0 <= live_subject_alpha_threshold <= 255:
            raise ValueError("live_subject_alpha_threshold must be between 0 and 255")
        if not effect_start_idx < freeze_idx < effect_end_idx:
            raise ValueError("Expected effect_start_idx < freeze_idx < effect_end_idx")
        if fade_duration_frames is not None and fade_duration_frames < 1:
            raise ValueError("fade_duration_frames must be at least 1")
        if recovery_transition_frames < 0:
            raise ValueError("recovery_transition_frames must not be negative")
        if recovery_timing not in ('after_freeze', 'before_freeze'):
            raise ValueError(f"Unknown recovery timing: {recovery_timing}")
        if min(stretch_head, stretch_ghost, stretch_fade, stretch_freeze, stretch_tail) < 1:
            raise ValueError("stretch values must be at least 1")
        if freeze_interp_mode not in ('rife', 'repeat', 'blend'):
            raise ValueError(f"Unknown freeze interpolation mode: {freeze_interp_mode}")
        if not 0 <= h264_crf <= 51:
            raise ValueError("h264_crf must be between 0 and 51")
        resolved_ffmpeg = resolve_ffmpeg_executable(ffmpeg_executable)
        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]
        tail_cam = self.resolve_tail_camera_id(
            camera_ids,
            freeze_idx,
            effect_end_idx,
            requested_tail_camera_id=tail_camera_id,
        )
        if not self.has_frame(freeze_idx, start_cam):
            raise ValueError(f"Camera {start_cam} does not contain freeze frame {freeze_idx}")
        for camera_id in camera_ids:
            if not self.has_frame(freeze_idx, camera_id):
                raise ValueError(f"Camera {camera_id} does not contain freeze frame {freeze_idx}")
        for frame_idx in range(0, freeze_idx + 1):
            if not self.has_frame(frame_idx, start_cam):
                raise ValueError(f"Camera {start_cam} does not contain frame {frame_idx}")
        for frame_idx in range(freeze_idx + 1, effect_end_idx):
            if not self.has_frame(frame_idx, tail_cam):
                raise ValueError(f"Camera {tail_cam} does not contain tail frame {frame_idx}")

        # Old verbose run directory retained for reference:
        # patch_suffix = initial_subject_patch_mode
        # if initial_subject_patch_mode == 'frame':
        #     patch_suffix = f"frame{freeze_idx if initial_subject_patch_frame is None else initial_subject_patch_frame}"
        # stretch_suffix = f"_sh{stretch_head}_sg{stretch_ghost}_sfd{stretch_fade}_sfz{stretch_freeze}_st{stretch_tail}_rife_fim{freeze_interp_mode}_tail{tail_cam}_patch{patch_suffix}_canvas{initial_canvas_mode}_base{effect_base_mode}_live{live_subject_alpha_threshold}_recoverybg{background_mode}_rtime{recovery_timing}_rt{recovery_transition_frames}"
        # run_name = f"freeze_{start_cam}_to_{end_cam}_seq{len(camera_ids)}_s{effect_start_idx}_f{freeze_idx}_e{effect_end_idx}{stretch_suffix}"
        # output_dir = os.path.join(self.output_root, run_name)
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)

        video_path = resolve_output_video_path(output_dir)
        print(f"Output video: {video_path}")
        print(
            f"H.264 encoding: libx264, CRF {h264_crf}, preset {h264_preset} "
            f"(FFmpeg: {resolved_ffmpeg})"
        )

        sample_frame = self.read_frame(0, start_cam)
        h, w = sample_frame.shape[:2]
        out = FfmpegH264Writer(
            video_path,
            fps=self.fps,
            frame_size=(w, h),
            executable=resolved_ffmpeg,
            crf=h264_crf,
            preset=h264_preset,
        )
        if not out.isOpened():
            try:
                out.release()
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to start FFmpeg video writer: {exc}") from exc
            raise RuntimeError(
                f"Failed to start FFmpeg video writer for: {video_path}"
            )

        all_ghosts = []
        permanent_indices = []

        slice_end_idx, total_fade_frames = self.resolve_effect_schedule(
            effect_start_idx, freeze_idx, fade_duration_frames, recovery_timing
        )

        # ============ 1. 写入片头 ============
        print("写入片头...")
        for i in range(0, effect_start_idx):
            self.write_frame_repeat(out, self.read_frame(i, start_cam), stretch_head)
        if effect_start_idx > 0:
            print(f"  片头完成: 0 -> {effect_start_idx-1} ({effect_start_idx} 帧, ×{stretch_head})")

        # ============ 2. 特效段: 固定机位 + 残影 + 透明度渐变 ============
        print(f"\n特效段: 机位 {start_cam} ({effect_start_idx} -> {slice_end_idx})")
        print(f"   残影透明度: {ghost_opacity_start:.0%} -> {ghost_opacity_end:.0%}, 插帧 ×{stretch_ghost}")
        if recovery_timing == 'before_freeze':
            print(f"   回收窗口: {slice_end_idx + 1} -> {freeze_idx} ({total_fade_frames} 帧)")
        else:
            print(f"   回收窗口: freeze 帧之后插入 {total_fade_frames} 帧")
        print("   特效底图模式: source（原始视频帧 + 已捕获切片）")
        source_start_frame = self.read_frame(effect_start_idx, start_cam).copy()
        initial_subject_replacement = self.resolve_initial_subject_replacement(
            initial_subject_patch_mode,
            start_cam,
            freeze_idx,
            initial_subject_patch_frame,
        )
        if initial_canvas_mode == 'patched_start':
            generation_canvas = source_start_frame
        elif initial_canvas_mode == 'clean':
            generation_canvas = (
                source_start_frame if initial_subject_replacement is None
                else initial_subject_replacement
            )
        else:
            raise ValueError(f"Unknown initial canvas mode: {initial_canvas_mode}")
        print(f"   Effect base mode: {effect_base_mode}, initial canvas: {initial_canvas_mode}")
        ghost_count, last_effect_frame = self.process_segment(
            strategy, start_cam, effect_start_idx, slice_end_idx + 1,
            ghost_interval, edge_feather, all_ghosts, permanent_indices, out,
            initial_canvas=generation_canvas,
            ghost_opacity_start=ghost_opacity_start,
            ghost_opacity_end=ghost_opacity_end,
            stretch_ghost=stretch_ghost,
            initial_subject_replacement=initial_subject_replacement,
            initial_patch_alpha_threshold=initial_patch_alpha_threshold,
            initial_patch_dilate=initial_patch_dilate,
            effect_base_mode=effect_base_mode,
            live_subject_opacity=live_subject_opacity,
            live_subject_alpha_threshold=live_subject_alpha_threshold,
        )

        # ============ 3. 片尾淡出: 回收所有残影 ============
        if background_mode == 'median':
            fade_background = self.build_temporal_median_background(start_cam)
        elif background_mode == 'freeze':
            fade_background = self.read_frame(freeze_idx, start_cam).copy()
        elif background_mode == 'start':
            fade_background = source_start_frame
        else:
            raise ValueError(f"Unknown background mode: {background_mode}")
        self.process_fade_out(out, all_ghosts, permanent_indices, fade_background, total_fade_frames,
                              stretch_fade=stretch_fade,
                              transition_from=last_effect_frame,
                              recovery_transition_frames=recovery_transition_frames)

        # ============ 4. 凝结转场: 多机位环绕 ============
        print(f"\n进入凝结状态，多机位环绕...")
        self.process_freeze_transition(
            camera_ids,
            freeze_idx,
            out,
            stretch_freeze=stretch_freeze,
            interpolation_mode=freeze_interp_mode,
        )

        # ============ 5. 继续播放: 凝结帧之后 -> 结束帧 ============
        if freeze_idx + 1 < effect_end_idx:
            print(f"\n继续播放: 机位 {tail_cam} ({freeze_idx + 1} -> {effect_end_idx - 1}), 插帧 ×{stretch_tail}")
            for i in range(freeze_idx + 1, effect_end_idx):
                self.write_frame_repeat(out, self.read_frame(i, tail_cam), stretch_tail)
                print(f"  > 播放帧 {i}/{effect_end_idx - 1}", end='\r')
            print()

        # ============ 6. 收尾 ============
        out.release()
        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"FFmpeg did not create a valid output file: {video_path}")
        print(f"视频已输出（H.264）！保存在: {video_path}")
        print(f"   残影: ghost_interval={ghost_interval}, 共 {ghost_count} 个, "
              f"透明度 {ghost_opacity_start:.0%}->{ghost_opacity_end:.0%}")
        print(f"   片尾淡出: {total_fade_frames} 帧 ×{stretch_fade}")
        print(f"   凝结转场: {len(camera_ids)} 个机位视角 ×{stretch_freeze}")
        if freeze_idx + 1 < effect_end_idx:
            print(f"   继续播放: {freeze_idx + 1} -> {effect_end_idx - 1} ({effect_end_idx - freeze_idx - 1} 帧, 机位 {tail_cam}, ×{stretch_tail})")
