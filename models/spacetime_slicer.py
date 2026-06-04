import cv2
import os
import numpy as np
import torch
from models.rife_ncnn import NeuralSlowMotionWriter


class SpacetimeSlicer:
    def __init__(self, input_dir, output_root, fps=25, camera_ids=None,
                 rife_interpolator=None):
        self.input_dir = input_dir
        self.output_root = output_root
        self.fps = fps
        self.camera_ids = camera_ids if camera_ids is not None else [0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rife_interpolator = rife_interpolator

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

    def resolve_fade_duration(self, effect_frame_count, fade_duration_frames):
        if fade_duration_frames is not None:
            return fade_duration_frames
        return max(effect_frame_count, max(2, self.fps // 2))

    def get_ghost_geometry(self, ghost):
        if 'geometry' in ghost:
            return ghost['geometry']

        mask_points = cv2.findNonZero((ghost['alpha'] > 8).astype(np.uint8))
        if mask_points is None:
            h, w = ghost['alpha'].shape
            geometry = np.array([w / 2.0, h / 2.0, float(w), float(h)], dtype=np.float32)
        else:
            x, y, w, h = cv2.boundingRect(mask_points)
            geometry = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h)], dtype=np.float32)
        ghost['geometry'] = geometry
        return geometry

    def align_ghost_to_geometry(self, ghost, target_geometry):
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

        target_w = max(1, int(round(target_geometry[2])))
        target_h = max(1, int(round(target_geometry[3])))
        target_x = int(round(target_geometry[0] - target_w / 2.0))
        target_y = int(round(target_geometry[1] - target_h / 2.0))

        frame_crop = ghost['frame'][source_y:source_y + source_h, source_x:source_x + source_w]
        alpha_crop = ghost['alpha'][source_y:source_y + source_h, source_x:source_x + source_w]
        resized_frame = cv2.resize(frame_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_alpha = cv2.resize(alpha_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        aligned_frame = np.zeros_like(ghost['frame'])
        aligned_alpha = np.zeros_like(ghost['alpha'])
        dst_x0 = max(0, target_x)
        dst_y0 = max(0, target_y)
        dst_x1 = min(frame_w, target_x + target_w)
        dst_y1 = min(frame_h, target_y + target_h)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return aligned_frame, aligned_alpha

        src_x0 = dst_x0 - target_x
        src_y0 = dst_y0 - target_y
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        aligned_frame[dst_y0:dst_y1, dst_x0:dst_x1] = resized_frame[src_y0:src_y1, src_x0:src_x1]
        aligned_alpha[dst_y0:dst_y1, dst_x0:dst_x1] = resized_alpha[src_y0:src_y1, src_x0:src_x1]
        return aligned_frame, aligned_alpha

    def apply_edge_feather(self, alpha_mask, edge_feather):
        if edge_feather < 0:
            kernel = np.ones((3, 3), np.uint8)
            return cv2.erode(alpha_mask, kernel, iterations=abs(edge_feather))
        return alpha_mask

    def process_segment(self, strategy, camera_id, start_idx, end_idx, ghost_interval, edge_feather,
                        all_ghosts, permanent_indices, out, initial_canvas=None,
                        initial_subject_replacement=None,
                        ghost_opacity_start=0.2, ghost_opacity_end=1.0,
                        stretch_ghost=1):
        """Accumulate translucent ghosts while keeping the current subject fully visible."""
        num_ghosts_expected = max(1, ((end_idx - 1 - start_idx) // ghost_interval) + 1)
        ghost_opacities = np.linspace(ghost_opacity_start, ghost_opacity_end, num_ghosts_expected)

        # A clean initial canvas prevents the first source-frame subject from surviving recovery.
        if initial_canvas is not None:
            canvas_ghosts = initial_canvas.copy()
        else:
            canvas_ghosts = self.read_frame(start_idx, camera_id).copy()

        ghost_count = len(permanent_indices)
        stage_writer = self.create_rife_writer(out, stretch_ghost, start_idx, end_idx - 1)
        last_frame_output = canvas_ghosts.copy()

        for i in range(start_idx, end_idx):
            current_frame = self.read_frame(i, camera_id)
            alpha_mask = strategy.process_frame(current_frame, i)
            alpha_mask = self.apply_edge_feather(alpha_mask, edge_feather)

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

            stage_writer.write(frame_output, i)

            all_ghosts.append({
                'frame': current_frame.copy(),
                'alpha': alpha_mask.copy(),
                'source_idx': i,
                'camera_id': camera_id,
                'opacity': ghost_opacities[min(ghost_count - 1, len(ghost_opacities) - 1)] if should_be_ghost else 1.0
            })

            print(f"  > 机位 {camera_id} 渲染特效帧: {i}/{end_idx-1} (已累积 {ghost_count} 个残影)", end='\r')

        return canvas_ghosts, ghost_count, last_frame_output

    def get_ghost_view(self, ghost, camera_id, strategy=None, edge_feather=0):
        if camera_id is None or ghost.get('camera_id') == camera_id:
            return ghost

        views = ghost.setdefault('views', {})
        if camera_id in views:
            return views[camera_id]

        source_idx = ghost.get('source_idx')
        if source_idx is None or strategy is None:
            return ghost

        frame = self.read_frame(source_idx, camera_id)
        alpha = strategy.process_frame(frame, source_idx)
        alpha = self.apply_edge_feather(alpha, edge_feather)
        view = {
            'frame': frame.copy(),
            'alpha': alpha.copy(),
            'source_idx': source_idx,
            'camera_id': camera_id,
            'opacity': ghost.get('opacity', 1.0),
        }
        views[camera_id] = view
        return view

    def build_camera_ghosts(self, all_ghosts, camera_id, strategy=None, edge_feather=0):
        return [
            self.get_ghost_view(ghost, camera_id, strategy, edge_feather)
            for ghost in all_ghosts
        ]

    def compose_static_ghosts(self, all_ghosts, permanent_indices, background,
                              strategy=None, camera_id=None, edge_feather=0):
        current_canvas = background.copy()
        for p_idx in permanent_indices:
            if p_idx >= len(all_ghosts):
                continue
            ghost = self.get_ghost_view(all_ghosts[p_idx], camera_id, strategy, edge_feather)
            ghost_opacity = ghost.get('opacity', 1.0)
            ghost_alpha_3ch = np.repeat(
                ghost['alpha'][:, :, np.newaxis], 3, axis=2
            ) / 255.0 * ghost_opacity
            current_canvas = (
                ghost['frame'] * ghost_alpha_3ch + current_canvas * (1 - ghost_alpha_3ch)
            ).astype(np.uint8)
        return current_canvas

    def process_freeze_transition(self, camera_ids, freeze_idx, out, stretch_freeze=1,
                                  interpolation_mode='rife', all_ghosts=None,
                                  permanent_indices=None, strategy=None,
                                  edge_feather=0):
        """Carry created slices through the freeze orbit in camera-id order.
        """
        print(f"\n处理凝结转场: {len(camera_ids)} 个机位 ({camera_ids[0]} -> {camera_ids[-1]})")

        all_ghosts = all_ghosts or []
        permanent_indices = permanent_indices or []
        last_transition_frame = None

        def build_freeze_frame(cam_id):
            base_frame = self.read_frame(freeze_idx, cam_id)
            if len(all_ghosts) == 0 or len(permanent_indices) == 0:
                return base_frame
            return self.compose_static_ghosts(
                all_ghosts,
                permanent_indices,
                base_frame,
                strategy=strategy,
                camera_id=cam_id,
                edge_feather=edge_feather,
            )

        if interpolation_mode == 'rife':
            stage_writer = self.create_rife_writer(out, stretch_freeze, 0, len(camera_ids) - 1)
            for i, cam_id in enumerate(camera_ids):
                frame = build_freeze_frame(cam_id)
                stage_writer.write(frame, i)
                last_transition_frame = frame
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        elif interpolation_mode == 'repeat':
            for i, cam_id in enumerate(camera_ids):
                frame = build_freeze_frame(cam_id)
                self.write_frame_repeat(out, frame, stretch_freeze)
                last_transition_frame = frame
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        elif interpolation_mode == 'blend':
            previous_frame = None
            for i, cam_id in enumerate(camera_ids):
                frame = build_freeze_frame(cam_id)
                if previous_frame is not None:
                    for step in range(1, stretch_freeze):
                        ratio = step / stretch_freeze
                        out.write(cv2.addWeighted(previous_frame, 1.0 - ratio, frame, ratio, 0))
                out.write(frame)
                previous_frame = frame
                last_transition_frame = frame
                print(f"    机位 {cam_id} 转场帧 {i+1}/{len(camera_ids)}", end='\r')
        else:
            raise ValueError(f"Unknown freeze interpolation mode: {interpolation_mode}")

        print(f"\n  凝结转场完成: 共 {len(camera_ids)} 个视角 × {stretch_freeze}")

        return last_transition_frame

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
        lower_frame, lower_alpha = self.align_ghost_to_geometry(
            all_ghosts[lower_idx], target_geometry
        )
        upper_frame, upper_alpha = self.align_ghost_to_geometry(
            all_ghosts[upper_idx], target_geometry
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
                 initial_subject_patch_mode='median', freeze_interp_mode='rife'):
        """
        生成时空切片视频（切片生成 → 携带切片环绕 → 回收 → 继续播放）

        流程:
          1. 片头: 0 -> effect_start_idx（固定机位原样播放）
          2. 特效段: effect_start_idx -> freeze_idx（按 ghost_interval 生成切片）
          3. 凝结转场: 按 camera_ids 顺序输出各机位 freeze_idx 帧并携带切片
          4. 片尾淡出: 环绕结束后回收所有切片
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
        if freeze_interp_mode not in ('rife', 'repeat', 'blend'):
            raise ValueError(f"Unknown freeze interpolation mode: {freeze_interp_mode}")
        start_cam = camera_ids[0]
        end_cam = camera_ids[-1]
        if freeze_idx >= len(self.frame_paths_dict[start_cam]):
            raise ValueError(f"Camera {start_cam} does not contain freeze frame {freeze_idx}")
        for camera_id in camera_ids:
            if freeze_idx >= len(self.frame_paths_dict[camera_id]):
                raise ValueError(f"Camera {camera_id} does not contain freeze frame {freeze_idx}")
        if effect_end_idx > len(self.frame_paths_dict[end_cam]):
            raise ValueError(f"Camera {end_cam} does not contain frames up to {effect_end_idx - 1}")

        stretch_suffix = f"_sh{stretch_head}_sg{stretch_ghost}_sfd{stretch_fade}_sfz{stretch_freeze}_st{stretch_tail}_rife_fim{freeze_interp_mode}_patch{initial_subject_patch_mode}_recoverybg{background_mode}_rt{recovery_transition_frames}"
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
        effective_ghost_opacity_start = 1.0
        effective_ghost_opacity_end = 1.0

        # ============ 1. 写入片头 ============
        print("写入片头...")
        for i in range(0, effect_start_idx):
            self.write_frame_repeat(out, self.read_frame(i, start_cam), stretch_head)
        if effect_start_idx > 0:
            print(f"  片头完成: 0 -> {effect_start_idx-1} ({effect_start_idx} 帧, ×{stretch_head})")

        # ============ 2. 特效段: 固定机位 + 残影 + 透明度渐变 ============
        print(f"\n特效段: 机位 {start_cam} ({effect_start_idx} -> {freeze_idx})")
        print(f"   残影透明度: {effective_ghost_opacity_start:.0%} -> {effective_ghost_opacity_end:.0%}, 插帧 ×{stretch_ghost}")
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
            ghost_opacity_start=effective_ghost_opacity_start,
            ghost_opacity_end=effective_ghost_opacity_end,
            stretch_ghost=stretch_ghost
        )

        # ============ 3. 凝结转场: 多机位环绕，携带已经产生的切片 ============
        print(f"\n进入凝结状态，多机位环绕...")
        last_orbit_frame = self.process_freeze_transition(
            camera_ids,
            freeze_idx,
            out,
            stretch_freeze=stretch_freeze,
            interpolation_mode=freeze_interp_mode,
            all_ghosts=all_ghosts,
            permanent_indices=permanent_indices,
            strategy=strategy,
            edge_feather=edge_feather,
        )

        # ============ 4. 片尾淡出: 环绕后回收所有残影 ============
        if background_mode == 'median':
            fade_background = self.build_temporal_median_background(end_cam)
        elif background_mode == 'freeze':
            fade_background = self.read_frame(freeze_idx, end_cam).copy()
        elif background_mode == 'start':
            fade_background = self.read_frame(effect_start_idx, end_cam).copy()
        else:
            raise ValueError(f"Unknown background mode: {background_mode}")
        fade_ghosts = self.build_camera_ghosts(
            all_ghosts,
            end_cam,
            strategy=strategy,
            edge_feather=edge_feather,
        )
        self.process_fade_out(out, fade_ghosts, permanent_indices, fade_background, total_fade_frames,
                              stretch_fade=stretch_fade,
                              transition_from=last_orbit_frame,
                              recovery_transition_frames=recovery_transition_frames)

        # ============ 5. 继续播放: 凝结帧之后 -> 结束帧 ============
        if freeze_idx + 1 < effect_end_idx:
            print(f"\n继续播放: 机位 {end_cam} ({freeze_idx + 1} -> {effect_end_idx - 1}), 插帧 ×{stretch_tail}")
            for i in range(freeze_idx + 1, effect_end_idx):
                self.write_frame_repeat(out, self.read_frame(i, end_cam), stretch_tail)
                print(f"  > 播放帧 {i}/{effect_end_idx - 1}", end='\r')
            print()

        # ============ 6. 收尾 ============
        out.release()
        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"Video writer did not create a valid output file: {video_path}")
        print(f"\n视频已输出！保存在: {video_path}")
        print(f"   残影: ghost_interval={ghost_interval}, 共 {ghost_count} 个, "
              f"透明度 {effective_ghost_opacity_start:.0%}->{effective_ghost_opacity_end:.0%}")
        print(f"   片尾淡出: {total_fade_frames} 帧 ×{stretch_fade}")
        print(f"   凝结转场: {len(camera_ids)} 个机位视角 ×{stretch_freeze}")
        if freeze_idx + 1 < effect_end_idx:
            print(f"   继续播放: {freeze_idx + 1} -> {effect_end_idx - 1} ({effect_end_idx - freeze_idx - 1} 帧, 机位 {end_cam}, ×{stretch_tail})")
