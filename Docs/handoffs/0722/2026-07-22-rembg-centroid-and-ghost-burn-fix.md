# 2026-07-22 — rembg 分割适配：质心锚点 + ghost-burn 阈值滤除

## 背景

项目默认使用 RVM（RobustVideoMatting）做人物分割。为支持手持物品（扇子）的分割，切换到 `rembg-isnet-general-use`。rembg 比 RVM 多分割了扇子等手持物品，但暴露了两个问题：

1. **残影回收阶段位置偏移** — 回收时残影汇合位置相对背景人物右偏
2. **特效段边缘像素块** — 人体边缘出现半透明像素累积

## 根因

### 问题 1：包围盒中心 ≠ 人体中心

`get_ghost_geometry()` 使用 alpha blob 的包围盒几何中心作为残影的"锚点"。RVM 只分割人体（锚点 ≈ 人体中心），rembg 额外分割了扇子，扇子向右延伸拉偏了包围盒，导致锚点 ≠ 人体中心。残影回收时 `interpolate_ghost` 通过 Catmull-Rom 样条插值各帧的锚点位置，不同帧的扇子角度不同 → 锚点偏移量不同 → 残影对齐后与背景人物错位。

### 问题 2：canvas 累积与 live-subject mask 的 alpha 不对称

特效段 `process_segment` 在 `patched_canvas` 模式下：
- **ghost burn**（残影烧入 canvas）：使用连续 alpha → α=5 的低置信度边缘像素也参与累积
- **live subject**（活体人物遮挡 canvas）：使用二元阈值 `alpha > 16` → α≤16 的像素透出 canvas

rembg 的边缘 alpha 不够锐利（α=5~20 的过渡像素多），导致边缘像素在 canvas 中累积了残影碎屑，却又因 α≤16 无法被活体人物覆盖，形成可见的半透明 fringe。回收阶段不存在此问题（直接 overlay 到干净背景，无二元阈值）。

## 解决方案

### 改动 1：`--centroid-mask`（质心锚点）

**文件**：[build_spacetime_slicer.py](../../build_spacetime_slicer.py), [models/spacetime_slicer.py](../../models/spacetime_slicer.py)

- 新增 `--centroid-mask` / `--no-centroid-mask` CLI flag（默认 `False`）
- 启用后，`get_ghost_geometry()` 使用 `cv2.moments()` 计算 alpha blob 的**面积加权质心**作为锚点，替代包围盒中心
- `align_ghost_to_center()` 裁剪时仍用包围盒边界，但平移目标以质心为锚（新增 `_get_alpha_bbox` 辅助方法）
- 扇子因细长而面积小，质心几乎不受其影响，锚点始终接近人体中心
- **auto-detect**：`main()` 检测到 `--method rembg-*` 时自动启用（用户可通过 `--no-centroid-mask` 显式关闭）

**关键代码路径**：
```
main() → args.centroid_mask → slicer.generate(centroid_mask=…) 
→ self.use_centroid = centroid_mask
→ get_ghost_geometry() 读取 self.use_centroid → 分支
→ align_ghost_to_center() 读取 self.use_centroid → 分支
```

### 改动 2：ghost-burn 阈值滤除

**文件**：[models/spacetime_slicer.py](../../models/spacetime_slicer.py) `process_segment()`

- 残影烧入 canvas 时新增过滤：`burn_mask = alpha_mask > live_subject_alpha_threshold`
- 仅 `burn_mask=True` 的像素参与 canvas 累积（`ghost_alpha = alpha_normalized * opacity` 受限）
- `all_ghosts` 中**仍存储完整 alpha**（回收阶段使用完整 alpha 做平滑过渡）
- 复用 `live_subject_alpha_threshold`（默认 16）——残影烧入与活体 mask 共享同一置信度边界，空间对齐
- 对 RVM 几乎无影响（RVM 边缘锐利，α≤16 的像素极少）

**参数文档更新**：`--live_subject_alpha_threshold` 的帮助文本、`generate()` 的 docstring、以及两处使用点的内联注释均已更新，明确标注双重用途。

## 受影响文件

| 文件 | 改动性质 | 说明 |
|------|----------|------|
| `build_spacetime_slicer.py` | 新增 | `import sys`, `argv_has_option()`, `--centroid-mask` flag, auto-detect 逻辑 |
| `models/spacetime_slicer.py` | 新增 | `_get_alpha_bbox()`, `centroid_mask` 参数/属性, 质心分支, ghost-burn 阈值, 注释 |

## 未包含的改动（非本次）

以下文件在本次会话前已有未提交修改，**不属于本次改动**：
- `batch_run.py`
- `config_effect75_orbit90.txt`
- `test/test_batch_run.py`
- `test/test_spacetime_slicer.py`（含 `resolve_output_video_path` 自动递增逻辑）

## 用法

```powershell
# rembg 方法 — centroid-mask 自动启用，ghost-burn 阈值生效
uv run python batch_run.py `
  --sub_dir 0722临时测试 `
  --pre-frame-count 75 --start_frame 1 --freeze_frame 75 `
  --dataset QPG_88-2026-07-22-091351 `
  --method rembg-isnet-general-use --force

# RVM 方法 — 行为不变（centroid 默认关，ghost-burn 阈值对 RVM 近乎空操作）
uv run python batch_run.py `
  --sub_dir 0722 --pre-frame-count 75 --start_frame 1 --freeze_frame 75 `
  --dataset QPB_75-2026-07-22-085332 --force

# 手动控制
uv run python batch_run.py ... --centroid-mask          # 强制启用质心
uv run python batch_run.py ... --no-centroid-mask       # 强制关闭（覆盖 auto-detect）
uv run python batch_run.py ... --live_subject_alpha_threshold 32  # 提高阈值
```

## suggested skills

- `code-introduce`: 向他人介绍本次改动
- `simplify`: 检查本次改动是否有可简化/合并的冗余逻辑
