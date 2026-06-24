# SpacetimeSlicer 脚本使用说明

本文档说明以下三个脚本的用途和调用方式：

- `utils/reorganize_frame_images.py`：把原始连续图片整理为切片程序需要的帧/机位目录结构。
- `build_spacetime_slicer.py`：根据已经整理的数据生成时空切片视频。
- `batch_run.py`：检测并整理图片，然后生成时空切片视频。

命令均假设当前工作目录为项目根目录。

## 配置覆盖规则

三个脚本统一遵循以下优先级：

```text
脚本内置默认值 < JSON 配置文件 < 本次命令行参数
```

命令行只需要提供本次需要修改的参数，其余参数继续使用 JSON 配置值。

## 1. utils/reorganize_frame_images.py

### 用途

该脚本读取指定目录根部的图片，根据文件名中的最后一组数字排序，然后：

1. 把原始图片备份到 `原始图片/`。
2. 生成按序编号的 `重命名数据/000.jpg`、`001.jpg` 等文件。
3. 创建 `0001/`、`0002/` 等帧目录。
4. 在凝结帧目录中写入多个机位图片。
5. 全部复制成功后，删除输入目录根部的原始图片。

该脚本仅保存在 `utils/reorganize_frame_images.py`。

### 默认配置

默认配置文件：

```text
configs/reorganize_frame_images.json
```

主要配置项：

| 配置项 | 命令行参数 | 说明 |
| --- | --- | --- |
| `pre_frame_count` | `--pre-frame-count` | 特效凝结帧之前的普通帧数量 |
| `camera_count` | `--camera-count` | 特效机位参数；凝结帧实际写入 `camera_count + 1` 张图片 |
| `original_dir_name` | `--original-dir-name` | 原始图片备份目录名 |
| `normalized_dir_name` | `--normalized-dir-name` | 连续编号图片目录名 |
| `image_ext` | `--image-ext` | 输出图片扩展名，不重新编码图片 |
| `dry_run` | `--dry-run` / `--no-dry-run` | 是否只预览操作 |

### 基本调用

以下两种输入目录写法等价：

```powershell
python utils/reorganize_frame_images.py ./data/QP-2026-06-23-175636
```

```powershell
python utils/reorganize_frame_images.py --input_dir ./data/QP-2026-06-23-175636
```

### 手动覆盖配置

```powershell
python utils/reorganize_frame_images.py `
  --input_dir ./data/QP-2026-06-23-175636 `
  --pre-frame-count 120 `
  --camera-count 60
```

### 使用其他配置文件

```powershell
python utils/reorganize_frame_images.py `
  --config ./configs/custom_reorganize.json `
  --input_dir ./data/QP-2026-06-23-175636
```

### 预览操作

`--dry-run` 只打印复制和删除计划，不修改文件：

```powershell
python utils/reorganize_frame_images.py `
  --input_dir ./data/QP-2026-06-23-175636 `
  --dry-run
```

> 注意：正式执行会在全部复制成功后删除输入目录根部的原始图片。首次处理新数据时建议先执行 `--dry-run`。

## 2. build_spacetime_slicer.py

### 用途

该脚本读取已经整理为帧/机位目录的数据，执行人物分割、残影叠加、残影回收、多机位凝结和片尾输出。

### 默认配置

默认配置文件：

```text
configs/spacetime_slicer.json
```

必须通过配置文件或命令行提供的参数：

- `input_dir`
- `output_dir`
- `freeze_frame`

`end_frame` 可以设置为 `null` 或省略，此时自动使用输入数据的最后一帧。

### 基本调用

如果默认配置中已经设置了所需参数：

```powershell
python build_spacetime_slicer.py
```

通常只在命令行指定输入和输出目录：

```powershell
python build_spacetime_slicer.py `
  --input_dir ./data/QP-2026-06-23-175636 `
  --output_dir ./results/QP-2026-06-23-175636
```

### 手动覆盖切片参数

```powershell
python build_spacetime_slicer.py `
  --input_dir ./data/QP-2026-06-23-175636 `
  --output_dir ./results/QP-2026-06-23-175636 `
  --camera_ids 1:60 `
  --fps 30 `
  --start_frame 25 `
  --freeze_frame 120 `
  --end_frame 160 `
  --ghost_interval 5
```

### 使用其他配置文件

```powershell
python build_spacetime_slicer.py `
  --config ./configs/custom_slicer.json `
  --input_dir ./data/QP-2026-06-23-175636 `
  --output_dir ./results/QP-2026-06-23-175636
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--camera_ids 1:90` | 使用闭区间机位范围 |
| `--camera_ids 1,5,10` | 使用指定机位列表 |
| `--fps` | 输出视频帧率 |
| `--start_frame` | 残影特效开始的源帧编号，从 1 开始 |
| `--freeze_frame` | 残影恢复完成并开始多机位凝结的源帧编号 |
| `--end_frame` | 输出结束源帧；省略时使用最后一帧 |
| `--ghost_interval` | 捕获残影的帧间隔 |
| `--fade_duration_frames` | 残影回收持续帧数 |
| `--method` | 分割方法，如 `RVM`、`Hybrid`、`SAM2_BBox`、`RMBG2` |
| `--freeze_interp_mode` | 凝结机位插值方式：`rife`、`repeat` 或 `blend` |
| `--rife_uhd` | 开启 RIFE UHD 模式 |
| `--no-rife_uhd` | 覆盖配置文件并关闭 RIFE UHD 模式 |

完整参数列表可通过以下命令查看：

```powershell
python build_spacetime_slicer.py --help
```

## 3. batch_run.py

### 用途

批处理脚本依次执行：

1. 校验重组参数和切片参数。
2. 检测输入目录是否已经具有连续帧目录、普通机位图片和完整凝结机位图片。
3. 如果结构已经满足要求，则跳过数据预处理。
4. 否则调用 `utils/reorganize_frame_images.py` 整理原始图片。
5. 调用 `build_spacetime_slicer.py` 生成切片视频。

如果参数校验失败，不会开始重组源数据。如果重组失败，不会继续执行切片。

### 默认配置

批处理配置文件：

```text
configs/spacetime_slicer_batch.json
```

该文件用于指定：

- `reorganize_config`：重组脚本配置文件。
- `slicer_config`：切片脚本配置文件。
- `output_dir`：可选的默认输出目录。

### 最简调用

```powershell
python batch_run.py -s ./data/QP-2026-06-23-175636
```

`-s` 等价于 `--input_dir`。以上命令默认输出到：

```text
./results/QP-2026-06-23-175636
```

### 指定输出目录

```powershell
python batch_run.py `
  -s ./data/QP-2026-06-23-175636 `
  --output_dir ./results/custom-output
```

### 同时覆盖两个子脚本的参数

重组脚本参数由批处理脚本直接识别，其他参数会转交给切片脚本：

```powershell
python batch_run.py `
  -s ./data/QP-2026-06-23-175636 `
  --pre-frame-count 120 `
  --camera-count 60 `
  --camera_ids 1:60 `
  --fps 30 `
  --freeze_frame 120 `
  --end_frame 160
```

上例中：

- `--pre-frame-count`、`--camera-count` 覆盖重组配置。
- `--camera_ids`、`--fps`、`--freeze_frame`、`--end_frame` 覆盖切片配置。

### 分别指定子脚本配置

```powershell
python batch_run.py `
  -s ./data/QP-2026-06-23-175636 `
  --reorganize-config ./configs/custom_reorganize.json `
  --slicer-config ./configs/custom_slicer.json
```

### 指定批处理配置

```powershell
python batch_run.py `
  --config ./configs/custom_batch.json `
  -s ./data/QP-2026-06-23-175636
```

### 预览重组

```powershell
python batch_run.py `
  -s ./data/QP-2026-06-23-175636 `
  --dry-run
```

此模式只预览重组操作，不修改输入目录，也不会执行切片。

## 查看完整帮助

```powershell
python utils/reorganize_frame_images.py --help
python build_spacetime_slicer.py --help
python batch_run.py --help
```
