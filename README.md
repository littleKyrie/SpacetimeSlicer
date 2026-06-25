# SpacetimeSlicer

时空切片视频生成工具 —— 将多机位摄像机阵列拍摄的连续帧图像，合成为带有残影冻结、多机位环绕等**子弹时间（Bullet Time）**效果的视频。

## 效果流程

| 阶段 | 描述 |
| --- | --- |
| **头部（Head）** | 从单个机位正常播放视频 |
| **残影段（Ghost）** | 主体被分割出来，以设定的帧间隔捕获为半透明残影，残影不透明度逐渐递增 |
| **回收（Recovery）** | 所有累积的残影汇聚至冻结帧的主体位置并淡出 |
| **冻结环绕（Freeze Orbit）** | 摄像机环绕静止主体旋转，在所有机位之间切换（可选 RIFE AI 帧插值实现更平滑的效果） |
| **尾部（Tail）** | 恢复单机位正常播放 |

## 目录结构

```
SpacetimeSlicer/
├── build_spacetime_slicer.py      # 主入口：生成时空切片视频
├── batch_run.py                   # 批处理入口：重组 + 切片一条龙
├── configs/                       # JSON 配置文件
│   ├── spacetime_slicer.json      # 切片器默认参数
│   ├── spacetime_slicer_batch.json # 批处理默认参数
│   └── reorganize_frame_images.json # 图像重组默认参数
├── models/                        # 核心模块
│   ├── spacetime_slicer.py        # SpacetimeSlicer 核心引擎
│   ├── seg_strategy.py            # 分割策略抽象基类
│   ├── rvm.py                     # RVM（Robust Video Matting）分割
│   ├── hybrid_rvm.py              # RVM + 背景差分混合分割
│   ├── yolo_sam2.py               # YOLOv8 + SAM2 分割
│   ├── rmbg2.py                   # Bria RMBG-2.0 分割
│   ├── rembg.py                   # rembg 库分割
│   └── rife_ncnn.py               # RIFE ncnn-vulkan 帧插值封装
├── utils/
│   └── reorganize_frame_images.py # 原始图像重组工具
├── test/                          # 单元测试与实验脚本
├── third_party/                   # 第三方依赖
│   ├── sam2/                      # SAM 2（Meta）
│   ├── rife-ncnn-vulkan/          # RIFE 帧插值二进制
│   ├── rembg/                     # rembg 库
│   └── RMBG-2.0/                  # Bria RMBG-2.0
├── data/                          # 输入数据目录
├── results/                       # 输出视频目录
├── checkpoints/                   # 模型权重（YOLO、SAM2 等）
└── Docs/                          # 项目文档
    └── scripts_usage.md           # 脚本使用详细说明
```

## 环境要求

- **Python** ≥ 3.10
- **Conda**（推荐用于环境管理）
- **GPU**：推荐使用 NVIDIA GPU（CUDA），用于深度学习分割模型和 RIFE 帧插值

### 核心依赖

| 依赖 | 用途 |
| --- | --- |
| `torch` / `torchvision` | 深度学习框架 |
| `opencv-python` | 图像处理与视频编解码 |
| `numpy` | 数值计算 |
| `Pillow` | 图像 I/O |
| `ultralytics` | YOLOv8 检测 |
| `transformers` | RMBG-2.0 模型加载 |

### 分割模型依赖（按需安装）

| 分割方法 | 额外依赖 |
| --- | --- |
| RVM | `torch.hub`（自动下载 `PeterL1n_RobustVideoMatting`） |
| Hybrid | RVM + 背景差分（无额外依赖） |
| SAM2_BBox | `third_party/sam2/`（Meta SAM 2）+ YOLOv8 权重 |
| RMBG2 | `transformers` + HuggingFace 模型 |
| rembg | `rembg` 库 |

### RIFE 帧插值

需在 `third_party/rife-ncnn-vulkan/` 下放置 [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) 可执行文件。详见下方[环境配置](#环境配置)章节。

## 快速开始

### 1. 环境配置

```powershell
# 创建 conda 环境
conda create -n spacetime python=3.10
conda activate spacetime

# 安装核心依赖
pip install torch torchvision opencv-python numpy Pillow

# 按需安装分割模型依赖
pip install ultralytics transformers rembg

# 下载 SAM2（如需使用 SAM2_BBox 方法）
git submodule update --init third_party/sam2

# 下载 RIFE ncnn-vulkan（如需帧插值）
# 从 https://github.com/nihui/rife-ncnn-vulkan/releases 下载 Windows 版本
# 将 rife-ncnn-vulkan.exe 及依赖 .dll 放入 third_party/rife-ncnn-vulkan/
```

### 2. 准备数据

输入数据为包含连续拍摄图片的目录。典型场景：90 个摄像机环绕表演者排列，每个摄像机拍摄连续帧。原始图片按拍摄顺序存放在同一目录中。

#### 方式一：批处理模式（推荐）

一条命令完成数据重组和视频生成：

```powershell
python batch_run.py -s ./data/your-dataset
```

输出视频默认保存至 `./results/your-dataset/`。

#### 方式二：分步操作

**第一步：重组图像**

```powershell
python utils/reorganize_frame_images.py ./data/your-dataset
```

重组后的目录结构：

```
data/your-dataset/
├── 原始图片/           # 原始图片备份
├── 重命名数据/          # 按序编号的标准化副本（000.jpg, 001.jpg, ...）
├── 0001/001.jpg        # 帧 1，机位 1
├── 0002/001.jpg        # 帧 2，机位 1
├── ...
├── 0125/001.jpg        # 帧 125（冻结帧），机位 1
├── 0125/002.jpg        # 帧 125，机位 2
├── ...
├── 0125/091.jpg        # 帧 125，机位 90+1
├── 0126/001.jpg        # 帧 126，机位 1
└── ...
```

**第二步：生成切片视频**

```powershell
python build_spacetime_slicer.py --input_dir ./data/your-dataset --output_dir ./results/output
```

### 3. 查看结果

输出目录中包含 `slicer.mp4`，即为最终生成的时空切片视频。

## 配置说明

### 配置优先级

所有脚本遵循统一的配置覆盖规则：

```
脚本内置默认值 < JSON 配置文件 < 命令行参数
```

只需在命令行中提供本次需要修改的参数，其余参数沿用 JSON 配置值。

### 核心参数

#### 切片参数 (`configs/spacetime_slicer.json`)

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input_dir` | str | `null` | 输入目录（必填） |
| `output_dir` | str | `null` | 输出目录（必填） |
| `camera_ids` | str | `"1:90"` | 机位范围（`1:90`）或列表（`1,5,10`） |
| `fps` | int | `25` | 输出视频帧率 |
| `start_frame` | int | `25` | 残影特效开始的源帧编号（从 1 开始） |
| `freeze_frame` | int | `125` | 冻结帧编号（残影回收完成、多机位凝结开始） |
| `end_frame` | int | `null` | 结束帧编号（`null` 表示使用最后一帧） |
| `ghost_interval` | int | `20` | 捕获残影的帧间隔 |
| `ghost_opacity_start` | float | `0.2` | 残影起始不透明度 |
| `ghost_opacity_end` | float | `1.0` | 残影结束不透明度 |
| `fade_duration_frames` | int | `12` | 残影回收持续帧数 |
| `method` | str | `"RVM"` | 分割方法 |
| `freeze_interp_mode` | str | `"rife"` | 冻结环绕插值方式 |
| `effect_base_mode` | str | `"patched_canvas"` | 效果基础模式 |
| `background_mode` | str | `"freeze"` | 背景生成模式 |
| `recovery_timing` | str | `"after_freeze"` | 残影回收时机 |
| `stretch_head` | int | `1` | 头部慢放倍数 |
| `stretch_ghost` | int | `1` | 残影段慢放倍数 |
| `stretch_fade` | int | `1` | 回收段慢放倍数 |
| `stretch_freeze` | int | `1` | 冻结环绕慢放倍数 |
| `stretch_tail` | int | `1` | 尾部慢放倍数 |
| `rife_uhd` | bool | `false` | 是否开启 RIFE UHD 模式 |
| `edge_feather` | int | `0` | 边缘羽化像素数 |

#### 分割方法 (`--method`)

| 值 | 说明 |
| --- | --- |
| `RVM` | Robust Video Matting，速度快，人体边缘效果好（默认） |
| `Hybrid` | RVM + 背景差分，可恢复被 RVM 遗漏的道具（如扇子） |
| `SAM2_BBox` | YOLOv8 框检测 + SAM2 精细分割，泛化能力好但较慢 |
| `RMBG2` | Bria AI RMBG-2.0 通用背景去除 |
| `rembg-<model>` | rembg 库各模型（如 `rembg-u2net`） |

#### 冻结环绕插值模式 (`--freeze_interp_mode`)

| 值 | 说明 |
| --- | --- |
| `rife` | 使用 RIFE AI 在机位间插值生成中间帧，效果最平滑（默认） |
| `repeat` | 保持原始帧重复，无插值 |
| `blend` | 交叉淡入淡出混合相邻机位 |

#### 重组参数 (`configs/reorganize_frame_images.json`)

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `pre_frame_count` | int | `125` | 冻结帧之前的普通帧数量 |
| `camera_count` | int | `90` | 机位数量（冻结帧写入 `camera_count + 1` 张图片） |
| `original_dir_name` | str | `"原始图片"` | 原始图片备份目录名 |
| `normalized_dir_name` | str | `"重命名数据"` | 连续编号图片目录名 |
| `image_ext` | str | `".jpg"` | 输出图片扩展名 |
| `dry_run` | bool | `false` | 是否仅预览操作（不实际修改文件） |

## 使用示例

### 基本用法

```powershell
# 批处理：一条命令完成全部流程
python batch_run.py -s ./data/QP-2026-06-25-152530

# 指定输出目录
python batch_run.py -s ./data/QP-2026-06-25-152530 --output_dir ./results/my-output

# 预览重组操作（不修改文件，不执行切片）
python batch_run.py -s ./data/QP-2026-06-25-152530 --dry-run
```

### 自定义参数

```powershell
# 批处理模式覆盖子脚本参数
python batch_run.py `
  -s ./data/QP-2026-06-25-152530 `
  --pre-frame-count 120 `
  --camera-count 60 `
  --camera_ids 1:60 `
  --fps 30 `
  --method Hybrid `
  --freeze_interp_mode blend

# 切片模式：精细控制
python build_spacetime_slicer.py `
  --input_dir ./data/QP-2026-06-25-152530 `
  --output_dir ./results/custom `
  --camera_ids 1:90 `
  --fps 25 `
  --start_frame 30 `
  --freeze_frame 120 `
  --ghost_interval 10 `
  --ghost_opacity_start 0.1 `
  --fade_duration_frames 20 `
  --method SAM2_BBox `
  --freeze_interp_mode rife `
  --stretch_freeze 2
```

### 使用自定义配置文件

```powershell
# 批处理模式
python batch_run.py `
  --config ./configs/custom_batch.json `
  -s ./data/my-dataset

# 分别指定重组和切片配置
python batch_run.py `
  -s ./data/my-dataset `
  --reorganize-config ./configs/custom_reorganize.json `
  --slicer-config ./configs/custom_slicer.json
```

### 调试模式

```powershell
# 导出中间帧（alpha 遮罩、裁剪等诊断信息）
python build_spacetime_slicer.py `
  --input_dir ./data/my-dataset `
  --output_dir ./results/debug `
  --debug_extract_frames ./results/debug/frames `
  --debug_extract_camera 1
```

### 查看完整帮助

```powershell
python utils/reorganize_frame_images.py --help
python build_spacetime_slicer.py --help
python batch_run.py --help
```

## 效果模式详解

### 效果基础模式 (`effect_base_mode`)

| 值 | 说明 |
| --- | --- |
| `source` | 仅对残影帧运行分割；其余帧保持原始画面 |
| `patched_canvas` | 每一帧都运行分割，在捕获间隔将主体累积到画布上（默认） |

### 背景模式 (`background_mode`)

| 值 | 说明 |
| --- | --- |
| `freeze` | 使用冻结帧作为背景（默认） |
| `median` | 使用时序中值构建干净背景板 |

### 残影回收时机 (`recovery_timing`)

| 值 | 说明 |
| --- | --- |
| `after_freeze` | 冻结帧之后执行残影回收（默认） |
| `before_freeze` | 冻结帧之前执行残影回收 |

### 初始主体修补模式 (`initial_subject_patch_mode`)

控制如何移除第一帧中的主体区域：

| 值 | 说明 |
| --- | --- |
| `freeze` | 用冻结帧对应区域替换（默认） |
| `median` | 用时序中值背景对应区域替换 |
| `frame` | 用指定帧（`initial_subject_patch_frame`）对应区域替换 |

## 输入数据格式

### 原始输入

一个目录，包含按拍摄顺序命名的图片文件。文件名中最后一组数字用于排序。

### 重组后结构

```
input_dir/
├── 0001/                    # 帧目录（4 位零填充编号）
│   └── 001.jpg              # 机位 1 的图片
├── 0002/
│   └── 001.jpg
├── ...
├── 0125/                    # 冻结帧目录
│   ├── 001.jpg              # 机位 1
│   ├── 002.jpg              # 机位 2
│   ├── ...
│   └── 091.jpg              # 机位 camera_count+1
├── 0126/
│   └── 001.jpg
└── ...
```

- **普通帧目录**：每个目录包含 1 张图片（单机位）
- **冻结帧目录**：包含 `camera_count + 1` 张图片（所有机位 + 1）

## 输出格式

输出目录命名格式：

```
freeze_<start_cam>_to_<end_cam>_seq<count>_s<start>_f<freeze>_e<end>_<stretch>_<mode>/
└── slicer.mp4
```

## 运行测试

```powershell
# 运行核心单元测试
python -m pytest test/test_spacetime_slicer.py -v

# 运行批处理测试
python -m pytest test/test_batch_run.py -v

# 运行重组测试
python -m pytest test/test_reorganize_frame_images.py -v

# 运行全部测试
python -m pytest test/ -v
```

## 常见问题

### Q: RIFE 帧插值不工作？

确保 `third_party/rife-ncnn-vulkan/` 目录下存在对应平台的 `rife-ncnn-vulkan` 可执行文件。可通过 `--rife_exe` 手动指定可执行文件路径，通过 `--rife_model_dir` 指定模型目录。

### Q: SAM2 分割报错？

需要初始化子模块并下载模型权重：

```powershell
git submodule update --init third_party/sam2
# 按 SAM2 官方文档将权重放入 checkpoints/sam2/
```

### Q: 如何选择分割方法？

- **RVM**：适合人物为主体、追求速度的场景
- **Hybrid**：人物持有道具（如扇子、乐器），需要更完整分割的场景
- **SAM2_BBox**：分割精度要求高、不在意速度的场景
- **RMBG2 / rembg**：通用场景，非人物主体

### Q: 首次处理新数据应该注意什么？

建议先使用 `--dry-run` 预览重组操作，确认文件映射正确后再正式执行。原始图片在重组成功后被删除前，会先备份到 `原始图片/` 目录。
