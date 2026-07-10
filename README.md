# SpacetimeSlicer

SpacetimeSlicer 是一个面向多机位相机阵列素材的时空切片视频生成工具。它将单机位连续帧与某一时刻的多机位冻结帧组合为完整视频，生成主体残影、残影回收和多机位环绕等子弹时间（Bullet Time）效果。

项目主入口为 `build_spacetime_slicer.py`。

## 功能概览

生成流程由以下阶段组成：

1. **片头（Head）**：从起始机位正常播放连续帧。
2. **残影（Ghost）**：按指定间隔分割主体，并将半透明主体逐步叠加到画布。
3. **回收（Recovery）**：累积残影向冻结时刻汇聚并淡出。
4. **冻结环绕（Freeze Orbit）**：依次输出冻结时刻的多机位画面，可进行帧扩展。
5. **片尾（Tail）**：选择具有连续尾帧的机位继续播放。

## 项目结构

```text
SpacetimeSlicer/
├── setup.ps1                     # Windows 一键环境配置
├── build_spacetime_slicer.py       # 单数据集视频生成入口
├── batch_run.py                    # 数据重组与批量生成入口
├── configs/
│   ├── spacetime_slicer.json       # 视频生成默认配置
│   ├── spacetime_slicer_batch.json # 批处理默认配置
│   ├── reorganize_frame_images.json
│   └── sam2.1/                     # SAM 2.1 模型配置
├── models/                         # 分割策略、合成逻辑和 RIFE 封装
├── utils/
│   └── reorganize_frame_images.py  # 原始图片重组工具
├── checkpoints/
│   ├── sam2/
│   └── yolo/
├── third_party/
│   ├── sam2/
│   ├── RobustVideoMatting/
│   └── rife-ncnn-vulkan/
├── data/                           # 默认输入根目录
├── results/                        # 可选输出目录
└── test/                           # 测试
```

## 环境配置

### 前置条件

- Git for Windows
- [uv](https://docs.astral.sh/uv/) —— Python 包管理与虚拟环境工具
- 使用 CUDA 后端时，安装与所选 PyTorch Wheel 兼容的 NVIDIA 驱动
- 使用 RIFE 时，显卡驱动需要提供 Vulkan 支持

项目使用 Python 3.10。NVIDIA GPU 并非基本数据整理的硬性要求，但 RVM、SAM2 和 RMBG2 在 GPU 上更实用。

### 安装 uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重新打开终端，确认 uv 可用：

```powershell
uv --version
```

### 环境配置

从远程仓库克隆代码后，在项目根目录执行以下步骤：

```powershell
git clone <repository-url> SpacetimeSlicer
Set-Location .\SpacetimeSlicer
```

#### 1. 创建虚拟环境

```powershell
uv venv --python 3.10
.venv\Scripts\activate
```

#### 2. 安装 PyTorch

根据显卡驱动选择对应的 CUDA 版本。其他 PyTorch/CUDA 组合参见 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

```powershell
# CUDA 12.8（推荐）
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

#### 3. 安装 Python 依赖

```powershell
uv pip install "numpy>=2.2,<2.3" "opencv-contrib-python>=4.10,<5" "pillow>=10,<13" "ultralytics>=8.3,<9" "transformers>=4.45,<6" "hydra-core>=1.3,<2" "iopath>=0.1.10,<0.2" "huggingface-hub>=0.26" "safetensors>=0.4" "kornia>=0.7,<1" "scipy>=1.14,<2" "matplotlib>=3.9,<4" "tqdm>=4.66" "pytest>=8,<10"
```

#### 4. 获取第三方源码

```powershell
# SAM2（固定版本 —— 仅 SAM2_BBox 方法需要）
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
uv pip install -e third_party/sam2

# RobustVideoMatting（固定版本 —— RVM / Hybrid 方法需要）
git clone https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
git -C third_party/RobustVideoMatting checkout 17d1774
```

#### 5. 下载模型权重（按需）

RVM 权重由 `torch.hub` 在首次运行时自动下载到 `~/.cache/torch/hub/checkpoints/rvm_resnet50.pth`，无需手动操作。

以下模型仅在启用对应分割方法时需要：

```powershell
# SAM2.1 Large（仅 SAM2_BBox 方法）
New-Item -ItemType Directory -Force .\checkpoints\sam2 | Out-Null
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -OutFile .\checkpoints\sam2\sam2.1_hiera_large.pt

# YOLOv8n（仅 SAM2_BBox 方法）
New-Item -ItemType Directory -Force .\checkpoints\yolo | Out-Null
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -OutFile .\checkpoints\yolo\yolov8n.pt
```

#### 6. 下载 RIFE（可选）

仅在需要帧插值（`stretch_ghost > 1` 或 `freeze_interp_mode=rife` 且 `stretch_freeze > 1`）时需要。

从 [rife-ncnn-vulkan Release](https://github.com/nihui/rife-ncnn-vulkan/releases) 下载 Windows 版本，将 `rife-ncnn-vulkan.exe` 及模型目录（如 `rife-v4.6`）放入 `third_party/rife-ncnn-vulkan/`。

#### 7. 验证

```powershell
python -c "import cv2, numpy, PIL, torch, torchvision; print('Core imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python build_spacetime_slicer.py --help
python batch_run.py --help
```

#### 自动化脚本（可选）

项目同时提供 `setup.ps1`，可一键完成上述步骤。执行前需安装 uv 和 Git：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-EnvironmentName` | `.venv` | 虚拟环境目录（相对于项目根或绝对路径） |
| `-TorchBackend` | `cu128` | PyTorch Wheel 后端：`cu128`、`cu126`、`cu118` 或 `cpu` |
| `-SkipModels` | 关闭 | 不下载 SAM2、YOLO 和 RVM 权重 |
| `-SkipRife` | 关闭 | 不下载 RIFE |

```powershell
.\setup.ps1 -TorchBackend cpu
.\setup.ps1 -SkipModels -SkipRife
```

### 脚本准备的目录

```text
third_party/
├── sam2/
├── RobustVideoMatting/
└── rife-ncnn-vulkan/

checkpoints/
├── sam2/sam2.1_hiera_large.pt
└── yolo/yolov8n.pt
```

RVM 权重由脚本写入标准 Torch Hub 缓存。项目代码从 `third_party/RobustVideoMatting` 加载 RVM 源码，不依赖固定绝对路径。

只有满足以下任一条件时程序才会启动 RIFE：

- `stretch_ghost > 1`
- `freeze_interp_mode=rife` 且 `stretch_freeze > 1`

### 可选分割依赖

- `RMBG2`：由 Transformers 首次运行时从 [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) 自动下载。使用前应阅读模型页面的许可证和访问要求。
- `rembg-*`：不由自动脚本安装。当前 rembg 版本要求 Python 3.11 以上，如需使用，请参照 [rembg 官方安装说明](https://github.com/danielgatis/rembg) 建立独立兼容环境。

## 模型与下载地址

### SAM 2.1

当前 `SAM2_BBox` 代码固定使用 `sam2.1_hiera_large.pt`；其余规格可用于后续调整。

| 模型 | 保存路径 | 官方下载 |
| --- | --- | --- |
| SAM2.1 Tiny | `checkpoints/sam2/sam2.1_hiera_tiny.pt` | [下载](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) |
| SAM2.1 Small | `checkpoints/sam2/sam2.1_hiera_small.pt` | [下载](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| SAM2.1 Base Plus | `checkpoints/sam2/sam2.1_hiera_base_plus.pt` | [下载](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| SAM2.1 Large | `checkpoints/sam2/sam2.1_hiera_large.pt` | [下载](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

PowerShell 下载示例：

```powershell
New-Item -ItemType Directory -Force .\checkpoints\sam2 | Out-Null
Invoke-WebRequest `
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt `
  -OutFile .\checkpoints\sam2\sam2.1_hiera_large.pt
```

### YOLOv8

`SAM2_BBox` 当前固定读取 `checkpoints/yolo/yolov8n.pt`。Pose 与 Seg 权重当前不参与主流程，仅供测试或扩展。

| 模型 | 用途 | 官方下载 |
| --- | --- | --- |
| `yolov8n.pt` | `SAM2_BBox` 人体框检测，主流程必需 | [下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) |
| `yolov8n-pose.pt` | 姿态实验 | [下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt) |
| `yolov8s-seg.pt` | YOLO 分割实验 | [下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt) |

```powershell
New-Item -ItemType Directory -Force .\checkpoints\yolo | Out-Null
Invoke-WebRequest `
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt `
  -OutFile .\checkpoints\yolo\yolov8n.pt
```

### RVM

RVM 代码会自动下载项目当前使用的 ResNet50 权重，也可以手动下载：

- [RobustVideoMatting 官方仓库](https://github.com/PeterL1n/RobustVideoMatting)
- [rvm_resnet50.pth 直接下载](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth)

默认 Torch Hub 权重缓存位置为：

```text
%USERPROFILE%\.cache\torch\hub\checkpoints\rvm_resnet50.pth
```

## 输入数据

项目支持两种输入状态：

1. 已经整理为“帧目录/机位图片”的数据，可直接交给 `build_spacetime_slicer.py`。
2. 根目录中只有连续拍摄的原始图片，可交给 `batch_run.py` 自动重组。

### 已整理数据格式

帧目录必须使用纯数字名称；图片名称为三位机位编号。当前读取逻辑固定使用 `.jpg`：

```text
dataset/
├── 0001/
│   └── 001.jpg              # 源帧 1，起始机位
├── 0002/
│   └── 001.jpg
├── ...
├── 0125/                    # 默认冻结帧
│   ├── 001.jpg              # 机位 1
│   ├── 002.jpg              # 机位 2
│   ├── ...
│   └── 091.jpg              # 重组器默认额外生成的机位
├── 0126/
│   └── 001.jpg              # 冻结后的连续尾帧
└── ...
```

要求：

- `0001` 到结束帧的目录编号应连续。
- 起始机位必须覆盖片头、残影和冻结帧。
- `freeze_frame` 对应目录必须包含 `camera_ids` 指定的全部机位图片。
- 片尾机位必须覆盖冻结帧之后到 `end_frame` 的连续帧。
- 所有输入图片应具有相同分辨率。
- `start_frame < freeze_frame < end_frame`，三个参数均使用从 1 开始的源帧编号。

默认配置使用 `camera_ids=1:90`。重组器默认在冻结帧中写入 `camera_count + 1` 张图片，即 `001.jpg` 至 `091.jpg`；额外机位只有在显式加入 `camera_ids` 时才会参与环绕。

### 原始连续图片格式

原始数据目录根部可以包含 `.jpg`、`.jpeg`、`.png`、`.bmp` 或 `.webp`。重组器读取文件名中**最后一组数字**作为排序编号，例如：

```text
dataset/
├── capture_000000.jpg
├── capture_000001.jpg
├── capture_000002.jpg
└── ...
```

默认参数为：

- `pre_frame_count=125`
- `camera_count=90`
- 最少图片数：`125 + 90 + 1 = 216`
- 输出扩展名：`.jpg`

重组完成后会：

1. 将原图备份到 `原始图片/`。
2. 将连续编号副本写入 `重命名数据/`。
3. 创建 `0001/`、`0002/` 等帧目录。
4. 将冻结机位序列写入 `0125/001.jpg` 至 `0125/091.jpg`。
5. 全部复制成功后删除输入目录根部的原始图片。

> 首次处理新数据时，必须先运行 `--dry-run` 检查排序和目标路径，并额外保留独立备份。

## 使用 build_spacetime_slicer.py

### 配置优先级

参数优先级统一为：

```text
脚本内置默认值 < JSON 配置文件 < 命令行参数
```

默认配置文件是 `configs/spacetime_slicer.json`。其中 `input_dir`、`output_dir` 和 `freeze_frame` 必须由配置文件或命令行提供。

### 基本用法

```powershell
.venv\Scripts\activate

python build_spacetime_slicer.py `
  --input_dir .\data\QP-2026-06-25-152530 `
  --output_dir .\results\QP-2026-06-25-152530
```

默认生成：

```text
results/QP-2026-06-25-152530/slicer.mp4
```

完整示例：

```powershell
python build_spacetime_slicer.py `
  --input_dir .\data\QP-2026-06-25-152530 `
  --output_dir .\results\custom `
  --camera_ids 1:90 `
  --fps 25 `
  --start_frame 25 `
  --freeze_frame 125 `
  --end_frame 160 `
  --ghost_interval 10 `
  --ghost_opacity_start 0.2 `
  --ghost_opacity_end 1.0 `
  --fade_duration_frames 12 `
  --method RVM `
  --freeze_interp_mode rife `
  --stretch_freeze 2
```

也可以复制默认 JSON 后集中修改：

```powershell
python build_spacetime_slicer.py `
  --config .\configs\custom_slicer.json `
  --input_dir .\data\QP-2026-06-25-152530 `
  --output_dir .\results\custom
```

### 核心参数

下表默认值来自 `configs/spacetime_slicer.json`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `configs/spacetime_slicer.json` | JSON 配置文件 |
| `--input_dir` | `null` | 已整理输入目录 |
| `--output_dir` / `--output_root` | `null` | 输出目录 |
| `--camera_ids` | `1:90` | 机位闭区间，如 `1:90`；或列表，如 `1,5,10` |
| `--fps` | `25` | 输出 MP4 帧率 |
| `--start_frame` | `25` | 残影开始源帧，从 1 开始 |
| `--freeze_frame` | `125` | 残影恢复完成并开始多机位冻结环绕的源帧 |
| `--end_frame` | `null` | 输出结束源帧；为空时使用起始机位最后一帧 |
| `--method` | `RVM` | 主体分割策略 |
| `--ghost_interval` | `20` | 残影捕获间隔，必须大于等于 1 |
| `--ghost_opacity_start` | `0.2` | 第一组残影不透明度 |
| `--ghost_opacity_end` | `1.0` | 最后一组残影不透明度 |
| `--fade_duration_frames` | `12` | 残影回收逻辑帧数 |
| `--edge_feather` | `0` | Alpha 边缘羽化像素数 |

### 分割策略

| `--method` | 说明 | 依赖 |
| --- | --- | --- |
| `RVM` | Robust Video Matting，适合人物视频，速度较快 | 本地 RobustVideoMatting + RVM 权重 |
| `Hybrid` | RVM 与时序背景差分融合，尝试保留扇子等道具 | 与 RVM 相同 |
| `SAM2_BBox` | YOLOv8 人体框检测后使用 SAM2 精细分割 | `yolov8n.pt`、SAM2 Large、SAM2 包 |
| `RMBG2` | Bria RMBG-2.0 通用背景移除 | Transformers、Hugging Face 模型 |
| `rembg-<model>` | 使用 rembg 会话，例如 `rembg-u2net` | 兼容版本的 rembg 和 ONNX Runtime |

### 时间与插值参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--stretch_head` | `1` | 片头每帧重复次数 |
| `--stretch_ghost` | `1` | 残影段扩展倍数；大于 1 时使用 RIFE |
| `--stretch_fade` | `1` | 回收段每帧重复次数 |
| `--stretch_freeze` | `1` | 冻结环绕机位间扩展倍数 |
| `--stretch_tail` | `1` | 片尾每帧重复次数 |
| `--freeze_interp_mode` | `rife` | `rife`、`repeat` 或 `blend` |
| `--tail_camera_id` | 自动 | 指定片尾机位；默认优先选择具有完整尾帧的末端机位 |
| `--recovery_timing` | `after_freeze` | `after_freeze` 或 `before_freeze` |
| `--recovery_transition_frames` | `3` | 进入回收画面的过渡帧数 |
| `--rife_exe` | 自动查找 | RIFE 可执行文件路径 |
| `--rife_model_dir` | `rife-v4.6` | RIFE 模型目录 |
| `--rife_uhd` / `--no-rife_uhd` | `false` | 开启或关闭 RIFE UHD 模式 |

### 画布与主体修补参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--effect_base_mode` | `patched_canvas` | `patched_canvas` 持续合成画布；`source` 使用每个原始帧作为底图 |
| `--background_mode` | `freeze` | 回收背景：`freeze`、`median` 或 `start` |
| `--initial_canvas_mode` | `patched_start` | 初始画布：主体修补后的起始帧，或 `clean` 替换帧 |
| `--initial_subject_patch_mode` | `freeze` | 起始主体替换来源：`none`、`median`、`freeze` 或 `frame` |
| `--initial_subject_patch_frame` | 冻结帧 | `frame` 模式使用的源帧编号 |
| `--initial_patch_alpha_threshold` | `1` | 起始主体修补 Alpha 阈值，范围 0–255 |
| `--initial_patch_dilate` | `1` | 修补掩码膨胀像素数 |
| `--live_subject_alpha_threshold` | `16` | 当前主体保留阈值，范围 0–255 |
| `--live_subject_opacity` | `1.0` | 当前主体不透明度，范围 0–1 |

### 调试导出

以下命令导出指定帧的 RVM Alpha、前景和黑底预览，然后退出，不生成完整视频：

```powershell
python build_spacetime_slicer.py `
  --input_dir .\data\QP-2026-06-25-152530 `
  --output_dir .\results\debug `
  --debug_extract_frames 25,50,125 `
  --debug_extract_camera 1
```

`--debug_extract_frames` 同样支持闭区间，例如 `25:40`。

## 使用 batch_run.py

`batch_run.py` 会先检查输入是否已经满足帧目录结构：

- 已整理：跳过重组，直接运行切片。
- 未整理：调用 `utils/reorganize_frame_images.py`，成功后再调用切片。
- `--dry-run`：仅预览重组，不生成视频。

无法被 `batch_run.py` 识别的参数会转交给 `build_spacetime_slicer.py`。

### 单数据集

先预览：

```powershell
python batch_run.py `
  -s .\data\QP-2026-06-25-152530 `
  --dry-run
```

正式执行：

```powershell
python batch_run.py -s .\data\QP-2026-06-25-152530
```

单数据集默认输出到输入目录同级的 `Slicers/<数据集名>/`：

```text
data/
├── QP-2026-06-25-152530/
└── Slicers/
    └── QP-2026-06-25-152530/
        └── slicer.mp4
```

可显式指定输出：

```powershell
python batch_run.py `
  -s .\data\QP-2026-06-25-152530 `
  --output_dir .\results\custom
```

### 批量处理目录

假设数据结构为：

```text
data/
└── 0630/
    ├── QPA-2026-06-30-161131/
    ├── QPB-2026-06-30-170200/
    └── QPC-2026-06-30-181500/
```

处理 `0630` 下所有尚未生成 `slicer.mp4` 的数据集：

```powershell
python batch_run.py --sub_dir 0630
```

只处理指定数据集：

```powershell
python batch_run.py `
  --sub_dir 0630 `
  --datasets QPA-2026-06-30-161131 QPC-2026-06-30-181500
```

强制重新处理已经存在结果的数据集：

```powershell
python batch_run.py --sub_dir 0630 --force
```

批量模式输出到：

```text
data/0630/Slicers/<数据集名>/slicer.mp4
```

### 同时覆盖重组和切片参数

```powershell
python batch_run.py `
  -s .\data\QP-2026-06-25-152530 `
  --pre-frame-count 120 `
  --camera-count 60 `
  --camera_ids 1:60 `
  --fps 30 `
  --start_frame 25 `
  --freeze_frame 120 `
  --end_frame 160 `
  --method Hybrid `
  --freeze_interp_mode blend
```

其中：

- `--pre-frame-count`、`--camera-count`、`--original-dir-name`、`--normalized-dir-name`、`--image-ext` 和 `--dry-run` 由重组器处理。
- 其他未知参数（例如 `--camera_ids`、`--method`）转交给切片脚本。

### batch_run 参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 批处理 JSON，默认 `configs/spacetime_slicer_batch.json` |
| `-s` / `--input_dir` | 单数据集输入目录 |
| `--data_root` | 批处理数据根目录，默认 `data/` |
| `--sub_dir` | `data_root` 下的批次目录 |
| `--datasets` | 只处理指定数据集名称 |
| `--force` | 忽略已有 `slicer.mp4` 并重新处理 |
| `--output_dir` | 单数据集输出目录 |
| `--reorganize-config` | 重组器 JSON 配置 |
| `--slicer-config` | 切片器 JSON 配置 |
| `--pre-frame-count` | 重组前的普通帧数 |
| `--camera-count` | 重组冻结机位参数，实际写入数量为该值加 1 |
| `--image-ext` | 重组目标扩展名；文件只复制，不重新编码 |
| `--dry-run` / `--no-dry-run` | 预览或关闭预览模式 |

## 配置文件

### `configs/spacetime_slicer.json`

包含 `build_spacetime_slicer.py` 的全部默认参数。建议为不同拍摄方案复制独立配置，而不是反复修改默认文件。

### `configs/reorganize_frame_images.json`

```json
{
  "pre_frame_count": 125,
  "camera_count": 90,
  "original_dir_name": "原始图片",
  "normalized_dir_name": "重命名数据",
  "image_ext": ".jpg",
  "dry_run": false
}
```

### `configs/spacetime_slicer_batch.json`

用于连接重组配置、切片配置和数据根目录：

```json
{
  "reorganize_config": "reorganize_frame_images.json",
  "slicer_config": "spacetime_slicer.json",
  "data_root": "data",
  "output_dir": null
}
```

相对的子配置路径以批处理配置文件所在目录为基准解析；`data_root` 相对项目根目录解析。

## 测试与帮助

```powershell
python build_spacetime_slicer.py --help
python batch_run.py --help
python utils/reorganize_frame_images.py --help

python -m pytest test/test_spacetime_slicer.py -v
python -m pytest test/test_batch_run.py -v
python -m pytest test/test_reorganize_frame_images.py -v
```

## 常见问题

### 找不到 `sam2`

确认安装位置和 Python 环境：

```powershell
uv pip install -e .\third_party\sam2
python -c "import sam2; print(sam2.__path__)"
```

### RVM 报本地仓库不存在

手动补全 RobustVideoMatting 源码：

```powershell
git clone https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
git -C third_party/RobustVideoMatting checkout 17d1774
```

源码应位于 `third_party/RobustVideoMatting/`。RVM 权重在首次运行时由 `torch.hub` 自动下载，也可以手动放入 `~/.cache/torch/hub/checkpoints/rvm_resnet50.pth`。

### RIFE 无法启动

确认可执行文件和 `rife-v4.6` 模型目录完整，或显式指定：

```powershell
python build_spacetime_slicer.py `
  ... `
  --rife_exe .\third_party\rife-ncnn-vulkan\rife-ncnn-vulkan.exe `
  --rife_model_dir .\third_party\rife-ncnn-vulkan\rife-v4.6
```

### OpenCV 无法创建 `slicer.mp4`

检查输出目录权限、路径长度以及 OpenCV 的 MP4 编码支持。项目当前使用 `mp4v` 编码器，输出图片分辨率必须保持一致。

### PowerShell 显示中文乱码

README 和 JSON 文件均使用 UTF-8。旧版 Windows PowerShell 可先执行：

```powershell
chcp 65001
```

也可以显式使用 UTF-8 读取：

```powershell
Get-Content -Encoding UTF8 .\README.md
```

## 已知限制

- `third_party/sam2` 尚未配置 `.gitmodules`，应通过上述步骤手动获取固定版本。
- 自动配置脚本当前只负责 Windows 环境；其他平台需要按相同步骤手动配置。
- `models/spacetime_slicer.py` 当前固定按 `.jpg` 查找整理后的图片；重组时不建议将 `image_ext` 改为其他格式。
- 生成视频不包含音频轨道。
