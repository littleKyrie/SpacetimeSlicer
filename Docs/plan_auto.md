# batch_run 批次发现、force 与统一输出目录计划

## 核心诉求

日常处理新数据时，不论数据在项目内还是共享盘，运行方式都保持一致：

```powershell
python batch_run.py --sub_dir 0630
```

数据根目录只通过 JSON 中的 `data_root` 手动切换：

```json
{
  "data_root": "data"
}
```

或：

```json
{
  "data_root": "X:/关键帧"
}
```

除了 `data_root` 的具体值不同，读取数据、发现新数据、构建输出目录、保存结果的逻辑完全一致。

## 当前问题

当前 `batch_run.py` 中，`-s` / `--input_dir` 只表示输入目录。没有传 `--output_dir` 时，程序用输入目录最后一级名称推导输出目录：

```python
source_path = Path(args.source_dir).expanduser()
if args.output_dir is None:
    args.output_dir = str(Path.cwd() / "results" / source_path.name)
```

所以当前：

```text
-s ./data/0630/QPA-2026-06-30-161131
=> ./results/QPA-2026-06-30-161131
```

它不会保留 `0630`，也不会检查该批次下哪些数据已经处理过。

## 目标行为

批次入口：

```powershell
python batch_run.py --sub_dir 0630
```

统一路径规则：

```text
batch_dir  = data_root / sub_dir
input_dir  = data_root / sub_dir / dataset_name
output_dir = data_root / sub_dir / Slicers / dataset_name
video      = data_root / sub_dir / Slicers / dataset_name / slicer.mp4
```

项目内示例：

```text
data_root  = <repo>/data
sub_dir    = 0630
input_dir  = <repo>/data/0630/QPA-2026-06-30-161131
output_dir = <repo>/data/0630/Slicers/QPA-2026-06-30-161131
video      = <repo>/data/0630/Slicers/QPA-2026-06-30-161131/slicer.mp4
```

共享盘示例：

```text
data_root  = X:/关键帧
sub_dir    = 0630
input_dir  = X:/关键帧/0630/QPA-2026-06-30-161131
output_dir = X:/关键帧/0630/Slicers/QPA-2026-06-30-161131
video      = X:/关键帧/0630/Slicers/QPA-2026-06-30-161131/slicer.mp4
```

不再需要 `output_near_input` 参数，也不再需要 `results_root` 分支。

## 数据集命名

不需要把 `--dataset_name` 作为日常参数。

- 数据集名默认来自实际输入目录名，例如 `QPA-2026-06-30-161131`。
- 输出目录默认直接使用这个原始名。
- 不做 `QPA-2026-06-30-161131 -> QPA-161131` 这种短名压缩。
- 如未来确实需要自定义单次输出名，可保留 `--result_name` 作为低优先级扩展，但本次计划不依赖它。

## 推荐参数

保留：

- `-s` / `--input_dir`
  - 单个数据集输入目录，优先级最高。
  - 主要用于临时处理单个目录。
- `--output_dir`
  - 单个输出目录，优先级最高。
  - 主要用于临时覆盖，日常批次模式不需要。

新增或调整：

- `--sub_dir`
  - 批次目录，例如 `0630`。
  - 日常入口只需要这个参数。
- `--data_root`
  - 可选命令行覆盖；默认从 batch JSON 读取。
  - 如果命令行未传，就使用 `configs/spacetime_slicer_batch.json` 中的 `data_root`。
- `--datasets`
  - 可选，`nargs="*"`，指定一个或多个数据集目录名。
  - 多个名称用空格分隔；PowerShell 中也可以用反引号换行。
  - 示例：`--datasets QPA-2026-06-30-161131`。
  - 示例：`--datasets QPA-2026-06-30-161131 QPB-2026-06-30-172000`。
- `--force`
  - 跳过“结果已存在”检查，强制重跑。
- `--dry-run`
  - 只打印将处理、跳过、重跑的数据集，不实际执行。

不需要新增：

- `--output_near_input`
- `--results_root`
- 日常用的 `--dataset_name`

## 配置文件建议

`configs/spacetime_slicer_batch.json` 扩展为：

```json
{
  "reorganize_config": "reorganize_frame_images.json",
  "slicer_config": "spacetime_slicer.json",
  "data_root": "data",
  "output_dir": null
}
```

项目内数据时：

```json
{
  "data_root": "data"
}
```

共享盘数据时，用户只手动改 `data_root`：

```json
{
  "data_root": "X:/关键帧"
}
```

相对路径 `data` 按仓库根目录解析为 `<repo>/data`；绝对路径或盘符路径 `X:/关键帧` 原样使用。

`configs/spacetime_slicer.json` 可以继续保留单数据集默认值，但批次模式优先使用 `--sub_dir + data_root` 发现数据，不要求每次修改其中的 `input_dir`。

## 数据集发现

在：

```text
batch_dir = data_root / sub_dir
```

下列出一级子目录，忽略输出/辅助目录：

- `Slicers`
- `results`
- `__pycache__`
- 以 `.` 开头的隐藏目录

发现结果按“数据集时间”排序：

- 优先解析 `QPA-YYYY-MM-DD-HHMMSS` 里的时间。
- 解析失败时按目录名排序。

## 输出已存在判定

一个数据集视为已完成，当且仅当：

```text
data_root / sub_dir / Slicers / dataset_name / slicer.mp4 存在，且文件大小 > 0
```

后续可选增加 `run_status.json`，但第一版用 `slicer.mp4` 足够直接。

## force 与数据集选择逻辑

参考 `LiteGSWin/batch_run.py` 的核心策略：发现所有候选目录，再根据输出是否存在和 `--force` 决定处理或跳过。

### 当前问题分析

当前实现虽然能发现 `data_root/sub_dir` 下的全部数据集，也能过滤已有有效 `slicer.mp4` 的目录，但最终使用：

```python
candidates = [unprocessed[-1]] if unprocessed else []
```

这会把全部未处理数据压缩为排序后的最后一个，因此 `0629` 下同时存在 `152057` 和 `180200` 两组新数据时，只会执行 `180200`。这与“默认一次处理该子目录下所有未处理数据”的新需求不一致。

计划调整为：

- 默认命令选中 `sub_dir` 下全部未处理数据，而不是只选最新一组。
- 保留确定性的目录排序，按排序结果依次执行，但排序不再用于限制候选数量。
- 排序使用正则 `.*?(\d{4})-(\d{2})-(\d{2})-(\d{6})$` 提取目录名末尾的 `YYYY-MM-DD-HHMMSS`，将拼接后的时间字符串按升序排列。例如 `QP-2026-06-29-152057` 先于 `QP-2026-06-29-180200`。
- 当前排序不校验日期和时间是否合法；时间字符串相同时，再按完整目录名字典序排列。
- 不能匹配上述时间格式的目录排在可匹配目录之后，并按完整目录名字典序排列。
- 已存在且大小大于 0 的 `slicer.mp4` 仍视为已完成并跳过。
- `--datasets` 用于将候选范围缩小到指定的一组或多组数据。
- `--force` 只控制是否忽略已有结果，不改变 `--datasets` 指定的候选范围。

### 1. 只传 `--sub_dir`

```powershell
python batch_run.py --sub_dir 0630
```

行为：

- 读取 JSON 中的 `data_root`。
- 发现 `data_root/0630` 下所有数据集。
- 过滤掉已经存在 `data_root/0630/Slicers/<dataset_name>/slicer.mp4` 的数据集。
- 按确定性顺序依次执行全部未完成数据集；有几组未处理，就处理几组。
- 如果没有未完成数据，打印 `All datasets already processed.` 并退出 0。

这个模式用于持续进入新数据时的日常处理，也能补齐同一目录中先前积压的未处理数据。

### 2. 传 `--sub_dir --datasets <name...>`

```powershell
python batch_run.py --sub_dir 0630 --datasets QPA-2026-06-30-161131
```

行为：

- 只检查指定的一个或多个数据集。
- 如果某个数据集已经有 `slicer.mp4`，且未传 `--force`，则跳过该数据集。
- 如果没有完成结果，则执行该数据集。
- 因此，发生意外后只想处理 `0630` 下某个尚未完成的数据目录时，使用本命令即可。

例如只处理 `QPA-2026-06-30-161131`：

```powershell
python batch_run.py --sub_dir 0630 --datasets QPA-2026-06-30-161131
```

多个数据集示例：

```powershell
python batch_run.py --sub_dir 0630 --datasets `
  QPA-2026-06-30-161131 `
  QPB-2026-06-30-172000
```

### 3. 传 `--sub_dir --force`

```powershell
python batch_run.py --sub_dir 0630 --force
```

行为：

- 发现 `data_root/0630` 下所有数据集。
- 不检查 `slicer.mp4` 是否存在。
- 重跑该批次下所有数据集。

### 4. 传 `--sub_dir --force --datasets <name...>`

```powershell
python batch_run.py --sub_dir 0630 --force --datasets QPA-2026-06-30-161131
```

行为：

- 只重跑指定的一个或多个数据集。
- 这是重跑一组或多组旧数据的推荐方式。
- 如果指定数据集已有有效 `slicer.mp4`，必须加 `--force` 才会重新执行。

多个旧数据集示例：

```powershell
python batch_run.py --sub_dir 0630 --force --datasets `
  QPA-2026-06-30-161131 `
  QPB-2026-06-30-172000 `
  QPC-2026-06-30-181500
```

### 5. 显式 `-s / --input_dir`

```powershell
python batch_run.py -s X:/关键帧/0630/QPA-2026-06-30-161131
```

行为：

- 单数据集模式。
- `sub_dir` 默认从 `input_dir.parent.name` 推导为 `0630`。
- `data_root` 默认从 `input_dir.parent.parent` 推导为 `X:/关键帧`，除非显式传入或 JSON 中已有更高优先级配置。
- 如果未传 `--output_dir`，输出到 `input_dir.parent / "Slicers" / input_dir.name`。
- 未传 `--force` 且已有 `slicer.mp4` 时跳过。

## 执行流程

建议拆成这些函数：

```python
def resolve_data_root(args, batch_config) -> Path:
    ...

def discover_datasets(data_root: Path, sub_dir: str) -> list[Path]:
    ...

def dataset_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / "Slicers" / input_dir.name

def output_already_exists(output_dir: Path) -> bool:
    video_path = output_dir / "slicer.mp4"
    return video_path.is_file() and video_path.stat().st_size > 0

def select_datasets(args, discovered: list[Path]) -> list[Path]:
    ...
```

选择逻辑伪代码：

```python
if args.input_dir:
    candidates = [Path(args.input_dir)]
elif args.datasets:
    candidates = [data_root / args.sub_dir / name for name in args.datasets]
else:
    discovered = discover_datasets(data_root, args.sub_dir)
    if args.force:
        candidates = discovered
    else:
        candidates = [
            p for p in discovered
            if not output_already_exists(dataset_output_dir(p))
        ]

if args.datasets and not args.force:
    candidates = [
        p for p in candidates
        if not output_already_exists(dataset_output_dir(p))
    ]
```

每个候选数据集执行时：

```text
source_dir = input_dir
output_dir = input_dir.parent / "Slicers" / input_dir.name
```

除非显式传了 `--output_dir`，否则不走其他输出分支。

## 简化最终视频输出目录

实际繁琐目录生成在 `models/spacetime_slicer.py` 的 `SpacetimeSlicer.generate()`：

```python
stretch_suffix = ...
run_name = f"freeze_{start_cam}_to_{end_cam}_seq..."
output_dir = os.path.join(self.output_root, run_name)
video_path = os.path.join(output_dir, "slicer.mp4")
```

按需求改为：

```python
# stretch_suffix = ...
# run_name = f"freeze_{start_cam}_to_{end_cam}_seq..."
# output_dir = os.path.join(self.output_root, run_name)
output_dir = self.output_root
video_path = os.path.join(output_dir, "slicer.mp4")
```

日志中打印最终 `video_path`。

## 示例

项目内日常处理全部未完成数据，JSON 中 `data_root = "data"`：

```powershell
python batch_run.py --sub_dir 0630
```

如果发现：

```text
<repo>/data/0630/QPA-2026-06-30-161131
<repo>/data/0630/QPB-2026-06-30-172000
```

如果两组数据都没有有效输出，则本次按顺序处理 `QPA` 和 `QPB`。如果 `QPA` 已有：

```text
<repo>/data/0630/Slicers/QPA-2026-06-30-161131/slicer.mp4
```

则跳过 `QPA`，只处理：

```text
<repo>/data/0630/QPB-2026-06-30-172000
```

输出：

```text
<repo>/data/0630/Slicers/QPB-2026-06-30-172000/slicer.mp4
```

共享盘日常处理全部未完成数据，JSON 中 `data_root = "X:/关键帧"`：

```powershell
python batch_run.py --sub_dir 0630
```

输出：

```text
X:/关键帧/0630/Slicers/<dataset_name>/slicer.mp4
```

只处理一个尚未完成的指定数据集：

```powershell
python batch_run.py --sub_dir 0630 --datasets QPA-2026-06-30-161131
```

重跑整个批次：

```powershell
python batch_run.py --sub_dir 0630 --force
```

只重跑一个旧数据集：

```powershell
python batch_run.py --sub_dir 0630 --force --datasets QPA-2026-06-30-161131
```

只重跑多个旧数据集：

```powershell
python batch_run.py --sub_dir 0630 --force --datasets `
  QPA-2026-06-30-161131 `
  QPB-2026-06-30-172000
```

## 测试计划

- `data_root = data` 且 `--sub_dir 0630`：
  - 发现 `<repo>/data/0630` 下数据集。
  - 跳过已有 `<repo>/data/0630/Slicers/<dataset_name>/slicer.mp4` 的目录。
  - 选择并按顺序处理全部未完成数据集。
  - 覆盖同一 `sub_dir` 下同时存在两组或更多未完成数据的场景。
- `data_root = X:/关键帧` 且 `--sub_dir 0630`：
  - 使用同一套逻辑构建 `X:/关键帧/0630/Slicers/<dataset_name>/slicer.mp4`。
- `--sub_dir 0630 --force`：
  - 选择该批次下所有数据集。
- `--sub_dir 0630 --datasets QPA-2026-06-30-161131`：
  - 只选择指定数据集。
  - 该数据集未完成时正常执行。
  - 如果输出已存在且未传 `--force`，跳过。
- `--sub_dir 0630 --force --datasets QPA-2026-06-30-161131`：
  - 只选择指定数据集并重跑。
- `--sub_dir 0630 --force --datasets QPA-2026-06-30-161131 QPB-2026-06-30-172000`：
  - 只选择这两个指定数据集并重跑。
- `-s X:/关键帧/0630/QPA-2026-06-30-161131`：
  - 默认输出为 `X:/关键帧/0630/Slicers/QPA-2026-06-30-161131/slicer.mp4`。
- `models/spacetime_slicer.py`：
  - 不再创建 `freeze_...` 子目录。
  - 视频直接写入 `<output_dir>/slicer.mp4`。

运行：

```powershell
python -m unittest discover -s test -p test_batch_run.py
```

## 风险与注意事项

- 共享盘路径可能包含中文，JSON 中建议写 `X:/关键帧`，避免反斜杠转义问题。
- `--sub_dir` 默认会处理全部未完成数据；如果目录中存在大量积压任务，本次运行时间会相应增加。
- 不再需要额外增加 `--all_new`，默认行为本身就是处理全部未完成数据。
- 仅处理某个未完成数据集使用 `--datasets <name>`；重新处理已有结果的数据集使用 `--force --datasets <name>`。
- `--force` 可能覆盖旧结果，执行前日志必须打印将重跑的数据集列表。
- 多次运行同一个输出目录会覆盖或复用 `slicer.mp4`，这是简化输出目录后的预期行为。
