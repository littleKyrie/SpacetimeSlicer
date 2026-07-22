# `batch_run.py` 双路径模式改造计划

## 1. 目标

保持现有 `data_root` 为 `"data"` 等普通相对字符串时的读取、发现、跳过和输出方式不变；仅当用户配置或传入的 `data_root` 是绝对路径时，启用共享盘目录布局。

日常命令仍统一为：

```powershell
python batch_run.py --sub_dir 0717
```

两种模式的差异如下：

| 模式 | `data_root` 示例 | 数据发现目录 | 默认输出根目录 |
| --- | --- | --- | --- |
| 现有相对路径模式 | `data` | `<repo>/data/0717` | `<repo>/data/0717/Slicers` |
| 新增绝对路径模式 | `Y:/0717/关键帧` | `Y:/0717/关键帧` | `Y:/0717/风暴时刻输出` |

本次只改变批次模式的路径推导。显式传入 `-s/--input_dir` 或 `--output_dir` 的单数据集用法继续保持最高优先级，不在本次需求中改变其既有语义。

## 2. 明确的路径契约

### 2.1 相对路径模式保持不变

当配置中的原始 `data_root` 不是绝对路径，例如：

```json
{
  "data_root": "data"
}
```

执行：

```powershell
python batch_run.py --sub_dir 0717
```

仍按当前规则构建路径：

```text
batch_dir  = <repo>/data/0717
input_dir  = <repo>/data/0717/<dataset_name>
output_dir = <repo>/data/0717/Slicers/<dataset_name>
video      = <repo>/data/0717/Slicers/<dataset_name>/<dataset_name>.mp4
```

这一分支继续沿用当前数据集发现规则、排序规则、`--datasets`、`--force` 和已有结果跳过规则，避免影响项目内已有工作流。

### 2.2 绝对路径模式使用固定的共享盘布局

当配置中的原始 `data_root` 是绝对数据目录，例如：

```json
{
  "data_root": "Y:/0717/关键帧"
}
```

该绝对路径被视为一个路径模板：

```text
<固定前缀>/<日期部分>/<输入目录名>
Y:/         0717       关键帧
```

其中：

- `--sub_dir` 只替换绝对路径中 `关键帧` 的直接父目录名，即日期部分。
- 输入目录和输出目录必须使用同一个 `sub_dir` 日期值；两者不允许出现不同日期，也不能继续使用配置模板中的旧日期。
- 盘符、日期目录之前的父路径以及末级输入目录名 `关键帧` 保持不变。
- 输出目录与 `关键帧` 同级，目录名固定为 `风暴时刻输出`。
- 不再把 `sub_dir` 追加到 `data_root` 末尾，也不在绝对输入目录下创建 `Slicers`。

路径推导公式：

```text
configured_root = Y:/0717/关键帧
fixed_prefix    = configured_root.parent.parent       # Y:/
input_leaf      = configured_root.name                # 关键帧
date_dir        = fixed_prefix / sub_dir              # Y:/0717
batch_dir       = date_dir / input_leaf               # Y:/0717/关键帧
output_root     = date_dir / 风暴时刻输出              # Y:/0717/风暴时刻输出
input_dir       = batch_dir / dataset_name
output_dir      = output_root / dataset_name
video           = output_dir / f"{dataset_name}.mp4"
```

对应关系必须严格为：

```text
--sub_dir 0717
  输入：Y:/0717/关键帧/<QP目录名>
  输出：Y:/0717/风暴时刻输出/<QP目录名>/<QP目录名>.mp4

--sub_dir 0718
  输入：Y:/0718/关键帧/<QP目录名>
  输出：Y:/0718/风暴时刻输出/<QP目录名>/<QP目录名>.mp4
```

例如配置仍为 `Y:/0717/关键帧`，但执行：

```powershell
python batch_run.py --sub_dir 0718
```

则只改变日期部分：

```text
数据发现目录：Y:/0718/关键帧
输出根目录：  Y:/0718/风暴时刻输出
```

### 2.3 绝对路径模式的数据集过滤

若 `Y:/0717/关键帧` 中存在：

```text
130-2026-07-17-144135/
QPA-2026-07-17-144135/
QPB-2026-07-17-150000/
QP-2026-07-17-160000/
```

只识别名称以大写 `QP` 开头的一级子目录，即使用与 Python `name.startswith("QP")` 等价的规则。上例会选择后三个目录，并忽略 `130-...`。

过滤条件应同时满足：

- 是 `batch_dir` 的一级子目录；
- 目录名以大写 `QP` 开头；
- 不是隐藏目录。

绝对路径模式下，即使通过 `--datasets` 显式指定名称，也应应用同一个 `QP` 前缀约束，避免绕过共享盘数据目录契约。不存在、不是目录或不以 `QP` 开头的显式名称应产生清晰错误，而不是静默执行。

相对路径模式不新增 `QP` 前缀限制，保留当前兼容行为。

### 2.4 从数据集目录名自动读取特效起始帧

支持在 `QP` 数据集名称与末尾时间戳之间携带特效起始帧数。例如：

```text
QPA_75-2026-07-19-103215/
```

目录名使用以下完整结构时，解析时间戳前紧邻的十进制正整数：

```text
QP<数据标识>_<pre_frame_count>-YYYY-MM-DD-HHMMSS
```

实现时应对完整目录名进行严格匹配，等价正则可写为：

```python
r"^QP.*_(?P<frame>\d+)-(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{6})$"
```

匹配 `QPA_75-2026-07-19-103215` 后，该数据集的有效参数必须自动设置为：

```text
pre_frame_count = 75     # 等价命令行 --pre-frame-count 75
start_frame     = 1      # 等价切片参数 --start_frame 1
freeze_frame    = 75     # 等价切片参数 --freeze_frame 75
```

`start_frame=1` 表示取消普通片头，从合成时间轴第一帧开始进入切片特效。

规则约束如下：

- 只读取末尾标准时间戳之前、由下划线和连字符包围的数字，不从时间戳、数据标识或路径的其他部分猜测帧数；
- 解析结果必须大于 `0`，`_0-`、负数、空值和非数字均视为无效命名并在处理前明确报错；
- `QPA-2026-07-19-103215` 等不携带该字段的普通名称继续使用命令行、JSON 和默认参数，行为保持不变；
- 匹配后的目录名仍完整用于输入、输出目录和 MP4 命名，不删除 `_75`；
- 该规则按数据集分别解析，同一批次允许不同目录使用不同帧数，不得修改共享的全局参数后污染后续数据集；
- 参数优先级为：显式命令行参数 > 目录名元数据 > JSON 值和默认值；三个关联字段按各自是否显式传入决定是否覆盖，其他参数不受影响；
- 若只显式传入 `--pre-frame-count` 或 `--freeze_frame` 之一，未显式传入的另一个参数自动跟随该手动值；若两者均显式传入但数值不同，则在预处理前明确报错；
- 此命名规则只负责设置帧参数，不改变 `camera_count`、`camera_ids`、`end_frame` 或输出路径规则。

## 3. 输出命名与完成判定

数据集原始目录名必须完整保留，不做缩写或重命名：

```text
输入：Y:/0717/关键帧/QPA-2026-07-17-144135
输出：Y:/0717/风暴时刻输出/QPA-2026-07-17-144135/QPA-2026-07-17-144135.mp4
```

最终视频名与输出目录名保持一致：

```text
<dataset_name>.mp4
```

一个数据集仅在下列文件存在且大小大于 `0` 时视为已完成：

```text
<该模式推导出的 output_dir>/<dataset_name>.mp4
```

因此绝对路径模式检查：

```text
Y:/0717/风暴时刻输出/<dataset_name>/<dataset_name>.mp4
```

相对路径模式仍检查：

```text
<repo>/data/0717/Slicers/<dataset_name>/<dataset_name>.mp4
```

`--force` 继续只控制是否忽略该完成判定，不改变候选目录范围和输出路径。

## 4. 实现设计

### 4.1 保留 `data_root` 的原始路径类型

当前 `load_batch_config()` 会立即调用 `resolve_root_path()`，导致 `"data"` 也被转换成绝对路径。若后续只检查 `Path(args.data_root).is_absolute()`，两种模式都会被误判为绝对路径模式。

计划调整为在解析原始配置值或原始命令行值时确定模式，并显式保留该信息，例如：

```python
data_root_raw = Path(raw_value).expanduser()
data_root_mode = "absolute" if data_root_raw.is_absolute() else "relative"
data_root = resolve_root_path(data_root_raw)
```

要求：

- 模式判断必须发生在相对路径被仓库根目录补全之前；
- `--data_root` 显式值优先于 JSON 配置值，其原始值同时决定模式；
- 不通过字符串是否包含盘符手工判断，统一使用 `pathlib` 的 Windows 路径语义；
- 解析完成后可继续使用规范化绝对 `Path` 做实际 I/O，但必须另外保存模式。

由于程序运行在 Windows，共享盘示例 `Y:/0717/关键帧` 和 `Y:\\0717\\关键帧` 均应识别为绝对路径。JSON 推荐使用正斜杠，避免反斜杠转义问题。

### 4.2 集中生成批次路径上下文

避免在发现、跳过和执行阶段分别拼路径。新增一个集中解析函数，返回本次批次的输入根和输出根，例如：

```python
@dataclass(frozen=True)
class BatchPaths:
    input_root: Path
    output_root: Path
    absolute_mode: bool


def resolve_batch_paths(data_root: Path, sub_dir: str, absolute_mode: bool) -> BatchPaths:
    if not absolute_mode:
        batch_dir = data_root / sub_dir
        return BatchPaths(batch_dir, batch_dir / "Slicers", False)

    fixed_prefix = data_root.parent.parent
    date_dir = fixed_prefix / sub_dir
    return BatchPaths(
        date_dir / data_root.name,
        date_dir / "风暴时刻输出",
        True,
    )
```

实际实现可以不用 `dataclass`，但必须形成单一事实来源，确保候选发现、输出已存在检查、日志打印和真正执行使用完全相同的路径。

### 4.3 调整辅助函数职责

计划将当前只依赖输入父目录的函数：

```python
dataset_output_dir(input_dir)
```

改为显式接收输出根：

```python
def dataset_output_dir(output_root: Path, input_dir: Path) -> Path:
    return output_root / input_dir.name
```

数据发现函数改为接收已经解析好的 `input_root`，并按模式应用过滤：

```python
def discover_datasets(input_root: Path, qp_only: bool) -> list[Path]:
    ...
```

数据集时间排序继续使用当前目录名末尾的 `YYYY-MM-DD-HHMMSS` 规则；本次不改变排序行为。

### 4.4 候选选择和运行阶段

批次模式的流程调整为：

1. 从命令行或 JSON 取得原始 `data_root`，在补全路径前确定相对/绝对模式。
2. 使用 `data_root + --sub_dir + 模式` 一次性解析 `input_root` 与 `output_root`。
3. 未传 `--datasets` 时，从 `input_root` 发现候选；绝对模式仅保留 `QP*`。
4. 传了 `--datasets` 时，在 `input_root` 下构造指定候选，并在绝对模式校验 `QP*`、存在性和目录类型。
5. 对每个候选解析可选的 `_<帧数>-<时间戳>` 元数据，生成该数据集独立的有效参数；未携带时保留原参数。
6. 对每个候选检查 `output_root / dataset_name / f"{dataset_name}.mp4"`。
7. 未传 `--force` 时跳过已有非空结果；传入时保留候选。
8. 执行时将同一个已解析输出目录和该候选独立的帧参数传给切片程序，确保不会在 `run_pipeline()` 中重新覆盖路径或把上一数据集的参数带入下一数据集。

需要特别修正当前 `run_pipeline()` 的回退逻辑：批次包含多个数据集时，不能再无条件调用旧的 `dataset_output_dir(source_path)`；应为每个候选预先绑定或现场使用统一的 `output_root` 计算输出目录。

### 4.5 生成每个数据集独立的帧参数

新增纯解析函数，例如：

```python
def parse_dataset_pre_frame_count(dataset_name: str) -> int | None:
    ...
```

在 `run_single_dataset()` 调用前复制基础预处理参数和切片参数。若返回正整数 `frame`，先结合显式命令行参数计算 `effective_frame`，再只对该副本设置未被手动指定的字段：

```python
dataset_preprocess_args.pre_frame_count = effective_frame
dataset_slicer_args.start_frame = explicit_start_frame if explicitly_set else 1
dataset_slicer_args.freeze_frame = effective_frame
```

上面的伪代码表示有效结果，实际实现应根据参数是否在命令行中显式出现来决定优先级。不得直接修改循环外共享的 `args` 或 `slicer_args`。这样批量处理 `QPA_75-...` 与 `QPB_125-...` 时，两者分别使用独立参数，执行顺序不会影响结果。

## 5. 参数行为

### 只传 `--sub_dir`

```powershell
python batch_run.py --sub_dir 0717
```

- 相对模式：发现 `<repo>/data/0717` 下所有符合现有规则且尚未完成的数据集。
- 绝对模式：发现 `Y:/0717/关键帧` 下所有 `QP*` 且尚未完成的数据集。
- 按现有确定性时间顺序依次处理全部未完成数据集。

### 配合 `--datasets`

```powershell
python batch_run.py --sub_dir 0717 --datasets QPA-2026-07-17-144135
```

- 只处理指定数据集；未传 `--force` 时仍跳过已有非空结果。
- 绝对模式下指定名称必须以 `QP` 开头，并且必须是输入根下真实存在的一级目录。

### 配合 `--force`

```powershell
python batch_run.py --sub_dir 0717 --force
```

- 重跑本模式下发现的全部候选。
- 绝对模式仍不会处理非 `QP*` 目录。
- 日志应打印将重跑的数据集及其实际输出目录。

### 配合目录名帧参数

```powershell
python batch_run.py --sub_dir 0719 --datasets QPA_75-2026-07-19-103215
```

该目录自动等价应用：

```text
--pre-frame-count 75 --start_frame 1 --freeze_frame 75
```

无需在命令行重复传入这三个值。目录名没有 `_数字-时间戳` 字段时，不启用自动覆盖。

### 显式单数据集模式

```powershell
python batch_run.py -s <input_dir> --output_dir <output_dir>
```

显式路径继续优先，本次不根据绝对 `data_root` 自动改写用户显式给出的 `input_dir` 或 `output_dir`。

## 6. 错误处理与日志

以下情况应在切片开始前给出包含最终解析路径的明确错误：

- 绝对 `data_root` 缺少可替换的日期父级或末级输入目录名；
- 解析后的 `input_root` 不存在或不是目录；
- `--datasets` 指定项不存在、不是一级目录，或在绝对模式下不以 `QP` 开头；
- `--sub_dir` 为空、包含路径分隔符、为 `.`/`..` 或是绝对路径。`sub_dir` 必须只是单个日期目录名，防止它改变固定路径结构。
- 数据集名称看起来携带帧字段但值不是正整数，例如 `_0-2026-...`、`_-2026-...` 或 `_abc-2026-...`。

运行前日志至少打印：

```text
Data-root mode: relative|absolute
Input root: ...
Output root: ...
Will process N dataset(s).
```

每个数据集继续打印完整输入目录和完整输出目录，方便确认中文共享盘路径是否正确。

匹配目录名帧参数时还应打印：

```text
Dataset frame metadata: encoded=75; effective pre_frame_count=75, start_frame=1, freeze_frame=75
```

## 7. 测试计划

在 `test/test_batch_run.py` 中保留现有测试，并增加以下覆盖。

### 相对路径回归

- 原始 `data_root = "data"` 时仍判定为相对模式，即使内部规范化后是绝对路径。
- `--sub_dir 0717` 仍从 `<repo>/data/0717` 读取。
- 输出为 `<repo>/data/0717/Slicers/<dataset>/<dataset>.mp4`。
- 不新增 `QP` 限制，现有普通目录发现行为保持不变。
- 已有非空同目录名 MP4、空文件、`--force`、多数据集排序和 `--datasets` 行为均不回归。

### 绝对路径路径推导

- 配置 `Y:/0717/关键帧` 且 `--sub_dir 0717` 时，输入根为 `Y:/0717/关键帧`，输出根为 `Y:/0717/风暴时刻输出`。
- 同一配置配合 `--sub_dir 0718` 时，仅日期段变成 `0718`，其余段保持不变。
- 绝对模式不生成 `Y:/0717/关键帧/0717`，也不生成 `Y:/0717/关键帧/Slicers`。
- `--data_root` 命令行覆盖配置时，由命令行原始值决定模式和路径。

### 绝对路径过滤与完成判定

- 同时存在 `130-*`、`QPA-*`、`QP-*` 时，只发现后两者。
- 小写 `qp-*` 不匹配大写 `QP` 规则。
- `--datasets 130-*` 明确报错。
- 输出 `Y:/0717/风暴时刻输出/QPA-.../QPA-....mp4` 非空时跳过对应输入。
- 输出文件为空时仍选择处理。
- `--force` 可重跑已有结果，但仍排除非 `QP*` 目录。
- 多个候选实际传入切片程序的输出目录分别位于同一个 `风暴时刻输出` 下，且保留各自完整目录名。

### 参数安全性

- 拒绝 `--sub_dir ../0717`、`Y:/0717`、空字符串和包含 `/` 或 `\\` 的值。
- 中文路径与正斜杠/正确转义的反斜杠写法都能得到相同路径。
- 显式 `-s`、显式 `--output_dir` 的既有测试继续通过。

### 数据集名称帧参数

- `QPA_75-2026-07-19-103215` 解析为 `75`，并为该数据集设置 `pre_frame_count=75`、`start_frame=1`、`freeze_frame=75`。
- `QPA_125-2026-07-19-103215` 同理解析为 `125`。
- `QPA-2026-07-19-103215` 不触发覆盖，完整保留原参数来源和优先级。
- 不把时间戳中的 `2026`、`07` 等数字误识别为帧数。
- `_0-`、负数、空字段和非数字字段给出包含目录名的清晰错误。
- 同一批次处理 `QPA_75-...`、普通 `QPA-...` 和 `QPB_125-...` 时，各自参数互不污染。
- 输出仍为 `风暴时刻输出/QPA_75-.../QPA_75-....mp4`，完整保留 `_75`。
- 未显式传入相关参数时，目录元数据优先于 JSON 和默认值，并保证 `freeze_frame == pre_frame_count`。
- `QPG_86-... --pre-frame-count 75 --start_frame 1 --freeze_frame 75` 的最终有效值为 `75/1/75`。
- 只显式传入 `--pre-frame-count 75` 或 `--freeze_frame 75` 时，另一个关联字段也使用 `75`。
- 同时显式传入不同的 `--pre-frame-count` 与 `--freeze_frame` 时明确报错。

测试命令：

```powershell
python -m unittest discover -s test -p test_batch_run.py
```

## 8. 验收标准

使用：

```json
{
  "data_root": "Y:/0717/关键帧"
}
```

并执行：

```powershell
python batch_run.py --sub_dir 0717
```

最终必须满足：

- 只读取 `Y:/0717/关键帧` 中名称以 `QP` 开头的一级数据目录；
- 不读取 `130-2026-07-17-144135` 等非 `QP` 目录；
- 每个结果保存为 `Y:/0717/风暴时刻输出/<原始QP目录名>/<原始QP目录名>.mp4`；
- 数据集目录名和视频文件名保持一致，例如 `QPA-2026-07-18-103215/QPA-2026-07-18-103215.mp4`；
- 将命令改成 `--sub_dir 0718` 时，仅路径中的 `0717` 日期目录变为 `0718`；
- `--sub_dir 0718` 时输入必须来自 `Y:/0718/关键帧/<QP目录名>`，输出必须写入 `Y:/0718/风暴时刻输出/<QP目录名>/<QP目录名>.mp4`，输入和输出日期始终一致；
- 将 `data_root` 恢复为 `"data"` 后，所有原有项目内读取和输出路径保持不变。
