# 0722 批处理逻辑精简 — force 永驻 + MP4 递增 + patch 参数合并

## 会话目标

1. 分析 `--force` 参数的作用路径
2. 将行为改为永远 force（不跳过已有输出的数据集），同时 MP4 自动递增命名避免覆盖
3. 合并 `--initial_subject_patch_mode` 和 `--initial_subject_patch_frame` 两个冗余参数

---

## 改动 1：永远 force 模式

### 涉及文件

[`batch_run.py`](../../batch_run.py)

### 变更内容

移除了 `--force` 的跳过逻辑，现在**总是处理所有候选数据集**：

- **数据集发现阶段**（原 L373-381）：`candidates = discovered`，不再用 `output_already_exists()` 过滤已有输出
- **逐个遍历阶段**（原 L392-394）：不再按 `--force` 跳过已有 MP4 的候选
- **`run_pipeline` 入口**（原 L551-557）：删除 `skipped_datasets` 相关的提前退出
- 删除了 `skipped` 列表和 `args.skipped_datasets` 赋值

`--force` 参数保留在 parser 中（向后兼容），但不再参与任何逻辑判断。

### 死代码说明

`output_already_exists()` 和 `dataset_video_path()` 不再被主流程调用，但保留在文件中。`dataset_video_path` 仍被测试 import。

---

## 改动 2：MP4 自动递增命名

### 涉及文件

[`models/spacetime_slicer.py`](../../models/spacetime_slicer.py) — `resolve_output_video_path()`

### 变更内容

```python
def resolve_output_video_path(output_dir):
    output_name = os.path.basename(os.path.normpath(output_dir))
    if not output_name:
        raise ValueError(...)
    base_path = os.path.join(output_dir, f"{output_name}.mp4")
    if not os.path.isfile(base_path) or os.path.getsize(base_path) == 0:
        return base_path
    index = 1
    while True:
        candidate = os.path.join(output_dir, f"{output_name}-{index}.mp4")
        if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
            return candidate
        index += 1
```

命名规则：
- 基础名 `.mp4` 不存在或为空 → 直接用基础名
- 已存在且非空 → 从 `-1` 开始递增，找到第一个不存在的编号

---

## 改动 3：`--initial_subject_patch_frame` 自动切模式

### 涉及文件

[`build_spacetime_slicer.py`](../../build_spacetime_slicer.py) — `normalize_cli_frame_args()`

### 变更内容

在函数开头加两行：

```python
if args.initial_subject_patch_frame is not None:
    args.initial_subject_patch_mode = 'frame'
```

只要传了 `--initial_subject_patch_frame`，自动将 mode 设为 `frame`，不再需要手动写 `--initial_subject_patch_mode frame`。

覆盖两条调用路径（直接 slicer 和 batch_run），因为都经过同一个 `normalize_cli_frame_args`。

向后兼容：旧写法 `--initial_subject_patch_mode frame --initial_subject_patch_frame 99` 仍然有效。

---

## 配置文件更新

[`config_effect75_orbit90.txt`](../../config_effect75_orbit90.txt) — 示例命令中移除了冗余的 `--initial_subject_patch_mode frame`。

---

## 测试

新增/更新测试：
- `test_spacetime_slicer.py`：4 个新测试（MP4 递增 × 3、patch frame 自动切模式 × 1），1 个现有测试增加断言
- `test_batch_run.py`：2 个现有测试更新断言（不再跳过已有输出的数据集）

全部 62 个测试通过。

---

## 建议技能

下一个接手此工作的 agent 可以 invoke 以下技能：
- `codegen` — 如需继续修改项目代码
- `simplify` — 如需进一步清理冗余代码（如 `output_already_exists` 死代码）
