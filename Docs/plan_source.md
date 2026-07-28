# `source` 模式逐帧跟踪与残影回收修改方案

## 0. 本轮修订：真实人物始终位于残影之上

在逐帧跟踪与密集回收轨迹已经建立的基础上，`source` 的可见合成进一步满足：

1. 完整原始帧始终作为输出基础，当前人物 RGB 不由 RVM 抠图重建。
2. 当前帧 RVM Alpha 只用于构造“真实人物保护遮罩”，限制历史残影可以覆盖的区域。
3. 历史残影与当前人物重叠时，只取消重叠位置的残影 Alpha，直接保留原始帧像素。
4. 新残影在捕获瞬间若与当前人物重合，会被当前人物遮住；人物移动后，留在原位置的残影才显露。
5. 回收使用 freeze 帧保护遮罩，残影汇聚到最终人物时进入人物后方，不覆盖最终人物。
6. `patched_canvas` 保持原有行为。

目标层级为：

```text
当前原始帧人物
历史永久残影
当前原始帧背景
```

由于原始帧同时包含人物和背景，不能简单把整张原始帧最后覆盖到画布上，也不能只反转残影循环顺序。实现时对每个残影计算：

```python
live_protection = alpha_mask > live_subject_alpha_threshold
effective_ghost_alpha = (
    ghost_alpha
    * ghost_opacity
    * (1.0 - live_protection)
)
```

新增 `live_subject_protect_dilate` 参数，对二值保护遮罩做可配置膨胀，覆盖 RVM 人物边缘的低置信度像素。保护遮罩只裁减残影 Alpha，不修改原始帧 RGB，因此分割边缘不会直接造成当前人物残缺或背景替换黑边。

## 1. 目标

修改 `effect_base_mode=source` 的内部逻辑，使切片生成、正常画面输出和残影回收相互解耦：

1. 从 `start_frame` 到回收目标帧连续运行分割策略，RVM 的递归状态按连续视频帧更新。
2. 只有满足 `ghost_interval` 的帧创建可见的永久残影，并使用对应的 `ghost_opacity`。
3. 非切片帧始终保留完整原始视频帧，不使用该帧 Alpha 重建当前人物；该帧 Alpha 只限制历史残影的覆盖范围。
4. 每一帧的原图和 Alpha 作为隐藏轨迹样本保留，供回收阶段把永久残影平滑移动到凝结姿态。
5. 回收完成后再进入现有多机位冻结环绕流程。

本次只改变 `source` 模式的数据采集和回收能力，不改变：

- `ghost_interval` 对永久残影捕获位置的定义；
- `ghost_opacity_start`、`ghost_opacity_end` 的计算方式；
- `source` 模式以原始视频帧作为正常画面底图的行为；
- `patched_canvas` 的当前可见合成效果；
- RIFE 插帧、冻结环绕和片尾播放逻辑；
- 除新增人物保护遮罩膨胀参数外的现有命令行和 JSON 配置字段。

## 2. 当前实现与缺口

当前 `process_segment()` 使用：

```python
needs_alpha = should_be_ghost or effect_base_mode == 'patched_canvas'
```

因此 `source` 模式只有在 `ghost_interval` 命中时才调用 `strategy.process_frame()`，并且 `all_ghosts` 中只包含永久切片：

```text
输入帧索引       0  1  2  3  4  5  6
ghost_interval   ●        ●        ●
RVM 推理          ●        ●        ●
all_ghosts        0        1        2
permanent_indices [0, 1, 2]
```

这可以保证非切片帧不受 Alpha 影响，但有三个问题：

1. RVM 的 `rec` 状态在相邻调用之间跨过了多帧，失去连续视频推理的输入条件。
2. 回收只能在稀疏切片之间插值，动作变化较大时轨迹和姿态不够准确。
3. `last_ghost_idx` 指向最后一次切片捕获，不一定对应 `freeze_frame` 或切片阶段最后一帧，回收终点可能提前。

目标结构应改为：

```text
输入帧索引       0  1  2  3  4  5  6
ghost_interval   ●        ●        ●
RVM 推理          ●  ●  ●  ●  ●  ●  ●
all_ghosts        0  1  2  3  4  5  6
permanent_indices [0,       3,       6]
正常画面底图      原 原 原 原 原 原 原
```

其中：

- `all_ghosts` 实际承担“逐帧主体轨迹样本”的职责；
- `permanent_indices` 仍然只标记真正创建并显示的残影；
- 非永久样本绝不参与切片生成阶段的静态残影叠加，只允许被回收插值读取。

## 3. 目标数据流

### 3.1 每帧分割

在 `source` 模式的跟踪范围内，每帧执行：

```python
alpha_mask = strategy.process_frame(current_frame, frame_idx)
```

这样 RVM 的四组递归状态 `rec` 会按连续帧更新。现有 `edge_feather` 处理继续应用于每个 Alpha，因为隐藏样本在回收阶段也会参与人物范围、位置和图像插值。

### 3.2 每帧保存轨迹样本

每个被跟踪帧在 `all_ghosts` 中只保存一次：

```python
sample = {
    'frame': current_frame.copy(),
    'alpha': alpha_mask.copy(),
    'opacity': ghost_opacity if should_be_ghost else 1.0,
}
sample_idx = len(all_ghosts)
all_ghosts.append(sample)
```

若当前帧命中 `ghost_interval`，再执行：

```python
permanent_indices.append(sample_idx)
```

这项顺序必须固定，避免当前实现中“切片帧先 append、非切片帧后 append”的分支导致索引语义不统一。

建议在代码注释和局部变量中明确：

- `all_ghosts`：密集的回收轨迹样本；
- `permanent_indices`：允许在生成阶段和回收阶段显示的残影起点。

本次为降低改动范围，可保留现有公开参数名，不强制重命名整个回收接口。

### 3.3 `source` 模式的正常画面

输出帧以完整原始帧为基础，并使用当前帧 Alpha 保护真实人物：

```python
base_frame = current_frame
frame_output = compose_static_ghosts(
    base_frame,
    all_ghosts,
    permanent_indices,
    live_subject_alpha=alpha_mask,
    live_subject_alpha_threshold=live_subject_alpha_threshold,
    live_subject_protect_dilate=live_subject_protect_dilate,
)
```

`compose_static_ghosts()` 只遍历 `permanent_indices`，因此：

- 当前帧完整保留；
- 非切片帧的 Alpha 不重建当前人物，只用来裁减与当前人物重叠的残影 Alpha；
- 已创建的永久残影继续按各自 Alpha 和 `ghost_opacity` 叠加；
- 非永久的密集轨迹样本不会在前期显示。

`source` 起始帧不再应用 `initial_subject_replacement`，而是保留完整原始人物。第一张切片与当前人物重合的部分由保护遮罩挡住，人物移动后残影才显露。`initial_subject_replacement` 继续服务于 `patched_canvas`，避免 source 首帧被修补为空背景。

### 3.4 回收阶段

回收阶段继续使用现有轨迹结构，并在 `background_mode=freeze` 时使用 freeze 帧 Alpha 保护最终人物：

```text
永久残影索引 p_idx
        ↓
build_recovery_trajectories()
        ↓
从 p_idx 平滑移动到 last_ghost_idx
        ↓
interpolate_ghost() 在相邻逐帧样本间插值
        ↓
compose_recovery_frame() 按永久残影原 opacity 合成
        ↓
使用 freeze 人物保护遮罩裁减重叠位置的残影 Alpha
```

修改后 `last_ghost_idx = len(all_ghosts) - 1` 对应跟踪范围的最后一帧，而不是最后一次 `ghost_interval` 命中帧。这样每个永久残影都会沿真实逐帧主体轨迹回到最终凝结姿态。

非永久样本只提供以下信息：

- 当前帧人物 RGB；
- 当前帧 Alpha；
- 人物包围盒或质心；
- 相邻回收位置之间的插值端点。

非永久样本自身的 `opacity=1.0` 不改变永久残影透明度。`compose_recovery_frame()` 仍从 `all_ghosts[p_idx]` 读取永久残影创建时的透明度。

## 4. 帧范围与 `recovery_timing`

### 4.1 `after_freeze`

当前默认值为 `recovery_timing=after_freeze`。此时：

```text
切片生成和跟踪：start_frame ... freeze_frame（包含）
合成回收帧：    紧接其后插入
冻结环绕：      回收完成后开始
```

`process_segment()` 已经处理到 `freeze_frame`，只需保证 `source` 对该范围逐帧运行 RVM并保存样本，回收终点就会对应 `freeze_frame`。

### 4.2 `before_freeze`

`before_freeze` 会把切片生成输出提前结束：

```text
slice_end_idx = freeze_idx - fade_duration_frames
```

如果只跟踪到 `slice_end_idx`，回收终点仍不是冻结姿态。为了让两种 timing 都满足“回收到 freeze_frame 再环绕”，计划将“输出范围”和“跟踪范围”分开：

```text
输出范围：start_idx ... slice_end_idx
跟踪范围：start_idx ... freeze_idx
```

实现方式：

1. 为 `process_segment()` 增加内部参数 `tracking_end_idx`，使用 Python 右开区间。
2. 现有 `end_idx` 继续表示切片生成输出的结束位置。
3. `source` 模式由 `generate()` 传入 `tracking_end_idx=freeze_idx + 1`。
4. 当 `frame_idx >= end_idx` 时：
   - 继续读取原始帧并运行 RVM；
   - 继续把帧和 Alpha 保存为轨迹样本；
   - 不写入视频；
   - 不再创建新的永久残影；
   - 不更新 `last_effect_frame`。
5. `patched_canvas` 暂不扩展隐藏跟踪范围，保持当前行为，避免超出本次 `source` 修改范围。

这样 `before_freeze` 的回收帧虽然替代了原本的源视频时间窗，但其运动路径仍可利用该时间窗内真实的逐帧 RVM 结果。

参数边界应满足：

```python
start_idx < end_idx <= tracking_end_idx
```

当未传入 `tracking_end_idx` 时默认等于 `end_idx`，保持调用兼容性。

## 5. `process_segment()` 重构步骤

修改文件：`models/spacetime_slicer.py`

### 步骤 1：区分输出结束位置和跟踪结束位置

扩展方法签名：

```python
def process_segment(
    ...,
    tracking_end_idx=None,
):
```

进入方法后解析：

```python
if tracking_end_idx is None:
    tracking_end_idx = end_idx
```

RIFE writer 仍只使用 `start_idx` 到 `end_idx - 1`，隐藏跟踪帧不得进入 RIFE 或视频输出。

### 步骤 2：逐帧运行分割

循环上限改为 `tracking_end_idx`。对 `source` 模式的每个跟踪帧计算 Alpha；`patched_canvas` 保持其现有输出范围内逐帧计算。

建议使用明确状态：

```python
is_output_frame = i < end_idx
should_be_ghost = (
    is_output_frame
    and (i - start_idx) % ghost_interval == 0
)
```

隐藏跟踪帧不得因刚好命中 `ghost_interval` 而被登记为永久残影。

### 步骤 3：统一样本 append

完成 Alpha 和可选 `edge_feather` 后，每帧只 append 一次：

1. 记录 `sample_idx`；
2. append 当前样本；
3. 若为切片帧，将 `sample_idx` 加入 `permanent_indices`；
4. 更新 `ghost_count`。

永久残影透明度仍使用现有线性序列，数量仍按输出切片范围计算，不能把隐藏跟踪帧计入 `num_ghosts_expected`。

### 步骤 4：`source` 使用当前帧 Alpha 限制残影覆盖区域

只有 `is_output_frame` 才执行：

- 当前帧人物保护遮罩；
- `compose_static_ghosts()`；
- `stage_writer.write()`；
- `last_effect_frame` 更新；
- 渲染进度输出。

在 `source` 分支中使用当前帧 `alpha_mask` 构造保护遮罩，但不使用该遮罩提取或重建当前人物。保护遮罩只参与 `effective_ghost_alpha` 的计算，遮罩内的输出像素保持为原始帧。

### 步骤 5：保持 `patched_canvas` 行为

`patched_canvas` 仍然：

- 每个输出帧分割当前人物；
- 在切片帧把人物烧入持久 canvas；
- 使用 `live_subject_alpha_threshold` 将当前人物覆盖到 canvas 上；
- 保存逐帧轨迹样本。

重构 append 顺序后必须验证其永久索引、残影透明度、输出像素和回收结果没有变化。

## 6. `generate()` 调用调整

修改文件：`models/spacetime_slicer.py`

在调用 `process_segment()` 时计算：

```python
tracking_end_idx = (
    freeze_idx + 1
    if effect_base_mode == 'source'
    else slice_end_idx + 1
)
```

然后同时传入：

```text
end_idx          = slice_end_idx + 1
tracking_end_idx = 上述结果
```

结果：

| 模式/时序 | 可见切片生成输出 | RVM 跟踪范围 | 回收目标 |
| --- | --- | --- | --- |
| `source + after_freeze` | 到 `freeze_frame` | 到 `freeze_frame` | `freeze_frame` |
| `source + before_freeze` | 到 `slice_end_idx` | 到 `freeze_frame` | `freeze_frame` |
| `patched_canvas` | 保持现状 | 保持现状 | 保持现状 |

## 7. 文档和帮助文本

修改文件：

- `build_spacetime_slicer.py`
- `README.md`

更新 `--effect_base_mode source` 的说明，明确：

```text
source：每帧运行分割并保存回收轨迹；正常画面使用完整原始帧；
只有 ghost_interval 命中帧创建可见残影；当前帧 Alpha 只阻止残影覆盖真实人物。
```

同时说明代价：

- 相比旧 `source`，RVM 推理次数从“切片帧数量”增加到“跟踪范围帧数量”；
- `all_ghosts` 会保存逐帧 BGR 和 Alpha，显存占用主要仍由模型决定，但系统内存占用会上升到接近 `patched_canvas`；
- 好处是切片帧获得连续 RVM 状态，回收轨迹也更贴近真实动作。

新增 `live_subject_protect_dilate` JSON/CLI 字段，默认值为 `2`。批处理继续通过现有未知参数转发机制传递该字段，无需增加专用批处理分支。

## 8. 测试方案

主要修改文件：`test/test_spacetime_slicer.py`

### 8.1 更新旧的稀疏推理测试

将当前：

```python
test_segment_only_runs_segmentation_on_slice_frames
```

调整为验证新语义：

- `source` 对输出/跟踪范围内每一帧调用策略；
- 三帧输入的调用索引为 `[0, 1, 2]`；
- `all_ghosts` 长度为 `3`；
- `ghost_interval=2` 时 `permanent_indices == [0, 2]`；
- 当前人物保护区内输出像素保持为原始帧。

这同时证明“逐帧推理用于遮挡保护和回收，但不重建当前人物”。

### 8.2 非切片 Alpha 不重建当前人物

增加逐帧返回不同 Alpha 的假策略：

- 切片帧返回有效人物 Alpha；
- 非切片帧返回全零、残缺或明显误判 Alpha；
- 验证非切片输出仍以完整原始帧为底；
- 验证只有 `permanent_indices` 指向的 Alpha 被 `compose_static_ghosts()` 显示。

### 8.3 当前真实人物始终位于残影之上

构造历史残影与当前人物部分重叠的二维测试帧，验证：

- 人物保护区内输出像素与当前原始帧逐像素一致；
- 保护区之外的历史残影仍按原 opacity 显示；
- 新残影捕获瞬间与当前人物重合的区域不可见；
- 当前人物离开后，该永久残影重新显露；
- 调高 `live_subject_protect_dilate` 会扩大保护范围，但不修改原始帧 RGB。

### 8.4 密集样本用于回收

构造人物位置逐帧移动的四帧 Alpha：

- 最后一帧不命中 `ghost_interval`；
- 确认最后一帧仍进入 `all_ghosts`，但不进入 `permanent_indices`；
- 确认 `last_ghost_idx` 指向该最后一帧；
- 确认回收最终帧中的永久残影到达最后一帧人物中心，而不是停在最后一次切片位置。

### 8.5 `before_freeze` 隐藏跟踪

覆盖以下断言：

- 输出只写到 `slice_end_idx`；
- 策略调用继续到 `freeze_idx`；
- 隐藏跟踪帧进入 `all_ghosts`；
- 隐藏跟踪帧不进入 `permanent_indices`；
- 隐藏跟踪帧不写入 `FrameCollector`；
- `last_effect_frame` 仍是最后一个可见切片生成输出；
- 回收终点对应 `freeze_idx`。

### 8.6 回收阶段保护 freeze 人物

- `background_mode=freeze` 时把最终密集样本 Alpha 传入回收合成；
- 回收残影在 freeze 人物保护区内不得覆盖背景原始像素；
- 保护区以外仍保留既有回收轨迹、插值和 opacity；
- `patched_canvas` 与未传保护遮罩的直接调用保持原有结果。

### 8.7 回归测试

保留并运行现有测试：

- 残影 Alpha 与 `ghost_opacity` 的有效透明度；
- 起始帧人物修补；
- `patched_canvas` 每帧分割；
- 回收保持每个永久残影创建时的透明度；
- 回收位置对齐、质心和宽高比；
- `after_freeze` / `before_freeze` 调度；
- RIFE 切片段插帧；
- 冻结环绕三种插值模式。

建议执行：

```powershell
python -m unittest discover -s test -p "test_spacetime_slicer.py"
python -m unittest discover -s test -p "test_batch_run.py"
```

如测试环境已配置完整依赖，再执行：

```powershell
python -m unittest discover -s test
```

## 9. 验收标准

实现完成后必须同时满足：

1. `source` 在跟踪范围逐帧调用 RVM，调用顺序连续且包含最终回收目标帧。
2. 只有 `ghost_interval` 命中帧进入 `permanent_indices`。
3. 非切片帧 Alpha 不重建该帧原始人物，只限制历史残影覆盖区域。
4. 人物保护区内的输出像素与原始帧一致，保护区外的永久残影正常显示。
5. 切片生成阶段可见残影数量、捕获帧位置和透明度与修改前一致。
6. 回收阶段使用逐帧隐藏样本，并最终汇聚到 `freeze_frame` 人物姿态，且不覆盖 freeze 人物。
7. `before_freeze` 的隐藏跟踪帧不被额外写入输出视频。
8. `patched_canvas` 的现有输出保持不变。
9. 现有测试通过，新增 source 人物顶层、保护膨胀和回收保护测试通过。

## 10. 风险与控制

### 性能

`source` 将从每隔 `ghost_interval` 帧推理一次改为每帧推理一次，运行时间会明显增加。这是维持 RVM 时序状态和获取密集回收轨迹的必要成本。

### 内存

逐帧保存 BGR 与 Alpha 会增加系统内存占用。第一版沿用 `patched_canvas` 已验证的数据结构，优先保证行为正确；若长视频内存成为问题，可在后续单独改造成：

- 只保存主体包围盒裁剪；
- 将轨迹样本暂存到磁盘；
- 对非永久样本按可配置步长降采样；
- 只保存 Alpha、几何信息和必要的前景数据。

这些优化不应与本次行为修复混在同一提交中。

### 分割异常对回收的影响

新逻辑能保证非切片 Alpha 不裁切或重建当前人物，但它会影响历史残影的遮挡范围和回收轨迹。漏分可能让残影穿透当前人物，误分可能让部分残影在背景区域被隐藏；某个中间样本严重异常时，回收过程中仍可能短暂出现形状异常。第一版通过阈值和可配置膨胀控制人物保护范围，并继续沿用现有最大连通域、质心和相邻帧插值策略；后续可独立增加：

- 面积突变检测；
- 质心速度异常检测；
- 异常 Alpha 使用前后有效样本插值替代；
- 回收专用轨迹平滑。

本次验收重点是隔离正常画面与回收数据，不额外改变现有分割质量策略。

## 11. 实施顺序

1. 在 `process_segment()` 中加入输出范围/跟踪范围分离。
2. 将 `source` 改为逐帧分割并统一保存轨迹样本。
3. 保证只有输出范围内的 `ghost_interval` 帧登记为永久残影。
4. 修改 `source` 静态合成，使当前帧 Alpha 只裁减与真实人物重叠的残影。
5. source 首帧保留完整原始人物，不再应用起始人物背景修补。
6. 在 `generate()` 中让 `source` 跟踪到 `freeze_idx`，并把 freeze Alpha 作为回收保护遮罩。
7. 新增可配置的保护遮罩膨胀参数。
8. 更新和补充单元测试。
9. 运行针对性测试与完整测试。
10. 更新 CLI help 和 README。
11. 使用实际视频重点检查残影与真实人物交叠帧、最后一个非切片帧、回收起点、回收终点和冻结环绕衔接。
