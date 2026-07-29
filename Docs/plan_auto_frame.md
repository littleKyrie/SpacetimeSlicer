# 起始人物背景自动替补与无重叠修补方案

## 1. 结论与适用范围

### 1.1 当前两种画布模式并不都需要起始背景替补

当前代码在 `process_segment()` 中只有以下条件成立时才实际修补起始人物区域：

```python
effect_base_mode == 'patched_canvas'
```

因此两种模式的真实行为是：

| 模式 | 起始帧是否需要替补背景 | 原因 |
| --- | --- | --- |
| `patched_canvas` | 需要 | 持久画布以起始帧为基础。如果不先清除起始人物，画布中会永久保留一份不透明人物，随后再叠加半透明切片，容易形成重影或人物堆叠 |
| `source` | 当前不需要 | 每个输出帧都重新使用完整原始帧，首张残影在捕获瞬间被当前真实人物遮挡，人物移动后才显露；不存在起始人物被永久烧入持久画布的问题 |

虽然 `generate()` 当前仍会为 source 调用
`resolve_initial_subject_replacement()`，但返回结果不会进入 source 的可见合成。
本次方案应顺便把替补解析延迟到 `patched_canvas` 真正需要时，避免 source
执行无意义的 `auto` 扫描。

如果未来希望 source 在第一输出帧就显示完整的半透明切片，而不是让真实人物位于
其上，应增加独立的显式开关，不应复用或暗改当前“真实人物永远顶层”的 source
语义。本计划默认不改变 source 行为。

### 1.2 目标

在保留现有手动模式的基础上，为 `patched_canvas` 增加：

1. 修正现有 `initial_canvas_mode=clean` 被首帧局部修补覆盖的问题，并新增
   `initial_canvas_mode=frame`，把“选择完整底层画布”和“填补该画布中的人物”
   拆成两个独立步骤。
2. `auto`：从起始机位的现有时间轴中自动选择与起始人物区域重叠最少的候选帧。
3. `masked_median`：从多帧中逐像素选择未被人物遮挡的背景并取中值，避免依赖单一候选帧。
4. `color`：用可配置纯色填充人物区域，默认黑色。
5. `external`：使用视频序列之外、人工准备的专用背景图片。
6. 自动候选评分、失败回退、质量报告和调试预览。
7. 可选的局部曝光匹配和修补边缘羽化，降低替补帧与起始帧之间的接缝。

### 1.3 不改变的行为

- 默认 `initial_subject_patch_mode` 继续保持 `freeze`，避免升级后已有任务静默改变结果。
- `freeze`、`frame`、`median`、`none` 继续保留。
- `initial_subject_patch_frame` 仍使用完整原始输入序列中的 1-based 图片编号。
- `source` 保持完整原始帧、当前人物顶层和逐帧 RVM 回收轨迹。
- `ghost_interval`、残影透明度、回收和冻结环绕逻辑不变。

## 2. 当前实现与问题

### 2.1 现有替补来源

`resolve_initial_subject_replacement()` 当前支持：

```text
none    → 不替换
median  → 固定机位时间轴抽样后直接逐像素中值
freeze  → 使用 freeze_frame
frame   → 使用 source_sequence_dir 中指定的完整原始图片
```

然后 `patch_initial_subject_region()` 使用起始帧 RVM Alpha 构造二值区域：

```python
patch_mask = alpha_mask > initial_patch_alpha_threshold
patch_mask = dilate(patch_mask, initial_patch_dilate)
output = base_frame * (1 - patch_mask) + replacement * patch_mask
```

### 2.2 现有缺口

1. `freeze` 和 `frame` 都要求使用者事前或事后人工检查人物是否重叠。
2. 现有 `median` 没有利用人物 Alpha 排除被遮挡像素。如果人物在某区域停留时间较长，
   中值背景仍可能保留人体、阴影或半透明轮廓。
3. 单纯判断候选帧“整帧是否有人”不够；真正需要判断的是候选人物 Alpha 是否覆盖
   起始帧需要挖空的局部区域。
4. RVM 是有递归状态的时序模型。不能为了扫描候选帧而乱序调用当前主策略，否则会
   污染正式生成阶段的 `rec` 状态。
5. 现有修补边界是硬二值边界，候选帧光照或背景动画不一致时可能出现明显接缝。
6. `frame` 只能引用已进入 `source_sequence_dir` 的图片，无法直接引用数据序列之外的
   专用 clean plate。

### 2.3 手动选择其他完整帧作为永久画布

这个需求在设计上是合理的，而且现有参数已经表达了所需语义：

```powershell
--effect_base_mode patched_canvas `
--initial_canvas_mode clean `
--initial_subject_patch_mode frame `
--initial_subject_patch_frame 200
```

预期语义是：

1. 从 `source_sequence_dir` 读取第 200 张完整原始输入图片。
2. 不再以特效起始帧作为永久画布。
3. 直接以第 200 张图片作为整张永久画布。
4. 后续把命中的半透明残影逐步烧入该画布，再把当前帧人物合成到最上层。

它与默认局部修补方式的区别是：

| 配置 | 永久画布主体 | 指定帧的用途 |
| --- | --- | --- |
| `initial_canvas_mode=patched_start` | 特效起始帧 | 只替换起始人物 Alpha 对应的局部区域 |
| `initial_canvas_mode=clean` | `frame` 指定的完整帧 | 整张图片直接成为永久画布，不再执行起始人物局部修补 |

不过，当前代码尚未真正实现上述差异。`generate()` 虽然会在 `clean` 模式下把
`initial_subject_replacement` 设为 `generation_canvas`，但 `process_segment()` 到达
特效首帧后，只要 replacement 非空，就会再次执行：

```python
patched_start_frame = patch_initial_subject_region(
    current_frame,
    initial_subject_replacement,
    alpha_mask,
)
canvas_ghosts = patched_start_frame.copy()
```

这会把刚刚选好的整张 `clean` 画布覆盖成“起始帧 + 局部替换”，导致当前
`clean + frame` 实际不能稳定实现手动选择整张永久画布。实现阶段需要修复这一分支，
不能只补充文档或命令示例。

建议让 `process_segment()` 明确接收 `initial_canvas_mode`：

```text
patched_start
    → 在首帧用起始 Alpha 对 replacement 做局部修补，并以结果初始化 canvas_ghosts

clean
    → 直接保留 generate() 传入的 replacement 作为 canvas_ghosts，
      跳过 patch_initial_subject_region()
```

修复后，`initial_patch_alpha_threshold` 和 `initial_patch_dilate` 只影响
`patched_start` 的局部修补，不影响 `clean` 的整帧画布。

手动选择完整画布时还必须注意：

- 指定帧应与 `start_cam` 视角、裁剪和分辨率一致。完整原始序列中可能混有冻结环绕
  的其他机位图片，误选后会造成整幅背景透视错位，而不只是局部接缝。
- 优先选择人物不在残影活动区域、背景和灯光接近特效段的固定机位帧。
- 动态 LED、灯光变化、摄像机抖动或自动曝光变化会永久保留在底层画布上。
- `initial_subject_patch_frame` 仍是 `source_sequence_dir` 的 1-based 图片编号，不是
  重组后的合成时间轴帧号。
- 该方式只适用于 `patched_canvas`；`source` 每个输出帧都会重新使用当前原始帧，
  不存在可被其他完整帧替换的永久画布。

因此，这个方案可以作为自动选择功能之外的低成本手动方案，但当前代码需要先修正
`clean` 分支被局部修补覆盖的问题。

### 2.4 推荐方案：指定画布帧，再独立修补画布人物

`clean + frame` 适合使用者已经确认指定图片本身就是无人物 clean plate 的情况。如果
希望选择一张构图、灯光更合适但仍然含有人物的帧作为底层画布，再从另一帧填补其中的
人物区域，就不能继续让一个 `initial_subject_patch_frame` 同时承担“整张画布来源”
和“局部填补来源”两种职责。

建议新增：

```text
initial_canvas_mode = frame
initial_canvas_frame = <完整原始输入图片序号>
```

并继续复用现有：

```text
initial_subject_patch_mode
initial_subject_patch_frame
```

完整示例：

```powershell
--effect_base_mode patched_canvas `
--initial_canvas_mode frame `
--initial_canvas_frame 160 `
--initial_subject_patch_mode frame `
--initial_subject_patch_frame 200
```

两个编号必须具有独立语义：

| 参数 | 示例 | 用途 |
| --- | ---: | --- |
| `initial_canvas_frame` | `160` | 读取第 160 张完整原始图片，作为永久画布的主体内容和构图来源 |
| `initial_subject_patch_frame` | `200` | 读取第 200 张完整原始图片，只为第 160 张画布中的人物区域提供替换像素 |

执行流程：

```text
读取 initial_canvas_frame 指定图片
        ↓
对该画布帧本身执行 RVM
        ↓
threshold + dilate 得到画布人物修补区域
        ↓
按 initial_subject_patch_mode 取得另一张替补图片
        ↓
只替换画布人物区域
        ↓
以修补结果初始化永久 canvas_ghosts
        ↓
重置 RVM 时序状态
        ↓
从 start_frame 正常生成残影和当前人物
```

三种画布模式的职责应固定为：

| `initial_canvas_mode` | 完整底图来源 | 修补 mask 来源 | 是否再次修补 |
| --- | --- | --- | --- |
| `patched_start` | 特效起始帧 | 特效起始帧 Alpha | 是 |
| `clean` | 已选出的完整 replacement/clean plate | 无 | 否，明确假定整张图片已干净 |
| `frame` | `initial_canvas_frame` 指定图片 | 指定画布帧自身的 Alpha | 按 `initial_subject_patch_mode` 决定 |

`frame` 模式下：

- `initial_subject_patch_mode=none`：直接使用指定画布帧，不修补其中人物。
- `initial_subject_patch_mode=freeze`：用冻结帧填补指定画布帧的人物区域。
- `initial_subject_patch_mode=frame`：用 `initial_subject_patch_frame` 指定的另一帧填补。
- 后续实现 `auto/masked_median/color/external` 后，同样可以作为指定画布帧的局部
  填补来源。

例如，指定帧已经没有人物时可以使用：

```powershell
--initial_canvas_mode frame `
--initial_canvas_frame 160 `
--initial_subject_patch_mode none
```

与 `clean` 相比，`frame` 的主要价值是它会对选中画布本身进行一次明确的 RVM 检查和
可选修补，而不是假定输入已经干净。

实现约束：

1. `initial_canvas_frame` 与 `initial_subject_patch_frame` 都使用
   `source_sequence_dir` 的 1-based 完整原始图片编号。
2. 两张图片必须与正式输出分辨率一致，不得静默缩放。
3. 画布帧必须与 `start_cam` 具有相同视角、裁剪和镜头参数；原始序列可能混有环绕
   机位，程序至少应打印所选路径并输出预览，最终由使用者确认。
4. 当 patch mode 也是 `frame` 且两个编号相同时，应报错或至少输出强警告，因为用
   同一张图片填补自身人物区域不会消除人物。
5. 对任意画布帧执行 RVM 不能污染正式特效段的递归状态。应使用独立评估实例，或者
   在画布 Alpha 计算结束后调用统一的 `strategy.reset()`。
6. `initial_patch_alpha_threshold` 和 `initial_patch_dilate` 应作用于画布帧自身
   Alpha，而不是特效起始帧 Alpha。
7. 该模式只对 `patched_canvas` 生效；`source` 不使用永久底层画布。

## 3. 新模式与参数设计

### 3.1 `initial_subject_patch_mode`

扩展为：

```text
none
freeze
frame
median
auto
masked_median
color
external
```

模式含义：

| 模式 | 替补来源 | 适用场景 |
| --- | --- | --- |
| `none` | 不修补 | 已有干净画布或调试 |
| `freeze` | 冻结帧 | 兼容旧流程 |
| `frame` | 完整原始序列中的指定编号 | 人工已找到合适帧 |
| `median` | 现有普通时间中值 | 人物移动充分、背景静态 |
| `auto` | 自动选择单个最低重叠候选帧 | 固定机位、存在完整无遮挡帧 |
| `masked_median` | 多帧按 Alpha 排除人物后逐像素中值 | 没有一整帧完全干净，但不同时间能看到不同背景像素 |
| `color` | 纯色全尺寸替补帧 | 黑色舞台或容许纯色区域 |
| `external` | 数据序列之外的指定图片 | 人工准备 clean plate，可靠性最高 |

### 3.2 建议新增参数

```json
{
  "initial_canvas_mode": "patched_start",
  "initial_canvas_frame": null,
  "initial_subject_patch_mode": "freeze",
  "initial_subject_patch_color": "#000000",
  "initial_subject_patch_image": null,
  "initial_subject_auto_alpha_threshold": 16,
  "initial_subject_auto_dilate": 1,
  "initial_subject_auto_max_overlap": 0.0,
  "initial_subject_auto_sample_count": 0,
  "initial_subject_auto_fallback": "masked_median",
  "initial_subject_masked_median_min_samples": 3,
  "initial_subject_patch_feather": 0
}
```

参数语义：

| 参数 | 建议默认值 | 说明 |
| --- | ---: | --- |
| `initial_canvas_frame` | `null` | `initial_canvas_mode=frame` 使用的完整原始输入图片序号，1-based |
| `initial_subject_patch_color` | `#000000` | `color` 模式的 RGB 颜色 |
| `initial_subject_patch_image` | `null` | `external` 模式图片路径 |
| `initial_subject_auto_alpha_threshold` | `16` | 候选帧人物 Alpha 的二值阈值 |
| `initial_subject_auto_dilate` | `1` | 候选人物遮挡区域安全膨胀次数 |
| `initial_subject_auto_max_overlap` | `0.0` | 可接受的最大人物区域重叠率；`0` 表示要求二值化后完全无重叠 |
| `initial_subject_auto_sample_count` | `0` | `0` 表示扫描起始机位全部可用时间轴；正数表示等距抽样以换取速度 |
| `initial_subject_auto_fallback` | `masked_median` | 无合格单帧时使用 `masked_median`、`best`、`color` 或 `error` |
| `initial_subject_masked_median_min_samples` | `3` | 每个像素至少需要多少个无遮挡样本 |
| `initial_subject_patch_feather` | `0` | 修补边缘羽化宽度；默认 0 保持旧输出，可在新模式中建议设为 2–5 |

路径参数规则：

- `initial_subject_patch_image` 为绝对路径时直接使用。
- 相对路径相对于当前数据集 `input_dir` 解析，而不是相对于进程工作目录。
- 如果只传入 `initial_subject_patch_image`，可自动切换到 `external`，与现有
  `initial_subject_patch_frame` 自动切换到 `frame` 的行为一致。
- 同时传入 `initial_subject_patch_frame` 和 `initial_subject_patch_image` 时应报错，
  不允许静默决定优先级。

颜色规则：

- 对外统一使用 `#RRGGBB`。
- OpenCV 内部转换成 BGR。
- 首版不必支持命名颜色，避免解析歧义。

## 4. `auto` 单帧选择算法

### 4.1 候选范围

只扫描起始机位 `start_cam` 的时间轴：

```python
self.frame_paths_dict[start_cam]
```

不扫描 `source_sequence_dir` 中混合的全部原始图片，因为该目录可能包含环形相机图片，
其视角与起始机位不一致。候选帧必须满足：

1. 来自同一机位。
2. 分辨率与起始帧一致。
3. 排除起始帧本身。
4. 文件可正常读取。
5. 如果使用抽样，必须固定排序并保证结果可复现。

freeze 帧、起始帧之前的固定机位帧和 freeze 之后仍属于起始机位的尾帧都可以成为候选。

### 4.2 起始目标区域

先对起始特效帧计算 RVM Alpha：

```python
start_alpha = segment(start_frame)
target_mask = start_alpha > initial_patch_alpha_threshold
target_mask = dilate(target_mask, initial_patch_dilate)
```

`target_mask` 表示真正需要从其他背景来源替换的区域。后续所有候选重叠评分都只关注
这个区域，而不是整帧。

### 4.3 候选人物区域

每个候选帧计算：

```python
candidate_mask = candidate_alpha > initial_subject_auto_alpha_threshold
candidate_mask = dilate(candidate_mask, initial_subject_auto_dilate)
```

重叠像素和重叠率：

```python
overlap_pixels = count_nonzero(target_mask & candidate_mask)
overlap_ratio = overlap_pixels / count_nonzero(target_mask)
```

当二值化后的 `overlap_pixels == 0` 时，该候选满足“人物完全不覆盖起始空洞”的要求。

不能仅使用两帧 RGB 差异判断人物重叠，因为相似衣服、黑色背景、灯光变化和运动模糊
都可能产生误判。人物 Alpha 重叠应作为第一优先级。

### 4.4 多个零重叠候选如何选择

不建议找到第一张零重叠帧就立即停止。多个零重叠候选中仍可能存在：

- 舞台灯光变化；
- LED 屏内容变化；
- 摄像机轻微抖动；
- 观众、影子、反射或道具变化。

建议构造起始目标区域外沿的局部环带：

```text
ring = dilate(target_mask, ring_width) - target_mask
```

在排除候选人物区域后，计算候选帧与起始帧在环带中的：

- 归一化平均绝对差；
- 可选 SSIM；
- 可选边缘梯度差。

选择顺序采用字典序，避免人为权重导致“光照更像但人物重叠更多”的错误：

```text
1. 是否达到 max_overlap
2. overlap_ratio
3. ring appearance difference
4. 与 start_frame 的时间距离
5. 帧索引
```

具体规则：

1. 如果存在一个或多个 `overlap_ratio <= max_overlap` 的候选，只在这些候选中选择
   环带外观最接近者。
2. 如果不存在合格候选，不得静默把某个明显重叠帧当成干净背景。
3. 按 `initial_subject_auto_fallback` 执行回退。

### 4.5 自动选择结果必须可审计

日志至少输出：

```text
Auto initial patch:
  scanned candidates: 146
  selected frame: 93 (1-based)
  overlap pixels: 0
  overlap ratio: 0.000000
  ring difference: 0.027
```

在输出目录增加：

```text
debug_initial_patch/
  start_source.jpg
  start_alpha.png
  target_mask.png
  selected_source.jpg
  selected_alpha.png
  overlap_visualization.png
  patched_preview.jpg
  candidate_scores.json
```

`candidate_scores.json` 至少保存前 5 个候选以及所有零重叠候选，便于使用者在人为检查后
改用现有 `frame` 模式精确覆盖自动结果。

## 5. RVM 时序状态与性能处理

### 5.1 不能直接复用主推理状态乱序扫描

RVM 和 Hybrid 都维护：

```python
self.rec = [None] * 4
```

如果在正式输出前用同一个实例跳着处理候选帧，主流程的递归状态会被候选扫描污染，
后续 Alpha 可能异常。

### 5.2 增加统一重置接口

为 `SegmentationStrategy` 增加：

```python
def reset(self):
    pass
```

RVM/Hybrid 实现：

```python
def reset(self):
    self.rec = [None] * 4
```

stateless 策略使用默认空实现。auto 预扫描应：

1. 按起始机位的真实时间顺序处理候选范围。
2. 计算起始 Alpha、候选 Alpha 和评分。
3. 只保留最佳候选 RGB、必要 ROI 数据和评分，不长期保存所有 4K Alpha。
4. 扫描结束后调用 `strategy.reset()`。
5. 再进入现有 `process_segment()`，保证正式生成从干净状态开始。

如果某策略无法安全 reset，首版应明确报错，或者由 `create_strategy()` 提供独立的
auto 评估实例；不能默默继续使用被污染状态。

### 5.3 性能取舍

全量 auto 扫描会增加一次起始机位分割预遍历，RVM 推理时间接近增加一倍，但不需要
把全部 4K RGB/Alpha 常驻内存。

优化顺序：

1. 首版优先保证选择正确，默认 `sample_count=0` 全量扫描。
2. 用户可设置 `sample_count=31` 等距抽样。
3. 后续可以缓存候选 Alpha 到磁盘，缓存键包含：
   - 文件路径与修改时间；
   - 分割 method；
   - 模型版本；
   - downsample ratio；
   - 分辨率。
4. `masked_median` 只缓存目标人物包围盒加羽化边距的 ROI，不缓存完整帧栈。

## 6. `masked_median`：比单帧 auto 更稳健的背景重建

### 6.1 原理

单帧 auto 要求存在一张候选帧，使整个起始人物区域都没有人物。如果人物始终在附近，
这个条件可能永远不成立。

`masked_median` 对目标区域中的每一个像素独立处理：

```python
valid_i(x, y) = candidate_alpha_i(x, y) <= auto_alpha_threshold
replacement(x, y) = median(
    candidate_rgb_i(x, y)
    for every valid_i(x, y)
)
```

这样即使没有任何一帧完整干净，只要不同时间能看到不同的背景位置，就可以组合出
不含人物的局部背景。

### 6.2 内存控制

只处理：

```text
bbox(target_mask) + feather/ring margin
```

对最多 N 个候选保存 ROI RGB 和 ROI 有效掩码。不要堆叠整张 4K 图片。

### 6.3 有效样本不足

每个像素统计有效样本数：

```python
valid_count(x, y)
```

当小于 `initial_subject_masked_median_min_samples` 时：

1. 优先使用零重叠或最低重叠单帧的对应像素；
2. 其次使用相邻已重建像素做小范围空间 inpaint；
3. 仍无法恢复时按 fallback 使用纯色或报错；
4. 在 debug 图中把不足区域标红，不能静默隐藏。

### 6.4 背景变化

对于动态 LED、灯光和反射，普通时间中值可能产生混合图案。可以在进入中值前：

1. 使用目标区域外环带估计每个候选帧到起始帧的逐通道增益/偏置；
2. 对候选 ROI 做局部亮度和颜色匹配；
3. 优先使用时间上接近起始帧且无遮挡的样本；
4. 可选使用加权中值而不是普通中值。

首版可先实现 Alpha 排除与普通中值，把曝光匹配作为第二阶段。

## 7. `color` 纯色填充

实现方式：

```python
replacement_frame = np.full_like(start_frame, bgr_color)
```

然后继续复用现有 `patch_initial_subject_region()`，只替换人物区域。

注意：

- 黑色舞台背景下，`#000000` 可以快速避免人物堆叠。
- 如果人物背后有灯带、文字、地面反射或渐变，纯黑会形成明显的人形黑洞。
- `color` 适合作为显式模式或 auto 失败后的可控回退，不应被当作通用背景恢复。
- `initial_canvas_mode=clean` 会使整个初始画布成为纯色，风险很高；首版应警告，
  或要求 color 模式配合 `patched_start`。

## 8. `external` 专用背景图片

### 8.1 这是最可靠的生产方案

用户可以提前准备一张：

- 来自相同固定机位；
- 与源视频完全相同分辨率；
- 构图和镜头参数一致；
- 起始人物区域没有人物或道具；
- 曝光和 LED 背景尽量接近；

的 clean plate，然后执行：

```powershell
--initial_subject_patch_mode external `
--initial_subject_patch_image clean_plate.png
```

相对路径 `clean_plate.png` 从每个数据集目录解析，因此批处理时每个数据集都可以放置
自己的同名 clean plate，无需把它加入 `重命名数据` 或伪造帧编号。

### 8.2 校验

外部图片必须：

1. 存在且是支持的图片格式；
2. 能被 OpenCV 正确读取；
3. 宽高与起始帧完全一致；
4. 至少包含 3 个颜色通道。

首版不要静默缩放或拉伸。尺寸不一致通常意味着视角或裁剪不一致，强行缩放会产生
错位，应直接报错。

可选后续增加：

```text
initial_subject_patch_align = none | ecc
```

用起始人物区域之外的背景估计 ECC 仿射或单应变换，再把 clean plate 对齐到起始帧。

## 9. 修补边缘与局部颜色一致性

### 9.1 遮挡正确不等于接缝自然

即使候选人物重叠为 0，候选帧与起始帧仍可能因灯光、压缩和背景动画不同而出现边缘。

建议新增 `initial_subject_patch_feather`：

1. 先按现有阈值和膨胀得到保证覆盖人物的硬核心遮罩。
2. 在硬遮罩外侧生成距离变换或高斯衰减。
3. 核心区域替补权重保持 1，避免原人物重新漏出。
4. 外沿在若干像素内从 1 平滑过渡到 0。

不能直接对未膨胀的原始 Alpha 做普通模糊，否则可能把原人物半透明边缘带回画布。

### 9.2 局部颜色匹配

在目标遮罩外环带拟合候选帧到起始帧的逐通道线性变换：

```text
start ≈ gain * candidate + bias
```

只对替补区域应用校正。应限制 gain/bias 范围，避免少量高亮 LED 导致整体失真。

第二阶段可以考虑多频段融合或 Poisson blending，但对动态 LED 图案未必优于简单羽化，
需要以实际数据评估。

## 10. 推荐的自动回退策略

为避免“自动化后仍静默产生人物堆叠”，建议 `auto` 的默认流程为：

```text
扫描固定机位候选帧
        ↓
存在 overlap <= max_overlap 的候选？
        ├─ 是：选择环带外观最接近的候选
        └─ 否：执行 masked_median
                    ↓
             所有目标像素样本充足？
                    ├─ 是：使用重建背景
                    └─ 否：按配置 error / color / best
```

建议默认：

```text
initial_subject_auto_fallback = masked_median
masked_median 最终失败 = error
```

这样系统不会在没有可靠背景时悄悄选择一个仍有人物的候选。需要“无论如何都生成”的
任务可以显式选择：

```text
best   → 选择最低重叠单帧并输出警告
color  → 使用纯色
```

## 11. 代码改动位置

### 11.1 `models/seg_strategy.py`

- 增加默认 `reset()` 接口。
- 可增加 `supports_temporal_reset` 属性供 auto 校验。

### 11.2 `models/rvm.py` 与 `models/hybrid_rvm.py`

- 实现 `reset()`，清空四组递归状态。

### 11.3 `models/spacetime_slicer.py`

新增或拆分：

```python
resolve_initial_subject_replacement(...)
parse_patch_color(...)
resolve_external_patch_image(...)
build_initial_target_mask(...)
scan_initial_patch_candidates(...)
score_initial_patch_candidate(...)
build_masked_temporal_background(...)
match_patch_exposure(...)
build_feathered_patch_alpha(...)
save_initial_patch_debug(...)
```

调整 `generate()`：

1. 仅在 `patched_canvas` 中解析起始替补背景。
2. 增加 `initial_canvas_frame` 解析；`initial_canvas_mode=frame` 时读取指定完整原始
   图片，并对该画布帧自身计算 Alpha。
3. 把 `initial_canvas_mode` 明确传给 `process_segment()`：
   - `patched_start` 对特效起始帧执行局部修补；
   - `clean` 直接保留完整 replacement，禁止首帧再次覆盖；
   - `frame` 直接保留预先修补完成的指定画布帧，禁止误用特效起始帧 Alpha。
4. 把“读取画布底图”“取得填补来源”“按画布 Alpha 修补”拆成独立函数，避免
   `initial_canvas_frame` 与 `initial_subject_patch_frame` 再次共用一个变量。
5. `auto`/`masked_median` 在创建正式画布前执行预扫描。
6. 所有画布预分割和候选预扫描结束后 reset 主分割策略。
7. 把最终画布及选择结果传入现有 `process_segment()`。
8. `source` 跳过画布和替补解析并打印明确日志。

### 11.4 `build_spacetime_slicer.py`

- 把 `initial_canvas_mode` choices 扩展为 `patched_start/clean/frame`。
- 增加 `--initial_canvas_frame`，使用与 `initial_subject_patch_frame` 相同的 1-based
  完整原始图片编号规则。
- 当设置 `initial_canvas_frame` 时自动切换为 `initial_canvas_mode=frame`；如果命令行
  同时显式指定冲突模式则报错，不静默决定优先级。
- 增加新 CLI 参数和 JSON 校验。
- 处理 `initial_subject_patch_image` 自动切换。
- 检测 frame/image 参数冲突。
- debug extraction 增加 auto 选择结果和实际 patch alpha。

### 11.5 `batch_run.py`

批处理参数当前会把未知切片器参数继续转发，因此多数新参数不需要专用分支。
需要确认相对 `initial_subject_patch_image` 最终由每个 dataset 的 `input_dir` 解析，
不能在 batch 根目录提前转成错误绝对路径。

### 11.6 配置与文档

- `configs/spacetime_slicer.json`
- `README.md`
- 本计划文档

默认 mode 保持 `freeze`；新字段加入明确默认值。

## 12. 测试方案

### 12.1 模式兼容

- `patched_start` 下现有 `none/freeze/frame/median` 输出与当前一致。
- `clean` 下修正现有错误覆盖行为，按参数原有设计使用完整 replacement。
- `initial_canvas_mode=frame` 使用 `initial_canvas_frame` 指定图片作为完整底图。
- `frame` 画布的修补 mask 来自画布帧自身 Alpha，不得错误复用特效起始帧 Alpha。
- 默认配置仍为 `freeze`。
- source 不调用 auto 扫描、不读取 external 图片、不应用 color patch。
- patched_canvas 继续在第一张残影烧入画布前修补起始人物。
- `patched_canvas + patched_start + frame` 只用指定帧修补起始人物局部区域。
- `patched_canvas + clean + frame` 使用指定帧的完整图片作为永久画布，首帧不得再次
  把画布覆盖为局部修补后的起始帧。
- `clean + frame` 下修改 `initial_patch_alpha_threshold` 或
  `initial_patch_dilate` 不应改变输出。

### 12.1.1 指定画布帧与独立修补帧

- 画布帧 160 含人物、修补帧 200 对应区域无遮挡时，输出画布保留第 160 帧背景，
  但人物区域像素来自第 200 帧。
- 交换 160 和 200 后结果应随两者职责改变，证明两个参数没有混用。
- `initial_subject_patch_mode=none` 时，指定画布帧保持逐像素不变。
- 修改 `initial_patch_alpha_threshold` 或 `initial_patch_dilate` 时，只改变指定
  画布帧被替换的局部范围。
- `initial_canvas_frame` 越界、文件不可读或分辨率不一致时给出清晰错误。
- `initial_canvas_frame == initial_subject_patch_frame` 且 patch mode 为 `frame` 时，
  按最终设计报错或输出可测试的明确警告。
- 完成画布帧 RVM 后必须 reset；正式特效首帧 Alpha 应与未启用 frame 画布时一致，
  证明 RVM 递归状态未被污染。
- `source + initial_canvas_mode=frame` 不读取指定画布帧、不额外执行 RVM，并明确提示
  该参数不参与 source 合成。

### 12.2 auto 评分

构造小尺寸 Alpha：

- 起始人物位于中间区域。
- 候选 A 重叠 50%。
- 候选 B 重叠 10%。
- 候选 C 重叠 0%。
- 验证选择 C。

多个 0% 候选时：

- 一个环带光照差异大；
- 一个环带更接近起始帧；
- 验证选择后者。

没有 0% 候选时：

- 验证 `max_overlap` 生效；
- 验证 `best/masked_median/color/error` 各自行为；
- 验证不会静默把超限候选当成合格帧。

### 12.3 RVM 状态

- auto 预扫描按时间顺序调用。
- 预扫描结束调用一次 `reset()`。
- 正式 `process_segment()` 的首个调用仍从 effect start 开始。
- auto 扫描不得让正式分割结果依赖最后一个候选帧。
- source 模式不产生额外预扫描调用。

### 12.4 masked median

- 不存在整帧完全干净，但每个目标像素至少在三帧中无遮挡时，可恢复无人物背景。
- 人物 Alpha 对应像素不会进入中值。
- 有效样本不足时产生明确 fallback 或错误。
- 只处理目标 ROI，结果正确贴回原分辨率。

### 12.5 color

- `#000000` 转成 BGR `(0, 0, 0)`。
- `#12A0FF` 通道顺序正确。
- 非法颜色报错。
- 只修改 patch mask 内区域。
- `clean + color` 输出警告或按设计拒绝。

### 12.6 external

- 绝对路径正常。
- 相对路径按 dataset 输入目录解析。
- 中文路径可读取。
- 文件缺失、格式错误、尺寸不匹配时报清晰错误。
- 同时设置 frame/image 参数时报错。
- 外部图片无需出现在 `source_sequence_dir`。

### 12.7 羽化和颜色匹配

- 人物硬核心区域 replacement 权重始终为 1。
- 羽化只发生在核心外沿。
- feather=0 与旧二值修补逐像素一致。
- 局部颜色匹配不会修改 mask 之外的起始帧。

### 12.8 批处理

- 多个 dataset 各自解析自己的相对 clean plate。
- auto 日志和 debug 输出进入各自输出目录。
- 已整理输入跳过重组时仍能找到 source timeline 和外部图片。

## 13. 验收标准

1. `patched_canvas + auto` 能自动选择起始人物区域重叠最小的同机位候选。
2. 存在二值重叠为 0 的候选时，最终选择的候选也必须为 0；不得因外观评分选择有人物的帧。
3. 多个 0 重叠候选中优先选择局部背景更接近起始帧者。
4. 无合格单帧时按显式 fallback 执行，不静默产生人物堆叠。
5. `masked_median` 能组合不同时间的无遮挡像素，且报告无法恢复的区域。
6. `external` 可以引用数据序列之外的同尺寸 clean plate。
7. `color` 默认黑色且只覆盖起始人物修补区域。
8. auto 预扫描不会污染 RVM/Hybrid 正式时序状态。
9. source 当前行为不变，不执行不必要的替补扫描。
10. 手动 `freeze/frame` 仍可用于人工修正自动结果。
11. 调试输出能明确说明选择了哪一帧、重叠率多少以及为何回退。
12. `patched_canvas + clean + frame` 可以把手动指定的完整输入帧作为永久画布，
    且不会被特效起始帧的局部修补结果覆盖。
13. `patched_canvas + initial_canvas_mode=frame` 可以分别指定完整画布帧和人物区域
    填补帧；修补 mask 来自画布帧自身 RVM Alpha。
14. 画布预分割不会污染正式特效段的 RVM/Hybrid 时序状态。

## 14. 推荐实施顺序

### 第一阶段：低风险功能

1. 修复 `clean` 画布被首帧局部修补覆盖的问题，并验证 `clean + frame`。
2. 新增 `initial_canvas_mode=frame` 和 `initial_canvas_frame`，实现画布来源与局部
   填补来源解耦。
3. 增加画布帧自身 Alpha 计算、修补和 RVM reset。
4. 把替补解析限定到 patched_canvas。
5. 增加 `color` 和 `external`。
6. 增加尺寸、路径和参数冲突校验。
7. 增加 debug 预览。

### 第二阶段：auto 单帧

1. 增加策略 reset 接口。
2. 实现同机位全量顺序预扫描。
3. 实现 Alpha 重叠率和环带外观评分。
4. 实现 fallback 与候选报告。

### 第三阶段：稳健背景重建

1. 实现 ROI 级 `masked_median`。
2. 实现有效样本覆盖率图和不足区域报告。
3. 增加局部曝光匹配和可选羽化。
4. 根据实际性能增加 Alpha/ROI 缓存。

### 第四阶段：高级可选能力

- ECC/特征对齐；
- 多频段融合；
- 小范围空间 inpaint；
- 外部神经网络背景修复。

## 15. 是否属于无法解决的业界难题

这不是所有情况下都无法解决，但存在明确边界。

### 可以稳定解决的情况

固定机位、背景相对静态，并且人物移动过程中目标区域曾经暴露：

- 自动零重叠候选；
- Alpha-aware masked median；
- 人工准备 clean plate；

都可以得到稳定结果。影视和虚拟制作中，预先拍摄 clean plate 是标准做法，也是本项目
最可靠、最容易验收的生产方案。

### 无法恢复“真实原像素”的情况

如果同时满足：

- 某些背景像素在所有帧中都被人物遮挡；
- 没有外部 clean plate；
- 摄像机或背景持续变化；

那么真实背景信息从未被相机观测到。此时任何算法都不可能确定原本真实像素，只能：

- 从周围纹理做传统 inpaint；
- 用生成模型合成看起来合理的内容；
- 使用纯色或人工修图；

这些方法能生成“合理背景”，但不能保证恢复“真实背景”。动态 LED、反射、阴影、
透明道具和快速灯光变化会进一步增加难度。

### 推荐实际工作流

按可靠性排序：

1. 每个固定机位提前拍摄一张曝光一致的 clean plate，并用 `external` 指定。
2. 无 clean plate 时使用 `masked_median`。
3. 背景变化较小且存在完整无遮挡帧时使用 `auto`。
4. 自动报告失败后，用 `frame` 手动覆盖候选。
5. 黑色舞台且允许丢失背景细节时使用 `color=#000000`。
6. 最后才考虑大区域生成式 inpainting。

因此，提前准备一张专用无人物背景帧完全可行，并且是最推荐的方案；`external` 模式
正是为了让它不必伪装成视频已有帧，也不必依赖现有 `frame` 编号体系。
