# 多人物残影回收：保留全部连通区域的参数化方案

## 0. 结论

本期只解决当前回收插值会丢弃非最大连通区域的问题，不实现人物检测、人数统计、身份跟踪或逐人物独立轨迹。

新增一个可扩展的多人回收策略参数：

```text
--multi_subject_mode
```

本期只接受两个已经实现的值：

| 值 | 是否默认 | 语义 |
| --- | --- | --- |
| `largest_component` | 是 | 完全保持当前行为。回收插值只裁取最大 Alpha 连通区域，兼容已有输出。 |
| `all_components` | 否 | 使用所有前景连通区域的联合范围作为一个整体残影参与回收，避免较小或分离的人物在首次非整数插值时被丢弃。 |

未来如果需要真正的逐人物身份轨迹，可以在同一参数架构下增加新的策略值，例如 `instance_tracks`，但本期：

- 不把未实现值加入 `argparse.choices`；
- 不增加检测器或跟踪器；
- 不改变 `all_ghosts` 与 `permanent_indices` 的数据模型；
- 不声称 `all_components` 能识别人数或保证残影回到对应人物。

## 1. 背景与问题定位

当前每个密集轨迹样本只保存一张完整 Alpha：

```python
all_ghosts.append({
    'frame': current_frame.copy(),
    'alpha': alpha_mask.copy(),
    'opacity': sample_opacity,
})
```

RVM 输出的是整幅画面的统一前景 Alpha。当画面包含多个互不相连的人物时，一张 Alpha 中会存在多个连通区域。

当前回收流程存在两种不同路径：

```text
整数轨迹位置
    -> interpolate_ghost() 直接返回完整原始样本
    -> 所有人物区域仍然存在

非整数轨迹位置
    -> align_ghost_to_center()
    -> _get_alpha_bbox() 只选择面积最大的连通区域
    -> 只裁取最大区域对应的 RGB 和 Alpha
    -> 其他人物区域从该帧开始消失
```

在 `QPG_88-2026-08-03-130712.mp4` 中：

- 第 79 帧仍处于整数轨迹位置，右侧较小人物残影存在；
- 第 80 帧首次进入非整数轨迹插值；
- 横向人物是较大的主连通区域，右侧人物属于另一个较小区域；
- 对齐裁剪只保留主连通区域，因此右侧人物在第 80 帧整块消失。

这不是视频压缩、透明度渐变或冻结人物保护遮罩造成的，而是回收几何裁剪策略造成的确定性结果。

## 2. 目标

1. 默认输出与当前版本保持一致，不让升级自动改变已有任务结果。
2. 通过显式参数启用“保留全部连通区域”的短期多人修复。
3. `all_components` 在任何非整数回收位置都不得仅因面积排序而丢弃某个前景区域。
4. 保持现有回收时间、透明度、轨迹进度、RIFE、背景和人物保护逻辑不变。
5. 参数使用枚举形式，使未来可以在不再新增互斥布尔开关的情况下扩展新策略。
6. 文档必须明确：本期的 `all_components` 是“多人作为一个整体回收”，不是“每个人拥有独立身份轨迹”。

## 3. 非目标

本期不实现：

- 自动确认画面人数；
- 按人物拆分 RVM Alpha；
- YOLO、SAM2、姿态或外观 ReID；
- ByteTrack、BoT-SORT 或其他跨帧身份跟踪；
- 一个人对应一条独立回收轨迹；
- 遮挡、交叉或人物进出画面时的身份恢复；
- 自动判断应该使用 `largest_component` 还是 `all_components`；
- 新增连通区域面积过滤参数；
- 修改现有 Alpha 阈值 `8`；
- 修复或重构与本问题无关的分割、画布、背景或编码逻辑。

## 4. 参数契约

### 4.1 命令行

在 `build_spacetime_slicer.py` 的解析器中新增：

```python
parser.add_argument(
    '--multi_subject_mode',
    default='largest_component',
    choices=['largest_component', 'all_components'],
    help=(
        'Recovery handling for alpha masks containing multiple disconnected '
        'subjects: largest_component preserves current behavior; '
        'all_components moves every foreground component as one group.'
    ),
)
```

命令示例：

```powershell
uv run python batch_run.py `
  --sub_dir 0803 `
  --dataset QPG_88-2026-08-03-130712 `
  --multi_subject_mode all_components `
  --force
```

`batch_run.py` 已把未消费的切片参数传递给 `build_spacetime_slicer.py`，因此预计不需要为该参数增加批处理专用解析逻辑；实现时仍需用参数传递测试确认。

### 4.2 JSON 配置

在 `configs/spacetime_slicer.json` 中增加：

```json
"multi_subject_mode": "largest_component"
```

必须使用当前行为作为默认值。配置加载器会根据 `argparse` 的 `choices` 验证 JSON，因此未知策略应在开始生成视频前报错。

### 4.3 Python 调用链

参数传递路径为：

```text
build_parser()
    -> args.multi_subject_mode
    -> SpacetimeSlicer.generate(..., multi_subject_mode=...)
    -> self.multi_subject_mode
    -> get_ghost_geometry()
    -> align_ghost_to_center()
    -> Alpha 范围选择辅助函数
```

`generate()` 应显式验证两个允许值，即使调用方绕过命令行解析器直接调用 Python API，也不能静默接受未知模式。

## 5. 两种模式的精确定义

### 5.1 `largest_component`

该模式必须保持当前行为：

1. 使用现有阈值建立二值前景：

   ```python
   binary_alpha = (alpha_mask > 8).astype(np.uint8)
   ```

2. 使用 `connectedComponentsWithStats()` 查找连通区域。
3. 选择 `CC_STAT_AREA` 最大的前景区域。
4. 对齐阶段只裁取该区域的包围框。
5. `centroid_mask` 的现有行为保持不变，避免默认模式产生非预期画面变化。
6. 没有有效前景时继续沿用当前全帧回退行为。

本期不顺便清理 `centroid_mask=True` 时“全 Alpha 质心与最大区域包围框”的历史语义，以免默认模式与旧输出不一致。该问题可以在后续单独评估。

### 5.2 `all_components`

该模式把一张 Alpha 中的全部有效前景视为一个群体：

1. 继续使用同一阈值：

   ```python
   binary_alpha = alpha_mask > 8
   ```

2. 不按面积丢弃任何前景连通区域。
3. 裁剪框使用全部非零前景像素的联合包围框：

   ```text
   left   = 所有前景像素的最小 x
   top    = 所有前景像素的最小 y
   right  = 所有前景像素的最大 x
   bottom = 所有前景像素的最大 y
   ```

4. `centroid_mask=True` 时，锚点使用全部二值前景的面积加权质心。
5. `centroid_mask=False` 时，锚点使用联合包围框中心。
6. `align_ghost_to_center()` 裁取联合包围框内的 RGB 和完整 Alpha，因此所有人物及其相对位置被整体保留。
7. 相邻密集样本仍按当前预乘 Alpha 方法混合，轨迹进度与缓动函数不变。
8. 没有有效前景时继续使用全帧回退，避免新增空框异常。

该方案只保证“不因最大连通区域裁剪而丢失其他区域”。它不保证：

- 不同人物拥有独立方向或速度；
- 人物 A 回收到冻结帧中的人物 A；
- 人物交叉时保持身份；
- 远距离多人不会扩大联合包围框；
- Alpha 噪点不会把联合包围框拉大。

## 6. 内部结构设计

### 6.1 集中范围选择，避免分支漂移

当前 `get_ghost_geometry()` 和 `_get_alpha_bbox()` 各自执行一次最大连通区域选择。新增模式后不能分别复制一套判断，否则几何锚点和实际裁剪框容易使用不同区域。

建议新增一个统一辅助函数，名称可在实现时按代码风格调整：

```python
def get_alpha_extent(self, alpha_mask):
    """Return the foreground bbox and anchor inputs for the active mode."""
```

返回内容至少包括：

```text
bbox_x, bbox_y, bbox_w, bbox_h
binary mask used for centroid calculation
```

行为为：

```text
largest_component
    -> 包围框取最大区域
    -> 质心输入保持当前兼容语义

all_components
    -> 包围框取全部前景联合范围
    -> 质心输入取全部前景
```

`get_ghost_geometry()` 和 `align_ghost_to_center()` 必须读取同一模式下的范围结果。

### 6.2 几何缓存

当前几何结果写入：

```python
ghost['geometry']
```

模式在一次 `generate()` 中固定，因此运行时不会切换。但为了测试隔离和未来扩展，缓存不应让同一个样本在不同模式间复用错误结果。

建议二选一：

1. 使用包含模式和质心开关的缓存键；或
2. 在样本中保存模式特定缓存，例如：

   ```python
   ghost.setdefault('geometry_by_mode', {})[(mode, use_centroid)] = geometry
   ```

优先选择第二种，语义明确，也不会依赖调用方在切换模式时手动删除缓存。

### 6.3 作用范围

`multi_subject_mode` 本期只参与回收插值几何：

- `get_ghost_geometry()`；
- `align_ghost_to_center()`；
- 它们共用的 Alpha 范围辅助函数。

它不影响：

- `compose_static_ghosts()`：该函数本来就使用完整 Alpha；
- `patched_canvas` 的残影烧入；
- 当前人物保护遮罩；
- `ghost_interval` 和永久残影捕获；
- `build_recovery_trajectories()` 的时间进度；
- `compose_recovery_frame()` 的透明度和层级；
- 回收背景、过渡帧和冻结环绕。

虽然问题在 `source` 输出中被发现，但几何对齐属于共用回收逻辑，因此参数对两种 `effect_base_mode` 使用一致语义，避免相同 Alpha 在不同画布模式下再次出现区域丢失。

## 7. 实施步骤

### 步骤 1：增加参数和配置默认值

修改：

- `build_spacetime_slicer.py`
- `configs/spacetime_slicer.json`

工作内容：

1. 新增 `--multi_subject_mode` 及两个 choices。
2. 默认设为 `largest_component`。
3. `main()` 调用 `slicer.generate()` 时传入参数。
4. 配置文件增加同名字段。

### 步骤 2：把模式传入回收几何

修改：

- `models/spacetime_slicer.py`

工作内容：

1. `generate()` 增加 `multi_subject_mode='largest_component'`。
2. 在创建视频 writer 前验证参数值。
3. 保存本次运行的模式，供几何辅助函数读取。
4. 对直接构造 `SpacetimeSlicer.__new__()` 的旧测试使用安全默认值：

   ```python
   getattr(self, 'multi_subject_mode', 'largest_component')
   ```

### 步骤 3：统一 Alpha 范围计算

修改：

- `models/spacetime_slicer.py`

工作内容：

1. 抽取统一的范围选择辅助函数。
2. `largest_component` 走兼容分支。
3. `all_components` 计算全部非零前景的联合包围框。
4. 空 Alpha 保留当前回退语义。
5. `get_ghost_geometry()` 和 `align_ghost_to_center()` 使用同一范围选择结果。
6. 几何缓存按模式和 `centroid_mask` 隔离。

### 步骤 4：补充测试

修改：

- `test/test_spacetime_slicer.py`
- 如有必要，新增或扩展参数解析测试文件

测试详见第 9 节。

### 步骤 5：更新使用文档

修改：

- `README.md`
- `config_effect75_orbit90.txt` 中与当前生产命令相关的参数说明（保持文件现有编码）

文档必须同时说明：

- 默认值不改变旧行为；
- `all_components` 解决的是区域被裁掉；
- 多人仍作为一个群体回收；
- 它不是人数识别或逐人物身份跟踪。

## 8. 向后兼容要求

1. 不传参数时，输出应与改动前的 `largest_component` 行为一致。
2. JSON 中不增加字段的自定义旧配置也应继续工作，由解析器默认值补齐。
3. 显式使用 `--multi_subject_mode largest_component` 应与不传参数等价。
4. 未知值必须在生成视频前报错，不能回退到任意策略。
5. `centroid_mask` 与 `multi_subject_mode` 是正交参数：

   | `multi_subject_mode` | `centroid_mask=True` | `centroid_mask=False` |
   | --- | --- | --- |
   | `largest_component` | 保持当前质心语义 | 最大区域包围框中心 |
   | `all_components` | 全部前景面积质心 | 全部前景联合框中心 |

6. 不改变现有 `alpha > 8` 的范围判定，避免本期同时引入阈值调参问题。

## 9. 测试计划

### 9.1 参数测试

1. 不传参数时解析为 `largest_component`。
2. CLI 可以选择 `all_components`。
3. JSON 可以设置 `all_components`。
4. CLI 能覆盖 JSON 中的值。
5. 未知值在参数解析阶段失败。
6. 直接调用 `generate()` 传入未知值时失败，并且失败发生在创建输出 writer 之前。

### 9.2 默认行为回归

保留并明确现有测试：

```text
test_recovery_geometry_ignores_disconnected_alpha_noise
```

该测试应在默认 `largest_component` 下继续确认：较小的离散区域不参与最大区域包围框。

新增断言：

- 未设置 `multi_subject_mode` 的旧式测试对象仍默认使用 `largest_component`；
- 整数轨迹位置与当前版本完全一致；
- 单人物 Alpha 在两个模式下输出一致。

### 9.3 `all_components` 几何测试

构造一张包含左右两个分离人物块的 Alpha：

```text
...AAA........BBB...
...AAA........BBB...
```

验证：

1. 联合包围框同时覆盖 A 和 B。
2. `centroid_mask=True` 的锚点位于全部前景的面积质心。
3. `centroid_mask=False` 的锚点位于联合框中心。
4. 对齐平移后 A、B 均保留非零 Alpha。
5. 当目标接近画面边界而发生裁边时，只允许几何上越界的部分被裁掉，不能按面积删除整个 B。

### 9.4 首个非整数插值回归

构造连续两帧，每帧同时包含一个大人物区域 A 和一个小人物区域 B：

1. `position=0.0` 时两者都存在。
2. `position=0.5` 且模式为 `largest_component` 时保持当前只对齐主区域的行为。
3. `position=0.5` 且模式为 `all_components` 时 A、B 都必须存在。
4. 混合 Alpha 使用当前预乘计算，不产生除零、溢出或背景黑边。

该测试直接覆盖本视频“第 79 帧存在、第 80 帧消失”的根因。

### 9.5 回收流程测试

使用至少两个永久残影、多个密集样本和两个分离人物区域，验证：

- 每个输出回收帧都保留两个区域；
- 回收末帧仍到达当前统一的最后密集样本；
- 原有 opacity 保持不变；
- freeze 人物保护遮罩仍能裁减重叠像素；
- `recovery_transition_frames` 数量不变；
- `source` 和 `patched_canvas` 均不抛出异常。

### 9.6 视频验收

使用本次问题视频相同输入重新生成两份输出：

```text
largest_component（默认基线）
all_components（修复候选）
```

重点检查输出第 75–90 帧：

1. 默认基线应复现右侧人物在首次非整数回收插值时消失。
2. `all_components` 中右侧人物不得在相同时点整块消失。
3. 全部人物应作为一个群体连续移动，不出现由裁剪框切换造成的瞬移。
4. 检查联合框是否因远端 Alpha 噪点过度扩大。
5. 检查人物相对运动差异是否造成群体轨迹视觉不自然。

第 4、5 项属于该短期方案的质量评估。如果不满足要求，应保留本期参数和默认兼容模式，后续通过新的策略值引入实例轨迹，不能把 `all_components` 静默改造成另一种语义。

## 10. 性能与资源预期

`all_components` 不新增模型推理。额外工作只包括：

- 在已有二值 Alpha 上计算全部前景范围；
- 使用可能更大的联合裁剪区域做平移和插值；
- 对联合区域内的 RGB 与 Alpha 做预乘混合。

连通区域/联合范围计算本身相对 RVM、RIFE 和 H.264 编码应很小。实际增加主要取决于联合包围框面积：

- 多个人物靠近时，开销接近当前模式；
- 多个人物分布在画面两端时，联合框可能接近整幅 4K 画面，回收插值的内存带宽和像素运算会增加；
- 不应为短期优化重新缩放或分别裁剪区域，因为那会引入超出本期范围的新合成语义。

验收时记录同一视频两种模式的总生成时间和峰值显存/内存；本期不预设未经实测的百分比阈值，但要求不存在数量级增长。

## 11. 已知风险与处理原则

### 11.1 联合框过大

如果 Alpha 中存在远端噪点，`all_components` 会按“全部区域”的严格语义保留它，并扩大联合框。本期不增加面积过滤，以免参数含义变成“保留部分区域”。通过调试帧和视频验收记录该风险。

### 11.2 群体中心不能表达独立人物运动

所有人物仍共享一个整体锚点。如果人物运动方向不同，回收可能出现整体漂移、相互交叉或视觉上接近错误目标。这是方案能力边界，不应在本期用最近邻匹配等隐式逻辑修补。

### 11.3 人物身份不受保证

`all_components` 只保留像素区域，没有 `track_id`。文档、CLI help 和日志不得使用“对应人物回收”“自动多人跟踪”等表述。

### 11.4 模式日志

生成开始时应打印当前模式，例如：

```text
Multi-subject recovery mode: all_components (single group trajectory)
```

这样问题复现和批量结果比较时可以确认有效参数，避免只根据配置文件猜测。

## 12. 验收标准

本期完成需要同时满足：

1. 新参数可从 CLI 和 JSON 配置进入完整生成链路。
2. 默认值为 `largest_component`，已有命令不改变行为。
3. `all_components` 的整数和非整数回收帧都保留所有有效前景区域。
4. 本次视频中的右侧人物不再在第 80 帧因最大区域裁剪而整块消失。
5. 单人物素材在两种模式下保持等价。
6. 不改变回收帧数、缓动、透明度、人物保护、背景和冻结环绕行为。
7. 自动测试覆盖参数验证、默认兼容、联合几何、首次非整数插值和完整回收。
8. README 明确标注该方案是群体轨迹而非逐人物跟踪。
9. 未引入人物检测、身份跟踪或未实现的枚举值。
10. 对问题视频完成两种模式的对照生成和 75–90 帧人工验收。

## 13. 后续扩展边界

本期只预留枚举式策略入口，不规划实例跟踪的实现细节。未来若 `all_components` 无法满足多人独立运动需求，可以增加新策略值，但应遵守：

1. `largest_component` 与 `all_components` 的既有语义保持不变。
2. 新策略拥有独立的数据结构和验证逻辑，不在 `all_components` 内隐式启用检测器。
3. 未实现的新策略不出现在 CLI choices 和默认配置中。
4. 新策略需要另建计划，单独讨论人数检测、身份保持、遮挡恢复、性能和人工校验。

