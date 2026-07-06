# Classification Sample and Product Inventory

本清单用于下一阶段“ARNC baseline + 亚型分类精度验证”的数据盘点。所有大体积数据保留在本地路径，不进入 Git。

## 1. Spatial Products

来源目录：`D:\草地遥感\result\汇报\260513\output`

| 文件 | 类型 | 尺寸/波段 | CRS | 用途判断 | 注意事项 |
|---|---:|---:|---|---|---|
| `Subtype_Classification.tif` | int8 单波段 | 3217 x 1774, 1 band | EPSG:4326 | 候选区域亚型硬分类图 | nodata = -1；可用于面积统计、掩膜和与 ARNC baseline 对比 |
| `Subtype_Classification_Full.tif` | int8 单波段 | 4266 x 1774, 1 band | EPSG:4326 | 候选完整亚型硬分类图 | 覆盖范围更大；需确认与气候/FVC 网格对齐关系 |
| `Subtype_Probabilities.tif` | float32 多波段 | 3217 x 1774, 5 bands | EPSG:4326 | 亚型概率图 | 可用于不确定性、过渡带识别和置信度筛选 |
| `Subtype_Probabilities_Full.tif` | float32 多波段 | 未抽取 | EPSG:4326 | 完整范围亚型概率图 | 文件较大；后续按需读取 |
| `Subtype_Confidence.tif` | float32 单波段 | 3217 x 1774, 1 band | EPSG:4326 | 分类置信度 | 可用于剔除低置信像元，构建训练样本 |
| `Subtype_Confidence_Full.tif` | float32 单波段 | 未抽取 | EPSG:4326 | 完整范围分类置信度 | 后续按需读取 |
| `State_Dominant.tif` | int16 单波段 | 3217 x 1774, 1 band | EPSG:4326 | GMM 生态状态硬分类 | 可辅助解释生态状态，但不是亚型标签 |
| `State_Probabilities.tif` | float32 多波段 | 未抽取 | EPSG:4326 | GMM 状态概率 | 可辅助不确定性和状态转移解释 |
| `State_Uncertainty.tif` | float32 单波段 | 未抽取 | EPSG:4326 | 状态不确定性 | 可作为不确定性辅助层 |
| `Trend_Classification_MK_InnerMongolia.tif` | int8 单波段 | 3217 x 1774, 1 band | EPSG:4326 | 内蒙古 FVC/MK 趋势分类 | 可用于退化/改善区域筛选，不能作为亚型分类精度 |
| `Trend_Classification_MK_OuterMongolia.tif` | int8 单波段 | 未抽取 | EPSG:4326 | 蒙古国趋势分类 | 后续用于内外蒙对比 |

## 2. CSV Tables and Candidate Samples

来源目录：`D:\草地遥感\result\汇报\260513\output`

| 文件 | 字段概览 | 用途判断 | 是否可作独立验证 |
|---|---|---|---|
| `GEE_Environmental_Samples.csv` | `lon, lat, precip, temp, night, landcover, dist_human` | 环境因子采样点；可用于 GeoDetector 或分类特征分析 | 暂不作为独立验证，需确认 `landcover` 来源 |
| `GEE_Samples_by_Region.csv` | `lon, lat, precip, temp, night, landcover, dist_human, region` | 分区域环境样本；适合空间分区和内外蒙对比 | 暂不作为独立验证 |
| `Hulunbuir_Labeled_Points.csv` | `dist_human, elevation, hetero, landcover, ndvi_cv, ndvi_mean, ndvi_trend, night, precip, slope, temp, .geo, Y_cluster, dist_level` | 呼伦贝尔标注/聚类样本；可作为候选验证样本或机制解释样本 | 待确认标签来源，不能直接默认独立 |
| `Feature_Importance.csv` | 特征重要性 | 记录模型解释结果 | 否 |
| `Feature_Importance_Full.csv` | 完整特征重要性 | 记录模型解释结果 | 否 |
| `Human_Factor_Stats.csv` | 人类活动统计 | 驱动解释 | 否 |
| `Human_Factor_Deep_GD.csv` | 人类因子 GeoDetector | 驱动解释 | 否 |
| `Period_GeoDetector_Results.csv` | 分阶段 GeoDetector | 驱动解释 | 否 |
| `Subtype_GeoDetector_Results.csv` | `subtype, factor, factor_en, q_value` | 分亚型驱动因子贡献 | 否 |
| `Region_GeoDetector_Comparison.csv` | 分区 GeoDetector 对比 | 区域驱动解释 | 否 |

## 3. Immediate Decisions

### Can Be Used as Classification Inputs

- `Subtype_Classification*.tif`：作为现有亚型产品和训练候选标签，但不能同时用于训练和验证；
- `Subtype_Probabilities*.tif`：用于构建过渡带、不确定性图和高置信样本；
- `Subtype_Confidence*.tif`：用于筛选高置信训练样本；
- GEE 环境样本：用于环境因子特征和驱动解释。

### Can Be Used as Candidate Validation Data

- `Hulunbuir_Labeled_Points.csv`：需要人工确认 `landcover`、`Y_cluster` 是否来自独立野外/人工标注，还是模型派生标签；
- 后续应继续寻找野外样方、人工解译点或开题报告中提到的呼伦贝尔调查数据。

### Cannot Be Used as Classification Accuracy Evidence Directly

- FVC 预测 R2；
- RESTREND 改善/退化比例；
- GeoDetector q 值；
- SHAP 重要性；
- 政策滞后相关图；
- 包头靶场训练精度。

这些结果可以解释生态响应和驱动机制，但不能替代亚型分类的 OA/UA/PA/F1。

## 4. Next Inventory Tasks

1. 已检查 `Hulunbuir_Labeled_Points.csv` 中 `landcover` 与 `Y_cluster` 的类别分布；标签来源仍需人工确认；
2. 已读取 `Subtype_Classification.tif` 与 `Subtype_Classification_Full.tif` 的类别值和像元数量；
3. 已读取 `Subtype_Confidence.tif` 与 `Subtype_Confidence_Full.tif` 的置信度分布，并形成第一版高置信样本阈值建议；
4. 扫描 `0522/ndvi_utils.py`，确认 FVC/NDVI 数据加载方式是否可直接复用到 ARNC baseline；
5. 建立 ARNC baseline 最小输入清单。

## 5. Label and Raster Audit, 2026-07-06

检查脚本：`src/validation/inspect_classification_inputs.py`

运行命令：

```powershell
python src/validation/inspect_classification_inputs.py
```

### 5.1 `Hulunbuir_Labeled_Points.csv`

- 样本量：1,549。
- 字段：`system:index`, `dist_human`, `elevation`, `hetero`, `landcover`, `ndvi_cv`, `ndvi_mean`, `ndvi_trend`, `night`, `precip`, `slope`, `temp`, `.geo`, `Y_cluster`, `dist_level`。
- `landcover`：全部为 `30`，共 1,549 条。
- `Y_cluster`：`0` = 401，`1` = 205，`2` = 406，`3` = 537。
- `dist_level`：`极近 (高干扰)` = 389，`较近` = 388，`较远` = 387，`极远 (无干扰)` = 385。

判断：

- `landcover = 30` 更可能是草地掩膜或土地覆盖编码，不是草地亚型标签。
- `Y_cluster` 是四类聚类/生态状态标签的可能性高，不能在未确认来源前作为独立亚型真值。
- 该 CSV 当前可用于机制解释、聚类状态分析或候选样本筛选；暂不能作为亚型分类 OA/UA/PA/F1 的独立验证集。

### 5.2 `Subtype_Classification*.tif`

`Subtype_Classification.tif`：

| class | count | percent |
|---:|---:|---:|
| 0 | 301,717 | 32.7526 |
| 1 | 294,386 | 31.9568 |
| 2 | 231,910 | 25.1747 |
| 3 | 1,944 | 0.2110 |
| 4 | 91,244 | 9.9049 |

`Subtype_Classification_Full.tif`：

| class | count | percent |
|---:|---:|---:|
| 0 | 837,009 | 33.4199 |
| 1 | 780,037 | 31.1451 |
| 2 | 485,973 | 19.4038 |
| 3 | 150,599 | 6.0131 |
| 4 | 250,908 | 10.0182 |

判断：

- 两个产品均为 5 类硬分类图，类别编码为 `0-4`。
- 非 Full 产品中 class 3 仅占 0.2110%，极度稀少；若用于训练，容易出现类别不平衡或空间覆盖不足。
- Full 产品 class 3 占 6.0131%，更适合用于构建第一版 balanced pseudo-label 样本，但仍不能作为独立验证真值。

### 5.3 `Subtype_Confidence*.tif`

`Subtype_Confidence.tif`：

- 有效像元：921,201。
- min = 0.353407，mean = 0.932246，max = 1.000000。
- 分位数 1/5/10/25/50/75/90/95/99：0.526904, 0.634180, 0.751469, 0.927297, 0.998846, 1.000000, 1.000000, 1.000000, 1.000000。

`Subtype_Confidence_Full.tif`：

- 有效像元：2,504,526。
- min = 0.329396，mean = 0.926699，max = 1.000000。
- 分位数 1/5/10/25/50/75/90/95/99：0.501570, 0.618876, 0.690919, 0.922880, 0.996000, 1.000000, 1.000000, 1.000000, 1.000000。

判断：

- 两个置信度图整体偏高，中位数接近 1。
- 第一版训练样本建议使用 `confidence >= 0.95` 作为严格高置信阈值；若某类样本不足，可放宽到 `confidence >= 0.90`，但必须在记录中标注阈值变化。
- `confidence < 0.60` 暂不进入 baseline 训练；`0.60-0.90` 可作为过渡带/不确定区分析对象。

## 6. ARNC Baseline v0.1 Sample Rule

第一版 ARNC baseline 的目标不是证明最终精度，而是建立一个可复现的对照基线，用于回答：现有亚型产品与 ARNC 光谱-物候规则/模型在空间分布上是否一致，以及哪些类型或过渡区最不稳定。

### 6.1 Training / Pseudo-label Samples

- 标签来源优先使用 `Subtype_Classification_Full.tif`，配套使用 `Subtype_Confidence_Full.tif`。
- 初始筛选条件：`confidence >= 0.95` 且 class in `0,1,2,3,4`。
- 采用类别分层抽样，避免 class 0/1/2 主导训练。
- 每类样本量先设为同量级；若 class 3 或 class 4 高置信样本不足，可记录实际可用数量，不做盲目过采样。
- 同一产品筛出的样本只能称为 pseudo-label training samples，不能称为 independent validation samples。

### 6.2 Internal Validation

- 如果暂时没有独立野外/人工解译标签，可先做 spatial block split，而不是随机像元划分。
- block split 的目的只是降低空间自相关泄漏，仍属于内部一致性验证。
- 报告中应写为 internal consistency / pseudo-label validation，不能写成最终分类精度。

### 6.3 Independent Validation

真正用于论文精度结论的验证集需要满足至少一项：

- 独立野外样方或样点，且有明确草地亚型标签；
- 独立人工解译样本，且解译依据和类别体系清楚；
- 开题报告或已有调查数据中可追溯的呼伦贝尔/内蒙古草地类型样点。

`Hulunbuir_Labeled_Points.csv` 只有在确认 `Y_cluster` 或其他字段来自独立人工/野外亚型标注后，才可进入独立验证；否则只能用于生态状态解释或候选样本辅助。
