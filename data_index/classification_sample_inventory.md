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

1. 检查 `Hulunbuir_Labeled_Points.csv` 中 `landcover` 与 `Y_cluster` 的类别含义；
2. 读取 `Subtype_Classification.tif` 的类别值和像元数量；
3. 读取 `Subtype_Confidence.tif` 的置信度分布，确定高置信训练样本阈值；
4. 扫描 `0522/ndvi_utils.py`，确认 FVC/NDVI 数据加载方式是否可直接复用到 ARNC baseline；
5. 建立 ARNC baseline 最小输入清单。

