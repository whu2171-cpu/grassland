# 2026-07-07 SoilGrids 数据修复记录

## 问题

当前模型需要加入 SoilGrids 土壤因子，但旧数据存在严重 nodata 问题。

排查发现存在两套 SoilGrids 数据：

1. `E:\1\soilgrids`
   - 栅格尺寸约为 `143999 x 64799`；
   - 小窗口抽样全部为 nodata；
   - 根因是早期 `gdalwarp` 裁剪时未指定 `-te_srs EPSG:4326`，经纬度范围被当作源投影坐标解释。

2. `D:\草地遥感\result\汇报\0604\download\soilgrids`
   - 当前配置实际引用此目录；
   - `sand`、`silt`、`bdod` 在样本点上有效率约 99%；
   - `clay` 和 `soc` 存在区域性缺失，其中 `clay` 在 105E 以东基本为空，`soc` 在样本点有效率约 19%。

## 根因判断

新版下载脚本已加入 `-te_srs EPSG:4326`，范围和分辨率正确，但 ISRIC SoilGrids 远程瓦片服务在下载 `clay/soc` 时出现过 `504` 或 `Connection reset`。

因此 `clay/soc` 不是坐标错误，而是远程瓦片读取不稳定导致的部分结果。

## 修复方案

1. 保留有效的 `sand_0-5cm_mean.tif`、`silt_0-5cm_mean.tif`、`bdod_0-5cm_mean.tif`；
2. 暂不启用低覆盖率的 `soc_0-5cm_mean.tif`；
3. 根据土壤质地闭合关系生成 derived clay：

```text
clay_derived = 1000 - sand - silt
```

并限制到 `[0, 1000]` g/kg。

生成文件：

```text
D:\草地遥感\result\汇报\0604\download\soilgrids\clay_0-5cm_mean_derived_from_sand_silt.tif
```

修复脚本：

```text
D:\草地遥感\result\汇报\0604\download\repair_soilgrids.py
```

## 配置更新

`configs/proxy_model.yaml` 中 SoilGrids 特征更新为：

- `soil_sand_0_5cm`
- `soil_silt_0_5cm`
- `soil_clay_0_5cm`，指向 derived clay
- `soil_bdod_0_5cm`

暂时移除：

- `soil_soc_0_5cm`

## 验证结果

在当前 subtype 样本点上，修复后的 SoilGrids 有效率为：

| 特征 | 有效率 | 数值范围 |
|---|---:|---|
| soil_sand_0_5cm | 0.9959 | 235-762 |
| soil_silt_0_5cm | 0.9923 | 143-544 |
| soil_clay_0_5cm | 0.9923 | 92-385 |
| soil_bdod_0_5cm | 0.9959 | 82-149 |

重建后的 `proxy_samples_subtype.csv`：

- 行数：15000；
- 列数：49；
- SoilGrids 特征列数：4；
- 全空列：无。

## 重跑模型结果

已重跑：

- `build_proxy_dataset.py`
- `train_baselines.py`
- `train_spatiotemporal.py`
- `evaluate.py`

最新 subtype 结果：

| 模型 | OA | Macro-F1 |
|---|---:|---:|
| HistGradientBoosting | 0.9854 | 0.9890 |
| Random Forest | 0.9845 | 0.9878 |
| MLP prototype | 0.9572 | 0.9667 |

当前 subtype 模型实际输入特征数为 43，其中包括 4 个 SoilGrids 特征。

RF 特征重要性中 SoilGrids 排名：

| 特征 | 排名 | importance |
|---|---:|---:|
| soil_bdod_0_5cm | 30 | 0.00429 |
| soil_sand_0_5cm | 33 | 0.00214 |
| soil_silt_0_5cm | 35 | 0.00177 |
| soil_clay_0_5cm | 39 | 0.00114 |

## 后续建议

1. 当前阶段可以使用 `sand/silt/derived clay/bdod` 作为可用土壤因子进入 baseline 和消融实验；
2. `soc` 暂不进入正式模型，避免低覆盖率导致大量中位数插补；
3. 如果后续网络稳定，可重新下载 `soc` 和原始 `clay`，并与 derived clay 做一致性检验；
4. 后续消融实验需要重新运行，确保 `soil` variant 实际使用这些土壤特征。
