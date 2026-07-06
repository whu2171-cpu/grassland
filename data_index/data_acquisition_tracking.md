# Data Acquisition Tracking

日期：2026-07-06

目标：围绕徐雪峰论文和本研究技术路线，追踪可作为地面真值、参考图、环境因子和主模型输入的数据来源。

## 1. 当前本机检索结果

检索范围：`D:\草地遥感`

### 未发现的关键地面真值/参考图

- 未发现 `2009 年中国内蒙古草地类型调查产品` 原始矢量或栅格；
- 未发现 `1980s 中国植被类型图` 原始矢量或栅格；
- 未发现 `1980s 中亚五国植被类型图` 原始数据；
- 未发现 `1990/2010 年蒙古国草地覆盖数据` 原始数据；
- 未发现徐雪峰论文中 `2021 年内蒙古东部 143 个野外样方` 表格。

### 已发现的可复用数据/脚本

| 本地路径 | 内容 | 可用性 |
|---|---|---|
| `D:\草地遥感\result\汇报\260513\dem\Mongolian_Plateau_DEM.tif` | 蒙古高原 DEM | 可直接复用 |
| `D:\草地遥感\result\汇报\260513\dem\srtm_*.tif` | SRTM 分块 | 可直接复用或拼接 |
| `D:\草地遥感\result\tiff\*_ndvi.tif` | 年度 NDVI tif | 可用于长时序/ARNC baseline |
| `D:\草地遥感\result\output\一、NDVI源数据\innermonglia源数据\*_ndvi_clip.tif` | 内蒙古 NDVI 裁剪数据 | 可用于 ARNC/趋势分析 |
| `D:\草地遥感\result\output\一、NDVI源数据\monglia源数据\*_ndvi_clip.tif` | 蒙古国 NDVI 裁剪数据 | 可用于 ARNC/趋势分析 |
| `D:\草地遥感\result\汇报\0604\download\download_era5.py` | ERA5-Land 下载脚本 | 可改造复用 |
| `D:\草地遥感\result\汇报\0604\download\download_soilgrids.py` | SoilGrids 下载脚本 | 可改造复用 |
| `D:\草地遥感\result\汇报\260513\output\Subtype_Classification*.tif` | 现有亚型分类产品 | 可作 pseudo-label/产品对照 |
| `D:\草地遥感\result\汇报\260513\output\Subtype_Confidence*.tif` | 亚型置信度产品 | 可筛选高置信伪标签 |

## 2. 数据来源追踪表

| 数据 | 状态 | 当前结论 | 下一步 |
|---|---|---|---|
| 2009 年中国内蒙古草地类型调查产品，1:100 万 | 未在本机发现；未找到明确公开下载入口 | 论文称由内蒙古自治区草原勘察规划院提供，可能不是公开数据 | 优先通过导师/学院/作者/草原勘察规划院渠道获取；同时找《中国草地资源数据》或内蒙古草地资源图替代 |
| 1980s 中国植被类型图，1:100 万 | 未在本机发现 | 可能来自《中国植被图集》或相关数字化产品，公开 GIS 下载不稳定 | 检索学校图书馆、数据平台、国家地球系统科学数据中心/资源环境科学数据中心；可接受扫描图集数字化但需标注精度限制 |
| 1980s 中亚五国植被类型图，1:250 万 | 未在本机发现 | 论文称由中国科学院中亚区域研究中心提供 | 作为低优先级，除非研究扩展到中亚，否则暂不影响内蒙古主线 |
| 蒙古国 1990/2010 草地覆盖数据，30 m | 未在本机发现；未定位到明确下载页 | 论文称来自 IKCEST 减灾知识服务网站，但搜索未命中具体数据集 | 继续在 IKCEST/CASEarth/相关论文中查找；可用 ESA CCI、GlobeLand30、Dynamic World 或自制 Landsat 分类作替代 |
| 2021 年内蒙古东部 143 个野外样方 | 未在本机发现；未检索到公开表格 | 最可能是论文课题组私有样方 | 联系作者或导师渠道；若无法获得，需自建人工解译/野外样点 |
| MOD13A1 NDVI | 可公开获取 | GEE/LP DAAC 均可获取；本机已有部分 NDVI 派生数据 | 可用于 ARNC baseline 和长时序背景 |
| MOD09A1 地表反射率 | 可公开获取 | 可用于 NDWI 和多光谱特征 | 如需复刻徐论文机器学习特征，可从 GEE 批量导出 |
| NOAA CDR AVHRR NDVI V5 | 可公开获取 | GEE 数据集 `NOAA/CDR/AVHRR/NDVI/V5`，1981-2013，0.05° | 本机已有年度 NDVI，可先复用 |
| NOAA CDR AVHRR Surface Reflectance V5 | 可公开获取 | GEE 数据集 `NOAA/CDR/AVHRR/SR/V5`，1981-2013，0.05° | 需要复刻红光/近红外月特征时再下载 |
| ERA5-Land monthly | 可公开获取 | Copernicus CDS，1950 至今，0.1°月尺度 | 改造本地脚本，下载温度、降水、土壤温度/水分、辐射等 |
| SRTM DEM | 可公开获取；本机已有 | GEE/USGS 均可获取 | 直接复用本机 DEM，并派生 slope/aspect/TWI |
| SoilGrids | 可公开获取；已有下载脚本 | 可替代土壤有机质、含砂量、容重等 | 改造脚本下载/裁剪 sand、soc、bdod、clay、silt |

## 3. 在线来源确认

可直接获取的数据源：

- MOD13A1 NDVI：GEE `MODIS/061/MOD13A1`，16 d，500 m；
- MOD09A1 Surface Reflectance：GEE `MODIS/061/MOD09A1`，8 d，500 m；
- NOAA CDR AVHRR NDVI V5：GEE `NOAA/CDR/AVHRR/NDVI/V5`，1981-2013，0.05°；
- NOAA CDR AVHRR Surface Reflectance V5：GEE `NOAA/CDR/AVHRR/SR/V5`，1981-2013，0.05°；
- ERA5-Land monthly：Copernicus CDS，1950 至今，0.1°；
- SRTM DEM：GEE `USGS/SRTMGL1_003`；
- SoilGrids：ISRIC SoilGrids，可获取 sand、clay、silt、soc、bdod、pH、CEC 等。

尚未定位到直接公开下载的数据：

- 2009 年中国内蒙古草地类型调查产品；
- 1980s 中国植被类型图 GIS 数据；
- 1980s 中亚五国植被类型图；
- 蒙古国 1990/2010 草地覆盖数据；
- 徐论文 2021 年 143 个野外样方表。

## 4. 建议的获取优先级

### P0：现在就能开始

1. 统一本地 DEM、NDVI、Subtype 分类图和置信度图的网格；
2. 生成 slope、aspect、TWI；
3. 改造 ERA5-Land 和 SoilGrids 下载脚本；
4. 建立主模型环境因子输入清单。

### P1：需要人工渠道获取

1. 向导师/学院/合作单位询问是否已有 `2009 年内蒙古草地类型调查产品`；
2. 尝试联系徐雪峰论文作者或课题组，询问 143 个样方和参考图是否可共享；
3. 检索学校图书馆或数据资源，寻找《中国植被图集》数字化数据；
4. 检索 IKCEST/CASEarth 是否有蒙古国 1990/2010 草地覆盖数据。

### P2：替代方案

若 P1 数据拿不到：

1. 使用现有 `Subtype_Classification_Full.tif` + `Subtype_Confidence_Full.tif` 生成高置信 pseudo-label；
2. 使用 Sentinel-2/Google Earth 人工解译样点构建小规模独立验证；
3. 使用 GlobeLand30、ESA CCI、Dynamic World、FROM-GLC 等产品构建草地范围和土地覆盖约束；
4. 报告中明确区分 pseudo-label、reference map 和 independent validation。
