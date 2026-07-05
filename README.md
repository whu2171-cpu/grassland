# grassland-remote-sensing

蒙古高原草原植被亚型遥感分类、FVC 变化分析与驱动机制研究的版本管理仓库。

## Repository Scope

本仓库用于追踪可复现的研究材料：

- 研究路线、方案、阶段报告和研究日志；
- 分类、验证、驱动分析相关脚本；
- 小体量结果表格和可用于论文写作的小图；
- 数据索引、处理记录和本地数据路径说明。

本仓库不直接追踪大体积原始数据和中间栅格结果，例如遥感影像、`.tif`、`.npy`、大型 `.pptx/.docx`、模型权重等。这些文件应保存在本地数据目录，并在 `data_index/local_data_manifest.md` 中登记。

## Suggested Workflow

1. 每完成一个阶段，更新 `CHANGELOG.md`。
2. 重要结果放入 `results/tables/` 或 `results/figures_small/`。
3. 大文件只登记路径和版本，不直接提交。
4. 每次提交信息说明：任务、输入数据版本、主要输出、精度/结论、风险。

## Version Tags

建议阶段标签：

- `v0.1-route-evaluation`
- `v0.2-arnc-baseline`
- `v0.3-classification-validation`
- `v0.4-driver-analysis`
- `v0.5-manuscript-draft`

