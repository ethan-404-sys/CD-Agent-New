# 统一初始掩码入口

2026-11-20起，三方法初始掩码的维护入口统一到本核心仓库；旧verifier工作树只作为既有实验产物来源，不继续维护独立生成实现。沿用已批准最终审核包中的MCI V4 native、Omni V3+前景补全A、ReA V3策略。600/60/60底图划分不重抽，旧成功结果不覆盖。

唯一CLI为 `experiments/verifier_init_masks/initialize_verifier_masks.py`，必要实现维护在同目录的 `initial_masks/`。原成功代码按原配置迁入，推理不再导入旧工作树的代码。已有数据目录继续使用 `wyh/verifier-v1-pilot-20261117/results/verifier_v1_600/`，不搬动或复制大体积产物。

MCI失败原因已由CPU作业193177确认：LEVIR train_69的第179个proposal原始面积1像素，nearest投影到448网格后为空，两期原生背景标签均为0，mean(empty)因此为NaN。修复记录每proposal采样面积和实际分数网格；独立QC仅在投影确实为空、原生类别为背景且没有Inf时允许这种原始NaN。保留原分数、标签、proposal和初始M，不改GT，不删除底图。非空proposal的非有限分数仍报错。

`validate_initial_masks.sbatch`为纯CPU验证：重建并逐像素核对旧最终包222条normal掩码，以及失败样本的原生M不变；另验证有采样像素的NaN仍被拒绝。`run_initial_masks.sbatch`单GPU先真实重跑train_69、三方法各数据集smoke，再续跑572对，独立生成QC、FP/FN/ignore导出和合并600对。已成功receipt仅在同一冻结任务清单下复用。

入口阶段：`regression`、`check-failure`、`infer --method ... [--source-id ...|--smoke]`、`qc`、`export`。原审核包路径可用 `--pilot-package` 指定。所有数值验证和推理均通过Slurm，日志在原数据目录 `logs/`，长作业成功失败均QQ通知。`dataset_report.json`的status=complete才表示全部训练初始化完成；验证、测试本轮仍只冻结60/60底图名单，不额外生成掩码。

## MCI 离线依赖

通过 Slurm 运行 `experiments/verifier_init_masks/prepare_mci_assets.sbatch` 安装初始掩码推理所需的资产。权重统一从 ``models/`（软链接至 `/guisongxia01/pangchao/wangyihan/models/`）` 读取。

- `SAM/sam_vit_h_4b8939.pth` 是现有 SAM ViT-H 权重。
- `dinov2-base/` 保存 Meta 的 DINOv2-base 权重和预处理配置。
- `CLIP/ViT-B-16.pt` 复用服务器上的 OpenAI 原版 CLIP。
- `SimFeatUp/xclip_jbu_one_million_aid.ckpt` 来自 SegEarth-OV 官方仓库。

DynamicEarth 源码从已核对版本的共享副本复制到 `third_party/DynamicEarth/`。首次准备资产后，通过 Slurm 运行 `experiments/verifier_init_masks/build_mci_runtime.sbatch` 编译 SimFeatUp。编译产物放在 `.runtime/dynamic-earth/`，由 MCI 入口直接加载。MCI 的模型和源码版本沿用原审核配置。Omni 和 ReA 继续从同一个模型根目录读取 `sam3/sam3.pt`。
