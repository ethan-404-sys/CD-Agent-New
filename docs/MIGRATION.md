# CD-Agent 迁移说明

本目录 `/guisongxia01/pangchao/wangyihan/CD-Agent` 由旧工作区精选活跃代码重建并统一收编至单一 `src/cd_agent/` 包。原 `/guisongxia01/pangchao/wangyihan/wyh/CD-Agent` 及外部 `/guisongxia01/pangchao/wangyihan/wyh/verifier-v1-pilot-20261117` 保持原样只读，不影响旧目录既有作业与产物。

## 已迁入并统一的资产

1. **统一核心包（`src/cd_agent/`）**：将原活跃主循环、规划器、原子工具箱、记录器、回放渲染、Worker、Prompt/Scope/Skill 配置，以及真实依赖的 8 个 `analysis/` 辅助模块与 8 个 `_compat/` 底层模块统一归入 `src/cd_agent/`，并新增 `paths.py`、`diagnostic/` 与 `verifier/`。
2. **实验入口（`experiments/`）**：
   - `experiments/t3_visual_ref/`：T3.2 Dev-32 作业脚本（已切换至 `PYTHONPATH=src`、`python -m cd_agent.t32_dev32` 与 `outputs/t3_visual_ref/t32_dev32`）。
   - `experiments/verifier_init_masks/`：统一初始掩码 CLI、实现与 4 个 Slurm 脚本（默认输出指向本仓库 `outputs/verifier_init_masks/verifier_v1_600`，避免误覆写旧目录在跑产物；如需指向旧目录可通过 `VERIFIER_DATASET_OUT` 显式指定）。
3. **基础设施与核心单测**：`scripts/exp_loop/`、`scripts/browser_checks/` 与 22 份核心单测 + `tests/test_paths_and_layout.py`。
4. **活跃文档**：`docs/CURRENT_STATE.md`、`docs/T32_DEV32.md`。
5. **非 Git 资产**：
   - `models/`：软链接至 `/guisongxia01/pangchao/wangyihan/models/` 下的 14 个遥感/视觉相关模型。
   - `datasets/`：软链接至 `/guisongxia01/pangchao/wangyihan/datasets/` 下的 9 个遥感变化检测数据集（含 `lim1/` 中的 `SECOND`、`WHU-CD`、`SYSU-CD`）。
   - `third_party/`：`DynamicEarth`、`OmniOVCD`、`ReA-OVCD` 去除嵌套 `.git` 与历史结果后的干净副本。
   - `.runtime/dynamic-earth/`：复用已编译好的 SimFeatUp 扩展 `.so`。

## 未迁入（保留在只读旧仓）

- `results/`（11 万+ 历史实验产物）、`figures/`、`interim/`、`doc/archive/`、`tools/archive/`
- `code/probe/`（232 文件，仅抽取 `sam3_tools.py`）、`code/ovcd_v0/`（仅抽取 7 个被导入模块）
- 原 `analysis/` 中未被引用的 259 个历史一次性脚本及对应 95 个旧单测
- 旧仓库的历史分支与外部 worktree
