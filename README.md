# CD-Agent

面向高分辨率遥感双时相变化检测的**闭环迭代验证智能体系统**。核心研究定位与阶段路线见 `docs/CURRENT_STATE.md`。

## 目录结构

- `src/cd_agent/`：统一核心 Python 包（路径管理 `paths.py`、主循环与原子工具链、Worker、轨迹回放，以及 `diagnostic/` 与 `verifier/` 模块）
- `experiments/`：按研究方向分组的实验入口与 Slurm 脚本（`t3_visual_ref/`、`multimodal_probe/`、`verifier_init_masks/`、`verifier_diag/`、`closed_loop/`）
- `configs/`：路径与实验配置
- `scripts/`：`exp_loop/` 与浏览器检查等复用脚本
- `tests/`：核心单元测试
- `docs/`：活跃文档（`CURRENT_STATE.md`、`INITIAL_MASKS.md`、`T32_DEV32.md`、`MIGRATION.md`）
- `models/`、`datasets/`：指向 `/guisongxia01/pangchao/wangyihan/{models,datasets}` 的本地软链接（不入 Git）
- `third_party/`：无 `.git` 的干净第三方参考代码副本（不入 Git）
- `outputs/`、`.runtime/`：正式实验产物与临时运行态（不入 Git）

## 环境安装

在目标 Conda 环境中执行可编辑安装（或运行脚本时设置 `PYTHONPATH=src`）：

```bash
pip install --no-build-isolation --no-deps -e .
```
