# 临时交接文档：多源异构输入摸底方向（Worktree 独立会话指南）

> **文档性质**：临时多会话交接文档；待 `feat/multimodal-probe` 合并回 `main`（或按摸底结论裁剪删除）后可直接删除。开工前必读 `AGENTS.md` 与 `docs/CURRENT_STATE.md`（方向二）。

## 1. Worktree 初始化（在新会话开工前执行一次）

在主仓库 `/guisongxia01/pangchao/wangyihan/CD-Agent` 下执行：

```bash
cd /guisongxia01/pangchao/wangyihan/CD-Agent
bash scripts/worktree.sh create multimodal feat/multimodal-probe
cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/multimodal
```

脚本会自动完成：
- 从最新 `main` 切出独立分支 `feat/multimodal-probe` 与隔离目录 `/guisongxia01/pangchao/wangyihan/CD-Agent-wt/multimodal`；
- 将 `models/`、`datasets/`、`third_party/`、`outputs/` 软链接回主仓库（可直接访问 `datasets/MUMUCD`、`datasets/SpaceNet8`、`datasets/xBD` 等共享多源/灾害数据）；
- 创建独立的本地 `.runtime/` 目录。

## 2. 本会话的职责与文件归属边界（防冲突硬规则）

- **研究目标（对齐 `docs/CURRENT_STATE.md` 方向二）**：
  - **冻结当前零样本 VLM 验证器（`ask`），不引入任何重型 Verifier 改动**，控制变量单独考察 Planner 在跨模态异源场景（如光学-SAR、历史矢量地图-最新光学影像）下的理解、选工具与跨期推理边界；
  - 设定**小规模快速试跑轮次上限**（早测早决），产出真实轨迹回放与可视化证据页，交由 Ethan 裁决“保留深挖”还是“果断裁剪”。
- **本会话可自由修改的目录（高度自治、方便后续整体保留或整目录裁剪）**：
  - `experiments/multimodal_probe/`（所有数据适配、试跑脚本、Slurm 作业、摸底分析放这里）
  - `tests/test_multimodal_*.py`
  - 实验产物统一落盘：`outputs/multimodal_probe/<YYYYMMDD>_<exp_name>/`（含 `logs/`、`metrics.json`、`viz/index.html`）
- **严禁本会话碰触的目录**：
  - `src/cd_agent/diagnostic/`、`src/cd_agent/verifier/`、`experiments/verifier_diag/`、`configs/splits/`（Verifier 主线占用）
  - `experiments/t3_visual_ref/`（T3 主线占用）
  - `src/cd_agent/loop_json.py`、`tools.py`、`planner.py`（摸底阶段原则上**不改公共核心**，优先通过独立 prompt/配置/封装调用；若确需给工具加异源参数，先与主线确认再最小改动）
  - 旧只读仓库 `/guisongxia01/pangchao/wangyihan/wyh/CD-Agent` 与外部 `wyh/verifier-v1-pilot-20261117`

## 3. 实验提交与验证门禁

- 提交代码或跑正式作业前必过：`python -B scripts/verify_repo.py`（使用 `exp5-env`）；
- 所有模型与数据走 `cd_agent.paths` 或 `models/`、`datasets/` 相对路径，禁止写死 `/guisongxia01/`；
- 小样本摸底可直接短 `sbatch`，长作业走 `/persist_data/home/guisongxia/bin/expctl submit`；务必调用 `cd_agent.evidence.build_evidence_page` 生成代表性可视化页供 Ethan 拍板。

## 4. 何时与如何合并回 `main`（或裁剪丢弃）

本方向遵循**“早测早决”**原则，分两种收尾路径：

- **情况 A：Ethan 裁决「保留深挖」**——在摸底结论清晰、代码通过 `verify_repo.py` 与单测后立即合并回 `main`：
  ```bash
  cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/multimodal
  /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B scripts/verify_repo.py
  git fetch origin && git rebase main

  cd /guisongxia01/pangchao/wangyihan/CD-Agent
  git merge --ff-only feat/multimodal-probe
  /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-git push origin main

  bash scripts/worktree.sh remove multimodal
  git branch -d feat/multimodal-probe
  ```
- **情况 B：Ethan 裁决「果断裁剪」**——只需保留 `outputs/multimodal_probe/` 下的摸底报告与可视化页作为组会依据，**无需把试跑代码合入 `main`**，直接清理 worktree 与分支即可：
  ```bash
  cd /guisongxia01/pangchao/wangyihan/CD-Agent
  bash scripts/worktree.sh remove multimodal
  git branch -D feat/multimodal-probe
  ```
