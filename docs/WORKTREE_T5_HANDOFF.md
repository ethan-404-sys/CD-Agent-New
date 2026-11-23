# 临时交接文档：T5 掩码优化方向（Worktree 独立会话指南）

> **文档性质**：临时多会话交接文档；待 `feat/t5-mask-refinement` 合并回 `main` 后可直接删除。开工前必读 `AGENTS.md` 与 `docs/CURRENT_STATE.md`（§1.3 $T_5$ 与 §3 方向三闭环约束）。

## 1. Worktree 初始化（在新会话开工前执行一次）

在主仓库 `/guisongxia01/pangchao/wangyihan/CD-Agent` 下执行：

```bash
cd /guisongxia01/pangchao/wangyihan/CD-Agent
bash scripts/worktree.sh create t5 feat/t5-mask-refinement
cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t5
```

脚本会自动完成：
- 从最新 `main` 切出独立分支 `feat/t5-mask-refinement` 与隔离目录 `/guisongxia01/pangchao/wangyihan/CD-Agent-wt/t5`；
- 将 `models/`、`datasets/`、`third_party/`、`outputs/` 软链接回主仓库（零磁盘复制，可直接访问 `models/SimpleClick/`、`models/sam3/`、`datasets/SECOND`、`datasets/LEVIR-CD`、`datasets/WHU-CD` 及只读初始掩码池）；
- 创建独立的本地 `.runtime/` 目录。

建议新 CLI 会话直接在该工作目录进入：

```bash
cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t5
```

## 2. 本会话的职责与文件归属边界（防冲突硬规则）

- **研究目标（对齐 `docs/CURRENT_STATE.md` $T_5$ 掩码优化任务）**：
  - 面向已有初始掩码或交互反馈，打通**误检剔除（FP）、漏检召回（FN）与边缘精修（Boundary Refinement）**的端到端迭代优化链路，覆盖：
    - **$T_{5.1}$ 自主掩码优化**：给定初始变化掩码，自主定位并修正错漏与锯齿/偏移边缘；
    - **$T_{5.2} / T_{5.3}$ 交互反馈优化**：基于点选（`T5.2`）或九宫格区域提示（`T5.3`，见 `src/cd_agent/feedback.py`）定向修正；
    - **$T_{5.4}$ 旧地图更新**：以历史建筑二值图 `map#0` 为 $T_1$、新光学影像为 $T_2$ 完成新增/拆除建筑更新（见 `src/cd_agent/map_input.py`）。
  - **与 Verifier 主线的解耦节奏**：Verifier 诊断模型正在主线训练中；现阶段 $T_5$ 优先完善任务开发集、原子精修工具链路（`add_region` / `remove_region` / `refine_region` / SimpleClick / SAM 3 局部重分割）与轨迹回放评测，暂不等待 Verifier 收敛；待主线 Verifier 权重定型合入 `main` 后，再通过只读推理接口接入闭环。
  - **可只读复用的现成初始掩码池**（严禁修改）：`outputs/verifier_diag/second_full_6cls/`（SECOND 6 类 3 模型初始掩码，`val` 18,589 条 / `train` 119,735 条）与 `outputs/verifier_diag/second_full_v1/`。
- **本会话可自由修改的目录**：
  - `experiments/t5_mask_refinement/`（新建，用于存放 $T_5$ 开发集构造、基准对比、Slurm 作业与消融脚本）
  - `src/cd_agent/feedback.py`、`src/cd_agent/map_input.py`、`src/cd_agent/skills/mask-correction/`、`src/cd_agent/skills/boundary-refinement/`（以及按需新建独立子包 `src/cd_agent/t5/`）
  - `tests/test_t5_*.py`
  - 实验产物统一落盘：`outputs/t5_mask_refinement/<YYYYMMDD>_<exp_name>/`（含 `logs/`、`metrics.json`、`viz/index.html`）
- **严禁本会话碰触的目录（其他并行会话占用）**：
  - `src/cd_agent/diagnostic/`、`src/cd_agent/verifier/`、`experiments/verifier_diag/`、`configs/splits/`（Verifier 主线占用）
  - `experiments/t3_visual_ref/`、`src/cd_agent/t32_*.py`、`src/cd_agent/examples_t3x.py`、`src/cd_agent/exemplar_semantics.py`、`src/cd_agent/semantic_bridge_tools.py`（T3 主线占用）
  - `experiments/multimodal_probe/`（多源异构主线占用）
  - 旧只读仓库 `/guisongxia01/pangchao/wangyihan/wyh/CD-Agent` 与外部 `wyh/verifier-v1-pilot-20261117`
- **公共核心文件改动约定**（`src/cd_agent/loop_json.py`、`tools.py`、`planner.py`、`AGENTS.md`、`docs/CURRENT_STATE.md`）：
  - 尽量通过独立子模块、配置或 Skill 扩展；
  - 若必须修补 `tools.py` 中的 `refine_region`/`add_region`/`remove_region` 或 `loop_json.py` 闭环开关，保持改动最小化，通过门禁后**当轮立刻合回 `main`**，再通知其他并行会话 `git rebase main`。

## 3. 实验提交与验证门禁

- 提交代码或跑正式作业前必过：`python -B scripts/verify_repo.py`（使用 `exp5-env`）；
- 所有模型与数据走 `cd_agent.paths` 或 `models/`、`datasets/` 相对路径，禁止写死 `/guisongxia01/`；
- 长耗时 GPU 作业通过 `/persist_data/home/guisongxia/bin/expctl submit` 异步提交，短 smoke test 可直接 `sbatch`（注意近期 `gpu10` 存在异常，必要时加 `--exclude=gpu10`），严禁登录节点跑计算或阻塞轮询；
- 每次正式评测务必调用 `cd_agent.evidence.build_evidence_page` 生成含双时相 **`Err@T1` + `Err@T2`** 叠加图的离线审核页（可直观核查新建与拆除两侧边缘/错漏），并严禁把 `outputs/`、图片、掩码或日志提交进 Git；
- 遵守项目防泄露与无兜底护栏原则：推理入口用 `cd_agent.verify.EpisodeInput` 隔离 GT；不添加隐藏式人工防退化回滚补丁，直接暴露真实优化瓶颈。

## 4. 何时与如何合并回 `main`

- **合并时机（小步快合，避免长分支冲突）**：
  1. 完成一个可独立验收的小里程碑（如：$T_5$ 开发集与评测脚本冻结、单期边缘精修工具跑通并量化增益、或 $T_{5.1}\text{–}T_{5.4}$ 某子任务端到端轨迹通过审核）；
  2. 只要改动了公共核心文件（`tools.py` / `loop_json.py` / `planner.py`），**当轮验收通过后立刻合并**，不跨天积压。
- **标准合并与清理步骤**：
  ```bash
  # 1. 在 t5 worktree 中通过门禁、提交并变基到最新 main
  cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t5
  /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B scripts/verify_repo.py
  git fetch origin && git rebase main

  # 2. 切回主仓库 fast-forward 合并并推送（严禁 --force）
  cd /guisongxia01/pangchao/wangyihan/CD-Agent
  git merge --ff-only feat/t5-mask-refinement
  /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-git push origin main

  # 3. 阶段结束后清理 worktree 与已合并分支
  bash scripts/worktree.sh remove t5
  git branch -d feat/t5-mask-refinement
  ```
