# 临时交接文档：T3 视觉参考方向（Worktree 独立会话指南）

> **文档性质**：临时多会话交接文档；待 `feat/t3-visual-ref` 合并回 `main` 后可直接删除。开工前必读 `AGENTS.md` 与 `docs/CURRENT_STATE.md`（方向一）。

## 1. Worktree 初始化（在新会话开工前执行一次）

在主仓库 `/guisongxia01/pangchao/wangyihan/CD-Agent` 下执行：

```bash
cd /guisongxia01/pangchao/wangyihan/CD-Agent
bash scripts/worktree.sh create t3 feat/t3-visual-ref
cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t3
```

脚本会自动完成：
- 从最新 `main` 切出独立分支 `feat/t3-visual-ref` 与隔离目录 `/guisongxia01/pangchao/wangyihan/CD-Agent-wt/t3`；
- 将 `models/`、`datasets/`、`third_party/`、`outputs/` 软链接回主仓库（零磁盘复制，产物统一归口）；
- 创建独立的本地 `.runtime/` 目录。

若启动新 CLI 会话，建议直接在该目录进入：
```bash
cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t3
```

## 2. 本会话的职责与文件归属边界（防冲突硬规则）

- **研究目标（对齐 `docs/CURRENT_STATE.md` 方向一）**：
  - 搭建由 Ethan 人工把关的 $T_3$ 视觉参考系列开发集（含稀缺类视觉参考 + OSM 协同），确保视觉参考指代清晰、真值无歧义；
  - 持续迭代 Agent 测试并输出清晰轨迹与中间掩码回放，供 Ethan 审查视觉参考检索与判别链路。
- **本会话可自由修改的目录**：
  - `experiments/t3_visual_ref/`（如 `run_t32_dev32.sbatch` 及新增 T3 脚本）
  - `src/cd_agent/t32_dev32.py`、`src/cd_agent/t32_replay.py`、`src/cd_agent/examples_t3x.py`、`src/cd_agent/exemplar_semantics.py`、`src/cd_agent/semantic_bridge_tools.py`（或新建独立子包 `src/cd_agent/t3/`）
  - `tests/test_t3_*.py`、`docs/T32_DEV32.md`
  - 实验产物统一落盘：`outputs/t3_visual_ref/<YYYYMMDD>_<exp_name>/`（含 `logs/`、`metrics.json`、`viz/index.html`）
- **严禁本会话碰触的目录（Verifier 主线正在并发使用）**：
  - `src/cd_agent/diagnostic/`、`src/cd_agent/verifier/`、`experiments/verifier_diag/`、`configs/splits/`
  - 旧只读仓库 `/guisongxia01/pangchao/wangyihan/wyh/CD-Agent` 与外部 `wyh/verifier-v1-pilot-20261117`
- **公共核心文件改动约定**（`src/cd_agent/loop_json.py`、`tools.py`、`planner.py`、`AGENTS.md`、`docs/CURRENT_STATE.md`）：
  - 尽量通过参数/配置/独立子模块扩展，不直接魔改公共核心；
  - 若确实必须改公共核心，先最小化改动、过门禁并**尽早合入 `main`**，再通知其他并行会话 `git rebase main`。

## 3. 实验提交与验证门禁

- 提交代码或跑正式作业前必过：`python -B scripts/verify_repo.py`（使用 `exp5-env`）；
- 长耗时 GPU 作业通过 `/persist_data/home/guisongxia/bin/expctl submit` 异步提交，严禁登录节点跑计算或阻塞轮询；
- 所有正式结果附带离线可视化审核页（`cd_agent.evidence.build_evidence_page` 或 `t32_replay`），严禁把 `outputs/`、图片或日志提交进 Git。

## 4. 何时与如何合并回 `main`

- **合并时机（满足任一里程碑即合并，切忌长分支拖成冲突）**：
  1. 完成一个可独立验收的小里程碑（如：新版 $T_3$ 开发集清单冻结、某项视觉参考检索工具修复并通过回归、或一轮 Dev 评测链路跑通）；
  2. 只要改动了公共核心文件（`loop_json.py` / `tools.py` / `planner.py`），**当轮验收通过后立刻合并**，不跨天积压。
- **标准合并与清理步骤**：
  ```bash
  # 1. 在 t3 worktree 中先通过门禁并提交
  cd /guisongxia01/pangchao/wangyihan/CD-Agent-wt/t3
  /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B scripts/verify_repo.py
  git fetch origin && git rebase main

  # 2. 切回主仓库合并并推送（严禁 --force）
  cd /guisongxia01/pangchao/wangyihan/CD-Agent
  git merge --ff-only feat/t3-visual-ref
  /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-git push origin main

  # 3. 任务阶段结束后立即清理 worktree 与已合并分支
  bash scripts/worktree.sh remove t3
  git branch -d feat/t3-visual-ref
  ```
