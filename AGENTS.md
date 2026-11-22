# CD-Agent 协作与执行规范

本文件是 `/guisongxia01/pangchao/wangyihan/CD-Agent` 仓库的协作入口，适用于本仓库及其子目录。以下路径均相对于本仓库根目录。

## 1. 单包结构与代码放置规范
- **唯一状态入口**：开工前先读 `docs/CURRENT_STATE.md`，将其作为架构边界、任务定义与阶段路线的唯一真相源。
- **单一 Python 包（`src/cd_agent/`）**：禁止在 `src/` 下新增第二个顶层包，所有可复用代码统一归属 `src/cd_agent/`：
  - `src/cd_agent/paths.py`：全局路径解析（模型、数据、产物、运行态、第三方依赖的唯一入口）。
  - `src/cd_agent/{loop_json,planner,tools,recorder,backends,state,tasks,evaluation}.py`：主循环、规划器、原子工具箱、记录器与评测核心。
  - `src/cd_agent/diagnostic/`：方向三第一步——多任务自适应空间错误诊断模型（数据构造、网络、训练与推理）。
  - `src/cd_agent/verifier/`：方向三第二步——只读轨迹感知验证智能体（Verifier Agent）。
  - `src/cd_agent/workers/`：GPU/HTTP 模型服务 Worker（`sam3_worker`、`sam_prompt_worker`、`ask_worker` 等）。
  - `src/cd_agent/prompts/`、`scopes/`、`skills/`、`site_assets/`：提示词、口径卡、技能与回放静态资源。
  - `src/cd_agent/_compat/`：从早期原型收拢的极简底层辅助模块（不继续往里堆新代码）。
- **实验与脚本隔离**：
  - `experiments/<track>/`：按研究方向组织的实验入口与 Slurm 脚本（`t3_visual_ref/`、`multimodal_probe/`、`verifier_init_masks/`、`verifier_diag/`、`closed_loop/`）；禁止把一次性实验/分析脚本塞进 `src/cd_agent/`。
  - `scripts/`：长期复用的基础设施脚本（浏览器核验等；旧 `exp_loop/` 已由全局 `expctl` 替代，不再新增依赖）。
  - `configs/`：路径与实验配置 YAML。
- **共享模型与数据**：所有模型和数据集通过本仓库 `models/`、`datasets/` 下的软链接统一指向 `/guisongxia01/pangchao/wangyihan/models` 与 `/guisongxia01/pangchao/wangyihan/datasets`；代码中统一经由 `cd_agent.paths` 或环境变量解析，禁止新增硬编码绝对路径。
- **轻量验证门禁（提交代码 / 跑正式实验前必过）**：
  - 仓库结构与卫生门禁：`python -B scripts/verify_repo.py`（检查单包边界、无硬编码路径、无违规追踪产物、软链接与核心导入）。
  - 科研不变式与交付契约：`src/cd_agent/verify.py`（推理输入 GT 隔离 `EpisodeInput`、划分防泄露 `check_split_isolation`、诊断标签集合公理 `check_diagnostic_labels`、产物目录校验 `python -m cd_agent.verify run <run_dir>`）。
  - 最小视觉证据包：`src/cd_agent/evidence.py`（自动抽取代表性高分/中分/高 FP/高 FN 样本生成离线审阅页 `index.html`）。
- **旧仓库隔离**：原 `/guisongxia01/pangchao/wangyihan/wyh/CD-Agent` 与外部 `wyh/verifier-v1-pilot-20261117` 仅作为历史只读参考或既有数据来源；严禁把旧历史目录内容反向混入本仓库。

## 2. 开发时限、训评一体与 `expctl` 闭环规范
- **耗时口径拆分（仅限前置开发）**：单次主动开发 **上限 ≤ 1 小时**，仅约束必要的代码编写、极小样本冒烟调通与 Slurm 提交准备；Slurm 排队及计算节点实际运行耗时**不计入**该时限。严禁为卡 1 小时而缩减已确定的训练 Epoch、样本量或评测规模。
- **严格锁定当轮范围**：仅执行当前 Prompt 明确要求的单一子任务。严禁擅自扩充消融实验、重构无关代码或提前启动未授权的大规模在线评测。
- **训评连续绑定（严禁只训不测）**：凡涉及模型训练，正式 Slurm 作业必须将 **`模型训练 → 权重落盘 → 标准验证/测试集测评出表` 串联为单次连续流水线**。正式提交前先通过 Slurm 用 1～2 个训练 batch 与极小评测样本跑通全链路；Smoke Test 不保留 checkpoint。
- **`expctl` 异步提交与唤醒接力**：长耗时实验统一通过 `/persist_data/home/guisongxia/bin/expctl submit --script <sbatch> --name <名称> --on-success "<说明>" --on-fail "<说明>"` 提交（Subagent 附带 `--submitted-by subagent` 并立即返回摘要；短验证 smoke test 可直接 `sbatch`），严禁阻塞轮询 `squeue`/`sacct`。主 Agent 拿到 `slurm_job_id`/`ticket_id` 后即刻汇报并结束回合；收到 `[EXPCTL_WAKE <event_id>]` 唤醒后先执行 `expctl ack <event_id>` 检查退出状态、日志尾部与指标再继续收尾。
- **卡壳即停与直接暴露真问题**：遇到需求歧义、数据异常，或同一报错连续 **2 次修复尝试**仍未解决时暂停汇报。严禁添加未经要求的兜底回滚补丁，严禁对真值（GT）做人工合成扰动，严禁用旧缓存替代真实运行。
- **探查未知体量的数据目录**：先看条目总数，超过 2000 条的子集严禁全量读图解码，像素级快检一律随机抽样 500 条封顶。

## 3. 算力调度、真值隔离与 Git 卫生
- **Slurm 调度**：耗时实验、GPU 推理、训练与批量评测必须通过 Slurm 提交，禁止阻塞登录节点；查询作业使用 `squeue -u guisongxia` / `sacct -u guisongxia`。
- **Git 卫生（硬约束）**：
  - 严禁把 `outputs/`、`.runtime/`、`third_party/`、`models/`、`datasets/`、图像、掩码、`.npz`、模型权重或 Slurm 日志提交进 Git。
  - 严禁自动脚本执行无过滤的 `git add .` 提交实验产物。
  - 日常以 `main` 为唯一稳定主干；仅在多人/多 Agent 并行改代码时开短期特性分支，合并后立即清理分支和 worktree；纯跑实验不开新分支。
  - 严禁 `git push --force`，严禁在分析、报告和产物中写哈希和 SHA-256，严禁将评测 GT 泄露进 Planner/Verifier 的在线推理上下文。
  - 提交身份：`ethan-404-sys` / `345240447+ethan-404-sys@users.noreply.github.com`；GitHub 操作使用 `/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-git`。

## 4. 文档卫生与产物落盘
- **主文档克制更新**：未经明确授权，严禁向 `docs/CURRENT_STATE.md` 追加实验流水账；严禁重新创建 `task_plan.md`、`findings.md`、`progress.md` 等过程账本。
- **产物落盘**：正式实验数据、权重、评测 JSON、可视化 HTML 与日志统一写入 `outputs/<track>/<YYYYMMDD>_<任务名>/`（日志放其下 `logs/`）；冒烟测试统一放 `.runtime/smoke/<任务名>/`，验证后清理。
- **极简忠实结项**：每次任务闭环输出 **≤ 25 行中文简报**：代码/配置改动、训练收敛与核心指标、产物路径、异常或待决策事项、Subagent 及 Slurm 作业实际状态。

## 5. 主 Agent 与 Codex 后端 / Subagent 协作
- **前后台分工**：前台主 Agent 负责需求对齐、架构边界、拆解窄任务单、验收 diff/门禁/可视化证据并向用户汇报；具体代码编写、数据准备、模型实现与实验提交优先通过全局技能 `codex-backend`（`~/.gemini/config/skills/codex-backend/`）调度 Ethan 的 Codex（`gpt-6.2-sol`，`high` + `fast`，最多 **3 个并发**）执行。
- **交卷门禁**：Codex 后端交卷前必须自跑 `python -B scripts/verify_repo.py` 及任务单指定校验；前台验收通过后再向用户汇报，未通过直接 `resume` 打回修复。
- **阶段汇报**：每次阶段汇报列出本阶段各后端任务 / Subagent 分工及状态与 Slurm 作业状态；未派生时写「本阶段无 Subagent / 后端任务」，无后台作业时写「无运行或排队作业」。
