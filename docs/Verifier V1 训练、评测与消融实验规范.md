# Verifier 训练、评测与消融实验规范（SECOND 专项修订版）

> **修订说明**：根据最新组内讨论，本规范由早期“三数据集 600 对 + 冻结 DINOv3 轻量头”正式修订为 **“纯 `SECOND` 单数据集深耕 + 放开视觉编码器微调 SAM/SAM3（注入 Mask 新分支）+ 引入微调模型生成高质量初始掩码”**。

## 1. 实验目的与任务边界

- **核心目标**：以 **SAM / SAM3** 为改造基座，引入初始掩码 $M_{\text{init}}$ 作为新输入分支并**放开特征提取器微调**（适应遥感多类别复杂地物），使其能够结合双时相影像与任务指令，准确识别初始变化掩码中的**误检区域（FP）**与**漏检区域（FN）**，并输出整体质量评分。
- **模型输入**：双时相影像 $(T_1, T_2)$ + 自然英文任务文本 $Q_{\text{en}}$ + 初始变化掩码 $M_{\text{init}}$。
- **预期输出**：
  1. 空间互斥的双通道诊断预测：$M_{\text{init}}$ 内部的误检图 $P_{\text{FP}}$ 与 $M_{\text{init}}$ 外部的漏检图 $P_{\text{FN}}$；
  2. 全局质量评分 $\hat{q}_{\text{score}}$（辅助判断是否停止返工）。
- **本阶段明确简化的部分**：
  - **暂不引入漏检归因分支**：具体漏检时相与补提工具选择先交由上游 Planner 结合高亮区域看图决策；
  - **先不引入独立 Edge 边缘分支**：避免初期被边界标注噪声主导，先打通核心 FP/FN 空间诊断。
- **红线与防作弊约束**：
  - 训练和评测均严格屏蔽 `ignore_mask` 区域，且网络不引入读取原图/掩码高频边缘的浅层旁路；
  - **保留时序方向性**：双期特征交互必须保留有向差异（区分 $T_1 \to T_2$ 顺序，不能只取对称绝对差），确保区分 `into`、`outof` 与 `transition`；
  - **输入与增强红线**：模型只读 `(T1, T2, M_init, prompt_en)`，严禁混入生成器名称字段或 GT 字段（推理入口强制走 `cd_agent.verify.EpisodeInput`）；做图像增强时严禁单独对调 `T1` 和 `T2`。

---

## 2. 数据集策略、规模与隔离原则（纯 SECOND）

### 2.1 聚焦 SECOND（放弃 LEVIR-CD / WHU-CD）
- **放弃 `LEVIR-CD` 与 `WHU-CD`**：二者以单类建筑变化为主，传统“先分割后检测”已有极高 IoU，继续在其上做验证器训练增益有限；现阶段集中全部精力把 **`SECOND`（6 类前景 + 多方向语义转换）** 做透。
- **数据量按可学习参数量定型**：摒弃 600 对底图够用的旧假设。参考微调 SAM / 遥感分割微调相关论文的数据量级规律，结合最终确定的可学习参数量，充分利用 `SECOND` 官方 `train` 全部 **2968 对底图**（排除活跃开发集后）× 多任务构造（`related / into / outof / transition`）× 多源初始掩码 + 反事实配对，建立足够规模的训练集。

### 2.2 高质量初始掩码来源升级
- **双轨初始掩码构造**：
  1. **微调/有监督同域模型（核心提质）**：在 `SECOND` 训练集上引入微调过的变化检测或语义分割模型（如已有 `SCanNet` / `SChanger` 或同域微调模型）生成一批**高质量、中高分档**初始掩码（训练集同源生成无需担心泄露，关键是覆盖真实精修误差分布）；
  2. **免训练开放词汇方法 + 受控扰动（补充多样性与中低分档）**：复用既有免训练方法（`experiments/verifier_init_masks/`）及必要的形态学/任务残差，确保覆盖低/中/高分全难度档与少量零误差真值锚点。
- **严格源图级隔离**（调参仅看 `val`，阶段定型再跑 `test`；通过 `cd_agent.verify.check_split_isolation` 强校验零重叠）：
  - **Train**：从 `SECOND` 官方 `train` 划分（剔除验证/活跃开发底图）；
  - **Val**：独立保留的 `SECOND` 验证底图子集（具体数量与覆盖分布随数据方案一并冻结）；
  - **Test**：独立保留的 `SECOND` 官方测试底图子集（锁死只读）。

### 2.3 关键路径槽位（统一基于仓库相对路径）
```yaml
dataset_root:     "datasets/SECOND"
train_source_ids: "TODO: outputs/verifier_diag/splits/second_train_ids.txt"
val_source_ids:   "TODO: outputs/verifier_diag/splits/second_val_ids.txt"
test_source_ids:  "TODO: outputs/verifier_diag/splits/second_test_ids.txt"

train_manifest:   "TODO: outputs/verifier_diag/data/train/manifest.jsonl"
val_manifest:     "TODO: outputs/verifier_diag/data/val/manifest.jsonl"
test_manifest:    "TODO: outputs/verifier_diag/data/test/manifest.jsonl"

sam1_ckpt:        "models/SAM/sam_vit_h_4b8939.pth"
sam3_ckpt:        "models/sam3/sam3.pt"
dinov3_sat_ckpt:  "models/dinov3-sat"   # 仅作可选对照消融
```

---

## 3. 统一测评指标与验证门禁

所有实验与消融必须调用统一评测脚本（`val` 选阈值固定用于 `test`，严格忽略 `ignore_mask`），且每个正式实验目录必须通过 `cd_agent.verify.verify_run_dir` 校验并调用 `cd_agent.evidence.build_evidence_page` 生成 6–8 个代表性样本的可视化审阅页。

### 3.1 核心表现指标
1. **错漏区域识别 `F1` 与 `IoU`**：
   - **误检识别（`FP-F1 / FP-IoU`）**：在 $M_{\text{init}}$ 有效支持域内衡量误检识别准度；
   - **漏检定位（`FN-F1 / FN-IoU`）**：在掩码外有效背景内衡量漏检定位与召回准度。
2. **模拟修补净增益（`Refined F1 / IoU` 与 `ΔIoU`）**：
   - 剔除高置信 FP、补入高置信 FN 后计算 **`ΔIoU = IoU_after - IoU_init`**，直接检验验证器是正向提分还是负向误伤。

### 3.2 问题诊断指标
1. **反事实翻转率（`CF Flip Rate`）**：同图同掩码切换指令时，真值差异区错漏响应正确翻转的比例。
2. **高分/满分锚点误报率（`Anchor FPR`）**：输入原样真值或高分掩码时触发误报返工的比例。
3. **SECOND 细粒度切片**：
   - 按**任务类型**：`related` / `into` / `outof` / `transition`；
   - 按**初始掩码难度档**：低分粗筛 `[0, 0.3)` / 中段过渡 `[0.3, 0.7)` / 高分精修 `[0.7, 1.0]`；
   - 按**初始掩码来源**：微调高质量掩码 vs 免训练开放词汇掩码。

### 3.3 实验结果统一展示模板

#### （A）主实验与消融总表（SECOND）
| 实验 ID | 模型 / 微调策略简述 | 评测集 | FP-F1 / IoU | FN-F1 / IoU | 模拟净增益 ΔIoU | 反事实翻转率 | 锚点误报率 | 核心现象与下一步判断 |
| :------ | :------------------ | :----: | :---------: | :---------: | :-------------: | :----------: | :--------: | :------------------- |
| `Exp01` | 主线：SAM/SAM3 + Mask 分支微调 | `val` | - / - | - / - | - | - | - | 待完成文献调研与数据构筑后首跑 |
| `Exp02` | 对照：冻结 `DINOv3-Sat` + 轻量头 | `val` | - / - | - / - | - | - | - | 验证放开特征提取器微调的纯收益 |

#### （B）分任务与分难度诊断卡
| 实验 ID | `related` (FP/FN F1) | `into` (FP/FN F1) | `outof` (FP/FN F1) | `transition` (FP/FN F1) | 低分档 ΔIoU | 中分档 ΔIoU | 高分精修档 ΔIoU | 证据页链接 |
| :------ | :------------------: | :---------------: | :----------------: | :---------------------: | :---------: | :---------: | :-------------: | :--------- |
| `Exp01` | - / - | - / - | - / - | - / - | - | - | - | `TODO: viz/index.html` |

---

## 4. 架构与消融备忘池

### 4.1 主路线待定型调研项（下一步优先完成）
- [ ] **SAM 基座选择**：对比 **SAM 1 (`ViT-H`)** vs **SAM 3** 在双时相输入 + Mask 分支注入 + 文本条件融合上的改造成本、显存占用与开放语义能力；
- [ ] **参数微调策略与数据量配比**：结合 SAM 微调论文（全参微调 vs LoRA / Adapter / Image Encoder 部分解冻），确定可学习参数规模与 `SECOND` 有效训练样本量的匹配关系；
- [ ] **高质量初始掩码生成管线**：确定在 `SECOND` 训练集上使用的微调模型（如 `SCanNet` / `SChanger` / 微调分割器）及其输出格式，快速落地第一批高质量训练数据。

### 4.2 可选对照消融
- [ ] **放开编码器微调 vs 冻结视觉编码器（DINOv3-Sat / 冻结 SAM）**：量化验证遥感多类别下微调视觉主干的必要性；
- [ ] **纯免训练初始掩码 vs 引入微调模型高质量初始掩码**：验证高质量数据对中高分精修档 `ΔIoU` 与 `Anchor FPR` 的提升；
- [ ] **反事实配对与排序正则消融**：验证同图换指令样本对 `into / outof / transition` 与 `CF Flip Rate` 的贡献。
