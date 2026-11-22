# 实验产物目录说明（不入 Git）

所有正式实验产物统一写入 `outputs/<track>/<YYYYMMDD>_<exp_name>/`，内部建议结构：

```text
outputs/<track>/<YYYYMMDD>_<exp_name>/
├── config.yaml      # 当次实验配置快照
├── metrics.json     # 标准评测指标
├── checkpoints/     # 正式权重（smoke test 禁止写入此处）
├── traces/          # 会话轨迹与中间掩码句柄
├── viz/             # 回放 HTML 与可视化图
└── logs/            # Slurm stdout/stderr 与运行日志
```

冒烟测试与临时运行态统一写入 `.runtime/smoke/<exp_name>/`，验证完成后清理。
