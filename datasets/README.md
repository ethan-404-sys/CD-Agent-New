# 数据集目录说明（软链接）

本目录中的数据集子目录全部为**软链接**，统一指向共享数据仓 `/guisongxia01/pangchao/wangyihan/datasets/`（其中 `SECOND`、`WHU-CD`、`SYSU-CD` 指向 `/guisongxia01/pangchao/wangyihan/datasets/lim1/` 下的对应入口）。

- 请勿在 Git 中提交任何原始数据或软链接（已在根 `.gitignore` 屏蔽）。
- 新增数据集请先接入 `/guisongxia01/pangchao/wangyihan/datasets/`，再在本目录创建软链接：
  ```bash
  ln -s /guisongxia01/pangchao/wangyihan/datasets/<dataset_name> datasets/<dataset_name>
  ```
