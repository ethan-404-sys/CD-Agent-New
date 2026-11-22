# 模型目录说明（软链接）

本目录中的模型子目录全部为**软链接**，统一指向共享模型仓 `/guisongxia01/pangchao/wangyihan/models/`。

- 请勿在 Git 中提交任何权重文件或软链接（已在根 `.gitignore` 屏蔽）。
- 新增模型权重请先下载到 `/guisongxia01/pangchao/wangyihan/models/<model_name>`，再在本目录创建软链接：
  ```bash
  ln -s /guisongxia01/pangchao/wangyihan/models/<model_name> models/<model_name>
  ```
