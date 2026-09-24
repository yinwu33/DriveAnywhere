# UPSTREAM_PATCHES.md — 对上游 drivestudio 的改动

> 按 AGENTS.md §0 规则 3，每处上游改动逐条记录。
> 状态分三种：`planned`（已规划、尚未实施）、`applied`（已实施，附 commit）、`dropped`（放弃，附原因）。
> 背景见 `docs/DECISIONS.md`。

| ID | 状态 | 文件 | 改动 | 原因 |
|---|---|---|---|---|
| P1 | applied（2026-09-24，尚未提交） | `datasets/waymo/waymo_preprocess.py:68-112`、`datasets/preprocess.py` | `WaymoProcessor` 新增参数 `file_list`，`preprocess.py` 新增 `--waymo_file_list`，默认值都是 `data/waymo_train_list.txt`，上游行为不变；新增 `data/waymo_val_list.txt`（本地 100 个 validation 段，按文件名排序） | D2：开发场景来自 validation |
| P2 | planned | `datasets/driving_dataset.py:79-83` 以及 AABB、scene_radius、帧数统计（`:58-59`） | 允许 `lidar_source` 为 None：跳过 LiDAR 投影；AABB 从配置里的点云文件（`points_fused.ply`）计算；帧数从 `images/` 统计 | C5：非 oracle 实验不得读取 LiDAR |
| P3 | planned | `models/trainers/scene_graph.py:125`（如果使用 SingleTrainer，还有 `models/trainers/single.py:87`） | Background 初始化新增 `from_ply` 分支，输出 `sampled_pts` [N,3] 和 `sampled_color` [N,3]∈[0,1] | C4：用 `points_fused.ply` 初始化 |
| P4 | planned | `models/trainers/base.py:527-531` | 如果 `image_infos` 里有动态掩码，就 `valid_loss_mask *= (1 - dynamic_mask)`，由配置显式开关 | C2：屏蔽动态区域的光度损失 |
| P5 | planned | `tools/train.py:99-103` | 代码备份列表加入 `dashrecon` | 保证实验结果可以复现 |

不需要改上游、放在 `dashrecon/train/` 的部分：
- 估计位姿、逐帧内参、估计深度和掩码的注入：通过 pixel source 子类实现，用 `data.pixel_source.type` 选择（`datasets/driving_dataset.py:148` 用 import_str 加载）。
- 只保留 Background + Sky 的训练配置：放在 `configs/dashrecon/`。
- 训练入口的 GT 读取断言：AGENTS §10 要求加在训练入口，具体实现位置等 P2 落地时再定。
