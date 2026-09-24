# UPSTREAM_PATCHES.md — 对上游 drivestudio 的改动

> 按 AGENTS.md §0 规则 3，每处上游改动逐条记录。
> 状态分三种：`planned`（已规划、尚未实施）、`applied`（已实施，附 commit）、`dropped`（放弃，附原因）。
> 背景见 `docs/DECISIONS.md`。

| ID | 状态 | 文件 | 改动 | 原因 |
|---|---|---|---|---|
| P1 | applied（2026-09-24，尚未提交） | `datasets/waymo/waymo_preprocess.py:68-112`、`datasets/preprocess.py` | `WaymoProcessor` 新增参数 `file_list`，`preprocess.py` 新增 `--waymo_file_list`，默认值都是 `data/waymo_train_list.txt`，上游行为不变；新增 `data/waymo_val_list.txt`（本地 100 个 validation 段，按文件名排序） | D2：开发场景来自 validation |
| P2 | applied（2026-09-24） | `datasets/driving_dataset.py:57-64, 83-90` | 配置 `data.frames_from_images: True` 时，总帧数改为统计 FRONT 图像数，不再列出 `ego_pose/`；`lidar_source` 可以为 None，此时跳过 LiDAR 投影，场景 AABB 回退到基类 `SceneDataset.get_aabb`，即用相机轨迹计算的包围盒。上游配置下行为不变 | C5，AGENTS §10 第 7 条 |
| P3 | applied（2026-09-24） | `models/trainers/scene_graph.py:124-134, 166-170` | Background 初始化新增 `from_dashrecon` 分支，调用 `dashrecon.train.init.sample_init_points`，可以从 ply 或网格采样；创建完 Gaussian 后，用 `apply_init_geometry` 按三角面设定网格样本的四元数和尺度（D9）。只修改了 MultiTrainer | C4，D9 |
| P4 | applied（2026-09-24） | `models/trainers/base.py:532-536, 559-561` | 配置中出现 `losses.exclude_dynamic` 时，`valid_loss_mask *= (1 - dynamic_masks)`，作用于 rgb、ssim、天空不透明度和深度这四项 loss；深度监督读取的键名可以用 `losses.depth.key` 配置，默认仍是 `lidar_depth_map` | C2、C3 |
| P5 | applied（2026-09-24） | `tools/train.py:101` | 代码备份列表加入 `dashrecon` | 保证实验结果可以复现 |

不需要改上游、放在 `dashrecon/` 的部分：
- `dashrecon/train/pixel_source.py`：`DashreconPixelSource` 和 `DashreconCameraData`，用 `data.pixel_source.type` 选择。负责注入估计的位姿、逐帧内参（像素中心 +0.5 后缩放到加载分辨率）、估计深度（新键 `est_depth_map`）、Grounded-SAM-2 动态掩码和 SegFormer 天空掩码。
- `dashrecon/train/trainer.py`：`DashreconTrainer(MultiTrainer)`，用 `trainer.type` 选择。E5 的网格深度和法向正则在这里实现：nvdiffrast 光栅化网格，再额外渲染一次 Gaussian 法向。
- `configs/dashrecon/static_bg.yaml`：只保留 Background、Sky、Affine、CamPose。
- `dashrecon/train/guard.py` 加 `scripts/train_gs.py`：训练入口的 GT 隔离断言（AGENTS §6 任务 3、§10）。
