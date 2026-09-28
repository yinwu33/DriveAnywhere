# DECISIONS.md — 设计决策与"代码与 AGENTS.md 不一致"的记录

> 按 AGENTS.md §0：代码与 AGENTS.md 冲突时以代码为准，并在此记录。引用行号基于提交 `e59bda4`。

---

## A. 环境与数据核查结果（2026-09-24）

| 项 | AGENTS.md 原假设 | 实际情况 |
|---|---|---|
| GPU | H100 80GB 或 A6000 48GB | 本机为 **1× RTX A6000 48GB**，驱动 555.42（CUDA 12.5），系统 nvcc 12.1 |
| Python | 3.10+ | drivestudio README 用 3.9；代码未使用 3.10+ 语法，3.8–3.10 均可运行 |
| 上游依赖 | — | `requirements.txt` 固定 torch 2.0.0+cu117，与本机 nvcc 12.1 不匹配；`models/gaussians/basics.py:13-14` 依赖 `gsplat.cuda_legacy`，**只能用 gsplat 1.3.0** |
| 数据版本 | "waymo perception 1.4 .5" | `data/data` → `/mnt/disk/data/public/waymo/perception_1_4_3`，即 **v1.4.3** |
| training LiDAR | 默认存在 | 抽样 16/798 个 training tfrecord **均无 `Frame.lasers`**（无 LiDAR），3D 框（laser_labels）存在 |
| validation LiDAR | — | 本地 100/202 个 validation tfrecord **含 LiDAR** |
| scene-flow | drivestudio 文档要求 scene-flow 版本 | 两个 split 都不是 scene-flow 版本（无 `range_image_flow_compressed`） |
| FRONT 标定 | — | 1920×1280，f≈2050，k1≈0.04，**k2≈-0.33**；在 vehicle 系位于 z=2.12m，vehicle 系地面 z≈-0.05 → 离地约 2.12m；卷帘方向 LEFT_TO_RIGHT |

---

## B. 代码与 AGENTS.md 假设不一致之处

**C1. "只训练静态背景"不是一个配置开关**
- `OmegaConf.from_cli` 不能删除 `model:` 下的节点，必须新写一份只含 `Background` + `Sky`（可选 `Affine`、`CamPose`）的 yaml，并设 `load_objects: False`、`load_smpl: False`。
- 所有节点代码都有 `"X" in self.model_config` 的判断（`models/trainers/scene_graph.py:92-109`），删掉节点不会崩。
- `Sky` 是强制的：`sky_model = self.models['Sky']` 没有判断（`models/trainers/scene_graph.py:251`）。
- 天空掩码是强制的：`valid_loss_mask = torch.ones_like(image_infos["sky_masks"])`（`models/trainers/base.py:531`）。
- 处理：新 yaml 放在 `configs/dashrecon/`。

**C2. 上游不支持用动态掩码屏蔽光度损失**
- loss 里只乘了 egocar mask（`models/trainers/base.py:527-534`），Waymo 没有 egocar mask，所以这里是全 1。
- `dynamic_masks` 只在评测指标里用到。
- 处理：打上游钩子，`valid_loss_mask *= (1 - dynamic_mask)`。

**C3. 深度监督只针对 LiDAR 深度图**
- 深度损失读 `image_infos["lidar_depth_map"]`（`models/trainers/base.py:553-554`），是 z-depth。
- `DepthLoss` 只从配置读 `loss_type/normalize/inverse_depth`（`models/trainers/base.py:245-249`）。`max_depth=80` 是默认参数（`models/losses.py:113`），yaml 里改不了，有效深度被截断在 80m 以内（`models/losses.py:123`）。
- `inverse_depth: True, normalize: False, loss_type: l1` 就是逆深度 L1，可以直接用。
- 处理：在 dashrecon 的 pixel source 子类里，把估计深度（MapAnything 的 `depth_z`，也是 z-depth）注入到这个键。

**C4. 不支持用点云文件初始化**
- Background 初始化只有两种：`from_lidar` 或随机（`models/trainers/scene_graph.py:121-131`）。
- `utils/misc.py` 只能写 ply，不能读。
- 处理：在 `scene_graph.py:125` 旁边加一个 `from_ply` 分支（上游钩子）。

**C5. drivestudio 强制读取 LiDAR**
- `datasets/driving_dataset.py:79-80` 断言必须同时有 pixel_source 和 lidar_source。
- `datasets/driving_dataset.py:81` 无条件把 LiDAR 投影到图像。
- 场景 AABB 和 scene_radius 由 LiDAR 计算。
- 帧数靠列举 `ego_pose/` 目录得到（`datasets/driving_dataset.py:58-59`）。
- 这与"非 oracle 不读 LiDAR"冲突。
- 处理：上游补丁，允许没有 lidar_source，AABB 改从 `points_fused.ply` 计算。

**C6. 相机 AABB 回退假设 z 轴向上**
- `datasets/base/pixel_source.py:754-786` 用前视相机轨迹 ±40m 构造 AABB，z 被截在 [-5, 20]，注释写明轴向为"front, left, up"。
- EnvLight 用固定的 `to_opengl` 旋转，同样假设 z 向上（`models/modules.py:186, 194`）。
- 影响：估计位姿必须先转换到"前-左-上"的度量世界系，见 D3。

**C7. 位姿与内参的来源和注入点**
- 上游相机位姿 = `inv(ego_pose[start]) @ ego_pose[t] @ extrinsic @ OPENCV2DATASET`（`datasets/waymo/waymo_sourceloader.py:31-33, 82, 90-98`）。
- 世界系是 start_timestep 时刻的 vehicle 系（前-左-上）；相机系加载后为 OpenCV。与 AGENTS §5 的约定一致。
- `pixel_source.type` 通过 `import_str` 加载（`datasets/driving_dataset.py:148`）。重写 `WaymoCameraData.load_calibrations`（`datasets/waymo/waymo_sourceloader.py:47`）即可注入估计的位姿和逐帧内参，这部分不需要改上游。
- 上游用的是 `frame.pose`（帧时间戳），不是逐图像的 `images[i].pose`，两者相差约一个读出时间（约 45ms）。将来做 GT 位姿评测时要注意。

**C8. 去畸变和 LiDAR 投影用的内参不一致（上游 bug）**
- 图像用 `cv2.undistort(rgb, K, dist)` 去畸变，K 保持不变（`datasets/base/pixel_source.py:249-256`）。
- LiDAR 投影却用 `cv2.getOptimalNewCameraMatrix(..., alpha=1)` 得到的新 K（`datasets/driving_dataset.py:649-658`），所以 LiDAR 深度图和图像是错位的。
- 处理：dashrecon 的 LiDAR 投影自己实现，不复用上游。

**C9. 上游的留出帧协议**
- `test_image_stride=10` 时测试帧为 10, 20, …, 190，不含第 0 帧（`datasets/driving_dataset.py:584-612`），留出是按时间步对所有相机一起留。
- 上游用全部帧（含测试帧）的 LiDAR 做初始化，并用测试帧图像给 LiDAR 点上色（`datasets/driving_dataset.py:695-699`）。这是测试帧泄漏，将来 E0 需要处理，已记入 OPEN_QUESTIONS。

**C10. 不能只评测不训练某个相机**
- 加载进来的相机都会参与训练，训练/测试只按时间步划分。
- 多加载相机会改变 `num_full_images`，进而改变 Affine 和 CamPose 的参数维度。
- novel-view 渲染写死使用 cam0 的内参和尺寸（`datasets/base/pixel_source.py:1083-1089`）。
- `ScenePixelSource.camera_data` 是类级共享的 dict（`datasets/base/pixel_source.py:668`）。
- 处理：跨相机和横向偏移渲染在 dashrecon 里自己构造相机，走 `novel_view=True` 路径。

**C11. 上游的动态掩码与 AGENTS 的定义不同**
- 上游做法：GT 框 8 个角点投影后取**轴对齐 2D 框**，速度阈值 **1.0 m/s**（`datasets/waymo/waymo_preprocess.py:450-451`），并且只保留框内有 LiDAR 点的物体。
- `fine_dynamic_masks` 是 SegFormer 结果与 GT 框相交得到的，仍然依赖 GT；加载时优先使用（`datasets/base/pixel_source.py:198-204`）。
- AGENTS 的 `moving` 模式默认阈值是 0.5 m/s。
- 处理：`dashrecon/masks/gt_boxes.py` 自己实现。非 oracle 实验禁止读取上游的 `dynamic_masks/` 和 `fine_dynamic_masks/`。

**C12. 分辨率与默认模块**
- `configs/datasets/waymo/1cams.yaml` 默认全分辨率 1920×1280（`downscale_when_loading: [1]`）；3cams 和论文用的是 960×640。
- omnire/streetgs 默认开启 `CamPose`（逐图像位姿优化）和 `Affine`（逐图像曝光）。测试帧的 CamPose 没有梯度，用的是未优化的位姿（`models/trainers/base.py:317-340`）。

**C13. 卷帘快门（AGENTS §11 待确认项）**
- 上游完全没有建模：`frame.pose`、逐图像 pose 和 shutter 时间都没有使用。
- Waymo FRONT 的读出方向是 LEFT_TO_RIGHT。

**其他小问题**
- `datasets/dataset_meta.py:20,25` 把侧视相机高度写成 866，实际是 886。本项目只用 FRONT，不受影响。
- `datasets/tools/extract_masks.py` 的天空掩码是 Cityscapes 类 10（`:163`），默认路径 `--segformer_path` 指向原作者的机器。
- `tools/train.py:99-103` 的代码备份不包含 `dashrecon/`。

AGENTS §11 待确认项的答复：
- 单相机配置：`dataset=waymo/1cams`，即 `cameras: [0]`。相机编号 FRONT=0、FRONT_LEFT=1、FRONT_RIGHT=2。
- 关闭动态建模：见 C1。
- 卷帘快门：见 C13。

---

## C. 用户决定（2026-09-24）

| # | 决定 | 影响 |
|---|---|---|
| D1 | **先搭非 oracle pipeline；评测（Phase 1）和 oracle 实验（Phase 2 / E0–E2）延后** | 阶段顺序改为 0 → 3 → 4 → 5 → 6 → 7。评测补上之前，验收改用临时标准（见 E 节） |
| D2 | **重建只用 FRONT 图像；5 个开发场景从本地 validation 中选**，以后用它们的 LiDAR 做评测 | 预处理需要上游补丁 P1（文件列表参数化）；`--process_keys` 不含 `lidar`，这样不需要 scene-flow。以后评测时从 tfrecord 直接读 LiDAR |
| D3 | **非 oracle 实验完全不使用 Waymo 标定**（内参、畸变、外参、相机高度都不用） | 见下方展开 |
| D4 | **CamPose 和 Affine 沿用上游默认（开启）** | 测试帧用的是未优化的位姿；将来 E0 的 GT 位姿也会被优化，不再是纯 GT 位姿 |
| D5 | 动态物体分割先用 **Grounded-SAM-2** | grounding-dino-base/tiny 已在本地 HF 缓存；SAM 3 延后（见 OPEN_QUESTIONS） |
| D6 | 天空和路面分割用 **HF transformers 版 SegFormer-B5 Cityscapes**（`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`） | 一次推理同时得到天空（类 10）和路面（类 0）掩码。与上游 mmcv 版 `segformer.b5.1024x1024.city.160k` 属于同一模型族，但推理实现和前处理可能不同，不能逐像素等同于上游天空掩码 |
| D7 | **保留每 10 帧留出 1 帧**（`test_image_stride: 10`，可配置） | 位姿估计用全部帧，留出帧不进入融合点云、深度监督和光度损失，方便以后补评测时不用重训 |
| D8 | **执行顺序改为：Phase 5 之后先做 Phase 7 的网格重建（NKSR），再做 Phase 6，E4 和 E5 一起训练**（2026-09-24） | 用户参照 LSD-3D 的"先网格、后 3DGS"。保留无网格的 E4 作对照组，所以网格带来的收益仍能量化 |
| D9 | **E5 按 LSD-3D 的几何部分使用网格：在网格面上初始化 Gaussian（朝向与尺度按三角面设定），训练时用网格渲染的深度和法向做正则**（2026-09-24） | 这比 AGENTS 原定的"只做正则"多了网格初始化。LSD-3D 的外观来自扩散模型的分数蒸馏（GGDS），属于 AGENTS §1 非目标和 §10 第 2 条禁止的内容，不采用；外观只来自真实 FRONT 图像的光度损失 |
| D10 | **引入生成式后处理（范围变更，2026-09-24）**：新增 Phase 8 和实验 E6。修改 AGENTS §1 非目标和 §10 第 2 条：在 Phase 8 内允许使用冻结的现成扩散模型，只做推理，不训练、不微调（§2 的"不训练网络"仍然有效） | 用户认为纯 3DGS 在偏离轨迹的视角上失真太严重，要参考 LSD-3D 的 GGDS。LSD-3D 先在 Waymo 上微调了 SDXL 再冻结；我们按 AGENTS 不微调，所以缺少驾驶场景的风格先验，要靠真实训练图像、网格视差条件和较低的噪声等级来约束 |
| D11 | **扩散模型用 SDXL base 1.0 + ControlNet depth（SDXL），条件为 NKSR 网格渲染的视差图**，贴近 GGDS | 权重是 OpenRAIL++ 许可；不使用 Difix |
| D12 | **生成模型的用法：E6 蒸馏进 3DGS 作为主方案；逐帧后处理（记作 E5+pp，渲染后逐帧去噪）作为对照。暂时不做跨相机评测，先把流程搭起来** | 蒸馏后的结果多视角、多帧一致；后处理的结果不写回 3D。评测延后意味着暂时无法区分"修复"和"编造"，结果必须标注为生成内容（见 OPEN_QUESTIONS 28） |
| D13 | **取消 Phase 3 修正的延后（2026-09-25）：相机自标定。**整段视频共用一台相机，用 GLOMAP 同时估内参、径向畸变和位姿，去畸变后，再把内参和 SfM 位姿交给 MapAnything 补稠密深度 | 依据：J 节和文献分析显示，留出帧质量主要由位姿抖动和逐帧内参不一致决定，生成模型补不了。D3 不变：标定仍全部从 FRONT 视频本身估计，不读 Waymo 标定。新实验记为 E4c、E5c（"c" = calibrated），执行记录见 L 节 |
| D14 | **Phase 8 的生成模型改用 NVIDIA Fixer（2026-09-25，研究用途）** | Fixer 是 Difix3D+ 的自动驾驶版：Cosmos-Predict 0.6B，用真实车载数据训练，专门把 3DGS 渲染伪影修成干净图像；576×1024，A100 上约 50 ms/张；NVIDIA Open Model License。E6（SDXL）保留为记录。新实验计划为 E7（E5c + Fixer 蒸馏，只修不确定的区域）和 E5c+fx（逐帧对照） |
| D15 | **现在就做跨相机检查（Phase 1 任务 2 的简化版，2026-09-25）** | 评测代码读 GT。FRONT_LEFT / FRONT_RIGHT 按"每帧估计的 FRONT 位姿 × GT 相机间外参（平移乘 Sim(3) 尺度）"放置；GT 侧前图像用 GT 标定去畸变；指标只在渲染不透明度 > 0.5 且非 GT 动态物体的重叠区上算，先做逐通道仿射颜色对齐。放置方式回答了 OPEN_QUESTIONS 2（逐帧相对外参 vs 全局 Sim(3)）：选逐帧相对外参 |
| D16 | **重建缺失部分（2026-09-25，范围变更）：原视频没拍到的区域用冻结的生成模型补全**，并蒸馏回同一个 3D 模型。修改 AGENTS §1（删去非目标"不重建未被观测的区域"）和 §10 第 2 条；新增 Phase 9 和实验 E8 | 依据：E7（Fixer）和 E5c 的跨相机指标几乎相同，侧前相机看到的主要是前视相机从没见过的区域，修复模型补不了（M 节）。补全内容是编造的，必须标注，并用跨相机检查和目测把关（OPEN_QUESTIONS 28、34） |
| D17 | **目标视角是自由视角，不只是横移**：平移（横向、纵向、升降）和转向（偏航、俯仰，包括侧视和回看） | 视角变化越大，空洞越大，生成越多、一致性越难保证。所以要用 3D 记忆逐步外扩；评测扩展到 SIDE_LEFT / SIDE_RIGHT（偏航约 90°）。偏离范围先用场景单位定义（尺度问题见 OPEN_QUESTIONS 15） |
| D18 | **补全模型：Wan2.1-VACE（视频 + 掩码补全）**。先在本地用 1.3B（Apache-2.0，81 帧 480×832，单卡可跑）打通；14B 可以本地显存卸载运行，或者走托管 API（fal.ai、Replicate、阿里云百炼都有 VACE 14B）。**API 需要用户确认数据外发合规后才用** | 本地跑数据不离开机器。API 会把 Waymo 衍生图像上传到第三方，是否符合 Waymo Open Dataset 许可和公司数据政策，要由用户确认（OPEN_QUESTIONS 36）。GEN3C（自带 3D 缓存，卸载后显存峰值约 43 GB）作为后续对比候选 |
| D19 | **Mapillary 部署放到以后**：先用 Waymo 干净数据打通；Mapillary 的计划（多次通行融合、同设备联合自标定、尺度、停放车辆、许可）记在 AGENTS §14 | — |

D3 的具体影响：
- `undistort: False`，直接在原始畸变图像上按针孔模型重建。FRONT 的 k2≈-0.33，图像边缘会不一致，已列为已知风险。
- 内参只用 MapAnything 的估计值。"使用标定内参"的选项延后。
- 尺度策略默认 `model`。`camera_height` 策略没有可用的高度来源，本阶段不用。
- 重力方向：先用所有帧相机 −y 轴的平均值作为"上"，再用路面掩码内的点拟合平面细化，然后转换到"前-左-上"世界系，以满足 C6 的假设。

---

## D. 未经用户确认、由 agent 采用的默认值（可推翻）

| 项 | 默认值 | 理由 |
|---|---|---|
| 训练分辨率 | 960×640（`downscale_when_loading: [2]`） | 与论文和 3cams 一致，算力约为全分辨率的 1/4 |
| 主训练环境 | Python 3.10 + torch 2.4.1 cu121 + gsplat 1.3.0（源码编译） | 与本机 nvcc 12.1 匹配，同时满足 AGENTS §9 的 3.10+ 要求 |
| 环境工具 | **uv**（用户要求）：`uv python install` 管理解释器版本，`uv venv` 建环境，`uv pip install` 装包，不用 conda | uv 不提供 CUDA toolkit，编译 CUDA 扩展（gsplat、pytorch3d、nvdiffrast）依赖系统的 `/usr/local/cuda-12.1`，因此 torch 必须用 cu121 构建，并加 `--no-build-isolation`。这也是主环境选 torch cu121 的原因之一 |
| 多环境 | 主训练、MapAnything（官方推荐 Python 3.12）、Grounded-SAM-2、NKSR、Waymo 预处理（TF 2.11，要求 Python ≤3.10）各用一个独立的 uv venv | 依赖互相冲突；按 AGENTS §5 用磁盘文件交接 |
| `dashrecon/io.py` 依赖 | 只用 numpy / PIL / json 这类轻量库 | 要能在所有环境中 import |
| MapAnything 权重 | ~~`facebook/map-anything-apache`~~ → 改为 `facebook/map-anything`（CC-BY-NC 4.0），理由见 F 节 | 原先考虑的是许可；实测后 CC-BY-NC 版的内参明显更好 |
| Phase 0 验收 | 改为：各 venv 能 import；gsplat 能编译并完成一次光栅化；5 个场景的 FRONT 图像预处理完成。完整训练的验证移到 Phase 6（P2 落地后） | 原验收"drivestudio 默认配置完整训练"强制读取 LiDAR（C5），而 D2 的开发场景不处理 LiDAR |

---

## E. 临时验收标准（评测补上之前，对应 D1）

每个阶段都要满足：
- 5 个开发场景都能通过命令行跑通；
- `meta.json` 记录耗时和显存峰值；
- 产出对应的定性结果。

| 阶段 | 定性产出 |
|---|---|
| Phase 3 | 位姿轨迹图 |
| Phase 4 | 掩码叠加图和掩掉的像素比例 |
| Phase 5 | 每一步前后的点云截图和点数 |
| Phase 6 | 训练视角的 PSNR（来自训练日志），以及沿原轨迹和横向偏移 0.5 / 1.0 / 2.0 m 的渲染视频（即 AGENTS Phase 1 第 5 项，不需要 GT，提前纳入） |
| Phase 7 | mesh 截图 |

---

## F. Phase 0 / Phase 3 执行记录（2026-09-24）

**开发场景**（`dashrecon/scenes.py`）
- 由 `scripts/scene_stats.py` 在 100 个本地 validation 段上统计后选出，统计结果在 `data/dashrecon/scene_selection/validation_stats.json`。
- 筛选条件：白天、晴天、自车行驶 > 80m。
- 入选场景：
  - static：val056、val039
  - dynamic：val041、val087
  - slope+curve：val094
- GT 标签和位姿只用于选场景。

**预处理**
- 上游补丁 P1 已应用。
- `--process_keys images calib pose dynamic_masks objects`（不含 `lidar`）。
- 输出到 `data/waymo/processed/validation/{056,039,041,087,094}/`，每个场景 197–199 帧 FRONT 图像。

**Phase 0 验收通过**：`main` venv 能 import drivestudio，gsplat 1.3.0 能跑 RGB+ED 光栅化，pytorch3d 可用，nvdiffrast 能 JIT 编译。版本见 `docs/DEPENDENCIES.md`。

**Phase 3（MapAnything）**
- 数据与流程约定：
  - io 约定：`intrinsics.npy` 对应原图 1920×1280 网格，像素中心在整数坐标（与 MapAnything 的 `recover_pinhole_intrinsics_from_ray_directions` 一致）。
  - 深度图保存在 518×336 的推理网格上，网格映射记在 `meta.json["depth_grid"]`，由 `dashrecon.io.depth_intrinsics` 换算内参。
  - Phase 6 接入时，还需确认 drivestudio/gsplat 的像素中心约定（`pixel_source.py` 光线加了 +0.5）。
- 分块推理：197–199 帧一次推理只需 31.5 GB，不需要分块。分块推理（Phase 3 任务 3）尚未实现，帧数超过 `--max_views` 时会显式报错。
- 世界系：用 `dashrecon/pose/world.py` 转成"前-左-上"，上方向取相机 −y 轴的平均。
- **权重选择：改用 `facebook/map-anything`（CC-BY-NC-4.0，研究用途可用）**，推翻 D 节原先的 Apache 默认。依据是在 5 个场景上对两个权重做的诊断（诊断读了 GT，属于评测侧）：
  - CC-BY-NC 版的焦距在全部 5 个场景都更接近真值，且逐帧稳定得多：val056 的 fx 标准差 7 px，Apache 版为 68 px。
  - 抖动方面两者互有胜负：Apache 在 val039、val094 抖动更小，在 val041 更大。
  - 两个权重的对比表见 `OPEN_QUESTIONS.md` 第 14–16 条和 Phase 3 可视化页面。

**可复现的诊断与可视化**（AGENTS §13）
- 两个权重的对比数字由 `scripts/diagnose_pose.py` 生成，写入 `data/dashrecon/diagnostics/phase3_pose.json`。
- Apache 对比 run 在 `data/dashrecon/_checkpoint_compare/map-anything-apache/`。
- 网页模板在 `dashrecon/viewer/index.html`，由 `scripts/build_viewer.py` 组装到 `data/dashrecon/viewer/phase3/`。
- `meta.json` 中的 `dashrecon_commit` 由 `dashrecon/provenance.py` 生成：代码目录有未提交改动时加 `-dirty` 后缀。
- 第一批 Phase 3 结果记录的是 `e59bda4`，但当时代码尚未提交；已从提交 `3d0e64e` 按 §13 的命令全部重跑。

---

## G. Phase 4 执行记录（2026-09-24）

**实现方式**
- Grounded-SAM-2（D5）没有用 IDEA-Research 仓库的脚本，而是用 Hugging Face `transformers` 5.17.0 里的 Grounding DINO 与 SAM 2.1，checkpoint 相同：`IDEA-Research/grounding-dino-base` 和 `facebook/sam2.1-hiera-large`。天空和路面分割（D6）在同一个 venv（`.venvs/masks`）里完成，这样一个库就覆盖了三个模型。
- `facebook/sam2.1-hiera-large` 是 `sam2_video` 类型的 checkpoint，加载进 `Sam2Model` 时 transformers 会给出类型警告。实测 `missing_keys`、`unexpected_keys`、`mismatched_keys` 都为 0，这个警告可以忽略。
- 逐帧独立处理，没有做视频跟踪。先做最简单的版本，漏检造成的帧间闪烁留待 Phase 5 的点云评估来判断影响。transformers 里有 `sam2_video`，需要时可以接入。

**参数**
- 检测分辨率用 Grounding DINO processor 的默认值（短边 800，即 800×1200）。在 1280×1920 全分辨率下，val041 和 val087 的检测数都降为 0。
- 框阈值 `box_threshold` 取 0.25（官方默认 0.35）。在 val041 第 100 帧，阈值 0.3 检出 9 个框，0.25 检出 14 个。漏掉一辆运动车辆会在重建里留下拖影，而多检只会多丢一些静态像素，所以偏向召回。`text_threshold` 取 0.25。
- 文本提示为 car、truck、bus、motorcycle、bicycle、person（AGENTS §3）。掩码膨胀 5 px，结构元素为圆盘。
- 天空取 Cityscapes 类 10，路面取类 0，加载时会对照模型 config 断言类别 id。输入尺寸取 1024×1536：processor 默认的 1024×1024 会把 3:2 的画面压扁。
- 掩码按原图分辨率（1920×1280）保存。Phase 5 用 `dashrecon.io.mask_to_depth_grid` 把掩码映射到深度网格，方法是最近邻缩放加裁剪，与 MapAnything 的预处理一致。

**目录约定**（对 AGENTS §5 的补充）
- 掩码与位姿后端无关，所以单独放在一个 backend tag 目录：`data/dashrecon/<scene_id>/mask-gsam2_sky-segformer/`。目录里有自己的 `frames.txt` 和 `meta.json`（逐帧掩掉的像素比例和检测框数也记在这里）。
- Phase 5 同时读取位姿目录和掩码目录。

**环境**
- `envs/setup_masks.sh`：torch 2.6.0 cu124。transformers 5.x 在 torch < 2.6 时拒绝用 `torch.load` 读取 `.bin` 权重（CVE-2025-32434），SegFormer 的权重正好是 `.bin`。

**已知局限**（Phase 5 评估后再定是否处理）
- 文本分割会把停着的车也掩掉，行为接近 GT 的 `all` 模式，见 OPEN_QUESTIONS 5。
- 远处的小车仍有漏检。
- 逐帧处理，没有利用时间上的一致性。

---

## H. Phase 5 执行记录（2026-09-24）

**输入与输出**
- 输入：Phase 3 目录（深度、位姿、内参、置信度）、Phase 4 目录（掩码），以及 FRONT 图像（用来取颜色）。
- 输出目录为 `data/dashrecon/<scene_id>/<pose_tag>__<mask_tag>/`，因为融合结果同时依赖位姿后端和掩码后端（对 §5 的补充，与 G 节的做法一致）。
- `points_fused.ply` 在 §5 约定的 xyz + rgb + 法向之外，多了一个 uint8 的 `label` 字段（0 = 其他，1 = 路面），供 Phase 6 和可视化使用。读写统一用 `dashrecon.io.read_ply` / `write_ply`。

**用哪些帧**：只用训练帧（D7）。留出帧既不出点，也不作为一致性检查的邻帧，否则测试图像会间接影响哪些点被保留。

**参数**（单位都是位姿 run 的单位，不是米。目前尺度只有真实的 0.39–0.69 倍，见 OPEN_QUESTIONS 15）
- **反投影**：去掉动态像素和天空像素；置信度取每个场景训练帧的第 30 百分位作阈值（MapAnything 的置信度量纲在不同场景间差别很大，1.0–3.3）；深度 < 60。
- **一致性检查**：K = 4，相对误差 5%，至少 2 帧一致（AGENTS 默认值）。邻帧深度取最近像素；邻帧用 MapAnything 自带掩码后的深度，不再额外去掉动态像素。
- **体素平均**：边长 0.05，在 numpy 里实现，这样能保留路面标签（多数投票）；Open3D 的 `voxel_down_sample` 会丢掉逐点属性。
- **统计离群点去除**：Open3D，`nb_neighbors` 20，`std_ratio` 2.0。
- **路面平滑**：
  - 做法：每个网格单元（边长 0.5）取路面点高度的中位数，再用归一化卷积做高斯平滑（σ = 2 个单元，单元内少于 5 个点的不参与），然后用双线性插值得到连续的高度场 z = f(x, y)。
  - 与 AGENTS 的差异：这是平滑后的分段常数场，不是 AGENTS 举例的"分段平面或 B 样条"。
  - 离拟合面超过 0.3 的路面点，视为标错的点（路沿、车底等），保持不动。
  - 前提：路面在重力对齐的世界系里可以写成 z = f(x, y)。开发场景里没有立交桥，所以成立。
- **法向**：Open3D 用 k = 30 个近邻做 PCA，法向朝向最近的训练帧相机中心（AGENTS 任务 5）。

**结果**
- 每个场景 32–42 s，内存约 2 GB。
- 一致性检查剔除的比例依次为 val056 1.6%、val041 2.5%、val087 3.4%、val039 13.5%、val094 24.9%。这个顺序与 Phase 3 的轨迹抖动一致，见 OPEN_QUESTIONS 21。
- 最终点数 1.9M–5.1M。每个场景约 1.3M–1.7M 个路面点被移动，平均 |Δz| 为 0.046–0.076。
- 目测俯视图：val056 的路面能看清黄色双中心线，路面法向一致朝上（70% 的点 nz > 0.8）。

**验收**
- 按临时标准验收：逐步点数记在 meta.json，并在网页里展示。
- AGENTS 原验收"一致性过滤后的几何指标优于未过滤版本"需要 LiDAR 评测，随 Phase 1 延后。
- 消融可以用 `run_fusion.py --skip ...` 生成。

---

## I. Phase 7 网格重建执行记录（2026-09-24，D8）

**输入与输出**
- 输入：`points_fused.ply` 中的 xyz、朝向已统一的法向和颜色。没有另外传每个点对应的相机位置：法向已经按最近相机定过朝向（H 节），满足 AGENTS"提供传感器位置或法向"中的后者。
- 输出：`mesh_nksr.ply` 与 `mesh_nksr.json`，放在融合目录下（§5）。

**NKSR 用法**（源码 commit `e403368`，nksr 1.0.3）
- **尺度**：用 `detail_level = 0.5`（按点密度自适应缩放输入），不用 `detail_level = None`（即按预训练的米制尺度）。原因是我们的点云单位不是米（F 节）。val056 上 NKSR 自动算出的输入缩放系数为 0.644，模型体素 0.1，折算到我们的单位约 0.155。
- **顶点颜色**：用 `field.set_texture_field(fields.PCNNField(xyz, rgb))`，即 NKSR 官方的上色方式。
- **提取网格**：`extract_dual_mesh(mise_iter=1, trim=True)`。内置的 mask field 会裁掉离输入点太远的面。
- **参数**：`approx_kernel_grad=True`、`solver_tol=1e-4`、`fused_mode=True`，照搬官方 Waymo 示例。不分块：每个场景峰值 1.8–4.7 GB。

**结果**
- 面数 1.0M–3.5M，每个场景 3–11 s。
- 路面覆盖率（融合点云中的路面点在 0.15 范围内有网格顶点的比例）：val056 99.8%、val039 98.0%、val041 99.7%、val087 99.8%、val094 96.2%。
- 边界边数：位姿好的场景约 2–3 万条，抖动大的 val039、val094 为 7.6–8.4 万条，洞更多、更碎，与 OPEN_QUESTIONS 21 一致。
- NKSR 在 GPU 上的求解不是逐位可复现的：同一输入两次运行，顶点数差 4 个（835,212 vs 835,208）。

**验收**
- AGENTS 的验收要求是"网格覆盖路面主体，且没有大面积穿洞"。按临时标准判断：覆盖率 ≥ 96%；val056 的着色俯视图目测路面连续平整，只在点稀疏的右侧车道口附近有空洞。
- "E5 与 E4 的对比"要等 Phase 6 训练后才能做。

**显示**
- 网页里的网格用 Open3D 二次误差简化到 15 万个面，完整网格保留在磁盘上。
- 为了把整个网页控制在 Claude Artifact 单版本 64 MB 以内，其余显示抽样也同步下调：原始点云 20 万点、融合点云 20 万点、剔除点 5 万点。

---

## J. Phase 6 执行记录（2026-09-24）

**接入方式**
- **上游改动：**只在 P2–P5 四处加钩子（见 UPSTREAM_PATCHES.md），在上游原有配置下都不生效。
- **其余逻辑放在 `dashrecon/train/`：**
  - `DashreconPixelSource` / `DashreconCameraData`：只读 FRONT 图像和 dashrecon 产物，接口是 `data.pixel_source.type`。
  - `DashreconTrainer(MultiTrainer)`：接口是 `trainer.type`。
- **训练入口 `scripts/train_gs.py`：**先用 `dashrecon.train.guard.assert_non_oracle` 检查合并后的配置，确认不读 LiDAR、GT 框和 `ego_pose/`，只用 FRONT，不去畸变，且所有输入都来自 `data/dashrecon/`。检查通过后才调用上游的 `tools/train.py:main`。
- **输出目录：**`results/<exp>/<scene_id>/`（AGENTS §8），包含：
  - drivestudio 原有的 `config.yaml`、checkpoint、`videos/`、`metrics/`；
  - 我们新增的 `metrics.json`、`meta.json`，以及 `renders/`（横向偏移视频和静帧）。

**约定**
- **像素中心：**dashrecon 的内参按 OpenCV 约定，像素中心在整数坐标；drivestudio / gsplat 的像素中心在 +0.5。所以 cx、cy 先加 0.5，再按加载分辨率缩放。这了结了 F 节留下的待确认项。
- **估计深度：**
  - 新键 `est_depth_map`，从深度网格用最近邻重采样到训练分辨率。
  - 只保留静态、非天空、置信度不低于训练帧第 30 百分位、且深度小于 60 的像素，规则与 Phase 5 相同。
  - 监督方式为逆深度 L1，权重 0.1。这个权重沿用 pvg.yaml 的逆深度设置，没有调过。
- **动态掩码：**`losses.exclude_dynamic` 把动态像素从 rgb、ssim、天空不透明度和深度这四项 loss 中去掉。
- **天空：**用 SegFormer 的天空掩码训练 EnvLight，天空不透明度 loss 权重 0.05（与上游相同）。

**E3 / E4 / E5 的差别**

各实验取 80 万个初始点（与 streetgs 的 LiDAR 采样数相同），另加 10 万近处随机点和 10 万远处随机点。

| 实验 | 初始点来源 | 网格正则 |
|---|---|---|
| E3 | 跳过全部清理步骤的反投影点云 `data/dashrecon/_e3_nocleanup/…`（`run_fusion.py --skip consistency voxel outlier road`，每个场景 940 万–1420 万点） | 无 |
| E4 | Phase 5 融合点云 `points_fused.ply` | 无 |
| E5 | 在 NKSR 网格上按面积均匀采样 | 网格深度 L1（权重 0.1）和法向 1 − cos（权重 0.05） |

E5 的细节：
- **初始化：**每个采样点成为一个扁平 Gaussian。切向尺度取"总面积 / 采样数"对应圆盘的半径，法向尺度取切向的 0.2 倍，使尺度比低于上游 `sharp_shape_reg` 的阈值 10。
- **网格深度：**用 nvdiffrast 在当前相机（已做位姿优化）下光栅化网格，逆深度 L1，权重 0.1。
- **法向：**
  - Gaussian 法向取最小尺度轴，朝向相机，再用 gsplat 渲染一遍；
  - 网格法向为面法向，同样朝向相机；
  - loss 为 1 − cos，只在 Gaussian 不透明度 > 0.5 的像素上计算，权重 0.05。
- **生效范围：**两项网格正则都只在网格覆盖、非动态、非天空的像素上计算。

**其余设置**（沿用上游 streetgs）：30,000 步；分辨率从 1/4 起每 250 步翻倍；CamPose 与 Affine 开启（D4）；每 10 帧留出一帧（D7）。

**验证**
- 在 val056 上分别对 E4 和 E5 做了 600 步冒烟测试，整条链路跑通。
- 网格渲染深度与 MapAnything 深度的中位相对差为 0.9–1.4%；故意把网格深度上下翻转后，差异变为 50.8%。说明 nvdiffrast 的投影约定正确。
- 批量训练：5 个场景 × 3 个实验，GPU 上同时跑两个。

**首轮运行中的问题**：val041 的 E4、E5 在第 795 步因深度 loss 为 NaN 崩溃。原因是末尾帧的估计深度被全部过滤，有效像素为 0（OPEN_QUESTIONS 27）。用上游补丁 P6 修复后，把这两个训练排到批量末尾补跑；失败的输出移到 `results/_failed/` 留档。另外，批量脚本的退出码记录有 bug：`$?` 取到的是同一行里 `$(date)` 的退出码，所以失败也显示为 exit=0。补跑脚本已改正。


**结果（2026-09-25，5 个场景 × E3 / E4 / E5 全部完成并渲染）**

指标是 drivestudio 在留出帧上的指标（每 10 帧留 1 帧，全图），格式为 PSNR / SSIM / LPIPS。"全序列 PSNR"是所有帧的均值，大部分是训练帧。

| 场景 | E3 | E4 | E5 |
|---|---|---|---|
| val056 | 27.40 / 0.824 / 0.156 | 27.42 / 0.823 / 0.155 | 27.41 / 0.823 / 0.163 |
| val039 | 16.94 / 0.564 / 0.454 | 16.81 / 0.564 / 0.452 | 16.97 / 0.569 / 0.445 |
| val041 | 28.63 / 0.890 / 0.227 | 28.59 / 0.889 / 0.226 | 28.42 / 0.889 / 0.242 |
| val087 | 21.22 / 0.722 / 0.316 | 21.23 / 0.723 / 0.313 | 21.35 / 0.726 / 0.323 |
| val094 | 16.57 / 0.574 / 0.560 | 16.59 / 0.574 / 0.560 | 17.89 / 0.588 / 0.541 |
| 5 场景均值 | 22.15 / 0.715 / 0.342 | 22.13 / 0.715 / 0.341 | 22.41 / 0.719 / 0.343 |
| 全序列 PSNR 均值 | 25.22 | 25.21 | 25.03 |
| 训练时间均值 | 79 min | 81 min | 94 min |

- 训练时间是 2–3 个任务共用 GPU 时测的，只能粗略比较。显存峰值约 10 GB。
- **E3 与 E4 几乎一样**：各场景 PSNR 相差不超过 0.13 dB。Phase 5 的清理（一致性过滤、降采样、去离群点、路面平滑）对留出帧的插值指标没有可见作用。推测 30,000 步的致密化和剔除抹平了初始化的差别。横移视角上是否有差别，只能靠对比拼图目测，要等跨相机评测才能定量。
- **E5 均值高 0.26 dB，主要来自 val094（+1.3 dB）**；val041 低 0.2 dB，val056、val041、val087 的 LPIPS 略差。E5 的全序列 PSNR 最低（25.03），说明网格正则牺牲了一点训练帧的拟合，换来 val094 这类位姿差的场景在留出帧上更好。
- **留出帧 PSNR 主要由位姿质量决定**：抖动大的 val039、val094 只有 16–18 dB，val056、val041 有 27–28 dB（OPEN_QUESTIONS 16）。
- **代码版本**：各运行的 `dashrecon_commit` 见各自的 `meta.json`。E4 / E5 val056 记为 `6ff0bbc-dirty`：训练开始时（15:50），网页模板以及渲染、可视化、组装网页的 4 个脚本还没提交，一分钟后提交为 c091046。训练日志目录里的代码备份（P5）与 6ff0bbc 逐文件一致；`scripts/train_gs.py` 在 6ff0bbc 与 c091046 之间没有改动。所以这两个结果可以用 6ff0bbc 复现，没有重跑。其余训练代码此后只改过 P6，它只影响"某帧没有有效深度"的情况，而这些运行没有碰到。
---

## K. Phase 8 执行记录（2026-09-24，D10–D12）

**生成器 `dashrecon/gen/ggds.py`（`SDXLRefiner`）**
- **权重：**SDXL base 1.0（fp16 UNet）+ `controlnet-depth-sdxl-1.0`（fp16）+ `sdxl-vae-fp16-fix`。两个文本编码器只在启动时把提示词编码一次，然后释放。
- **流程**（照 GGDS 的生成步骤）：
  1. 渲染图放大到 1248×832。SDXL 按约 1 MP 训练，这个尺寸的宽高比 1.5 与 960×640 相同。
  2. VAE 编码，取后验均值。
  3. DDIM 反演到 T = round(t·999)：5 步，只用条件分支。
  4. DDIM 去噪 5 步，CFG 3.0。
  5. 解码，缩回渲染分辨率。

  整个过程是确定性的，不采样噪声。
- **ControlNet 条件：**当前视角 NKSR 网格的视差 1/z，除以有效像素视差的第 90 百分位后截断到 [0, 1]。天空和网格没覆盖的像素为 0。条件强度 0.8。
  - 第一版用 min–max 归一化到第 99 百分位，远处的树和房子几乎全黑，所以改成现在的做法。
- **提示词：**与场景无关，部署到 Mapillary 时不用改。
  - 正面："a dashcam photo of a street, realistic, sharp, highly detailed"
  - 负面："blurry, smeared, low quality, distorted, artifacts, painting, cartoon"
- **噪声等级：**在 val056 帧 101、横移 0 / 1 / 2 m 上测试：
  - t = 0.3：输出几乎等于输入；
  - t = 0.5–0.7：车道线、树冠明显变锐利；
  - t = 0.85：开始改内容，黄色双实线变成了白线。

  所以 t 的上限取 0.7。
- **速度：**与两个训练任务共享 GPU 时，每张约 4 s。

**E6（`scripts/train_ggds.py`）的做法，以及与 GGDS 的差别**
- **续训：**从 E5 的 `checkpoint_final.pth` 接着训 6000 步（第 30001–36000 步）。每步包含两部分：
  - 一个真实训练帧上 E5 的全部 loss；
  - 池中一个新视角上的生成 loss 和网格 loss。
- **新视角池：**
  - 每 4 个训练帧取 1 个，约 45 个视角；留出帧不进池。
  - 每个视角在该帧优化后的位姿上横移 ±U(0.5, 2.5)，并偏航 U(−3°, 3°)。
- **目标的生成方式（与 GGDS 不同）：**
  - 每 1000 步把整个池的目标重新生成一次，即迭代式数据集更新。GGDS 每步都重新生成，这里每步要 4 s，太贵。
  - 第 r 轮取 t ~ U(0.3, t_max(r))，t_max 从 0.7 线性退火到 0.4。
  - 权重 w(t) = sqrt(ᾱ)，取起始时间步的值。这是我们选的，论文里的 ω(t) 没有照搬。
- **生成 loss：**gen_w · w(t) · (L1 + 0.5 · LPIPS-VGG)。
  - 只在网格覆盖、或 Gaussian 不透明度 > 0.5 的像素上计算；其余像素的目标换成渲染图本身（detach）。
  - 新视角不做 CamPose；前向时冻结 Affine 和 Sky，所以生成目标只更新 Gaussian。
- **新视角上的网格 loss：**权重与 E5 相同，只在网格覆盖的像素上计算。
- **对 E5 配置的改动：**
  - 关掉不透明度重置（`reset_alpha_interval = 1e9`）。否则上游在第 30100 和 33100 步会无条件重置。
  - 学习率线性预热 500 步。drivestudio 的 checkpoint 不存优化器状态；全新的 Adam 在最初几步会把每个参数都挪动约一个学习率，而且第一步还会用未经调度的初始 xyz 学习率。

**冒烟测试**
- 设置：val056，300 步，每轮 6 个视角，预热 100 步。
- GPU 与 Phase 6 批量训练共享，约 2.3 it/s。
- 指标是 drivestudio 在留出帧上的指标，最后一行是非天空像素的 PSNR。

| | E5（起点） | 续训，gen_w = 0（对照） | gen_w = 0.5，不遮天空 | gen_w = 0.5，遮天空并冻结 Sky | E5+pp，t = 0.6（逐帧，不训练） |
|---|---|---|---|---|---|
| PSNR | 27.41 | 27.64 | 27.31 | 27.58 | 25.59 |
| SSIM | 0.823 | 0.831 | 0.798 | 0.829 | 0.792 |
| LPIPS | 0.163 | 0.163 | 0.281 | 0.170 | 0.264 |
| 非天空 PSNR | 25.84 | 26.08 | 26.02 | 26.07 | 24.58 |

- **不遮天空的问题：**SDXL 把淡淡的云纹放大成粗条纹，再经由所有视角共享的 Sky 模型传到每一帧。结果测试帧 LPIPS 从 0.163 升到 0.281，而非天空 PSNR 基本不变。遮掉天空、冻结 Sky 后，LPIPS 回到 0.170。
- **正式运行的 gen_w 取 1.0：**遮掉约 1/3 的像素后，loss 大约减半。这是推断，没有在全长运行上验证过；先跑 val056 全长，确认后再跑其余场景。

**E5+pp（`scripts/postprocess_frames.py`）**
- 按 `render_lateral.py` 的方式渲染 E5，然后每帧独立地用 `SDXLRefiner`（t = 0.6）细化。不训练，也不写回 3D。
- 测试帧指标由 drivestudio 的 `render_images` 计算：用代理 trainer 把 rgb 换成细化后的图像，所以计算方法与 E3–E6 完全相同。
- 只算 test，不算 full。full 每个场景约要 15 min。

**评测**：按 D12，暂不做跨相机评测。上表的测试帧指标只能说明，生成内容有没有破坏训练轨迹附近的保真度；不能说明偏移视角是否被"修对了"（OPEN_QUESTIONS 28）。

**结果（2026-09-25，5 个场景的 E6 和 E5+pp 全部完成）**

指标是 drivestudio 在留出帧上的指标，格式为 PSNR / SSIM / LPIPS。留出帧都在原轨迹上，所以这里只能看出生成内容有没有破坏轨迹附近的保真度。

| 场景 | E5 | E6 | E5+pp |
|---|---|---|---|
| val056 | 27.41 / 0.823 / 0.163 | 27.20 / 0.818 / 0.203 | 25.59 / 0.792 / 0.264 |
| val039 | 16.97 / 0.569 / 0.445 | 16.95 / 0.576 / 0.476 | 16.61 / 0.553 / 0.495 |
| val041 | 28.42 / 0.889 / 0.242 | 27.77 / 0.881 / 0.253 | 26.39 / 0.865 / 0.267 |
| val087 | 21.35 / 0.726 / 0.323 | 21.35 / 0.724 / 0.357 | 20.30 / 0.696 / 0.374 |
| val094 | 17.89 / 0.588 / 0.541 | 17.82 / 0.591 / 0.553 | 17.59 / 0.579 / 0.570 |
| 5 场景均值 | 22.41 / 0.719 / 0.343 | 22.22 / 0.718 / 0.368 | 21.30 / 0.697 / 0.394 |
| 非天空 PSNR 均值 | 21.96 | 21.79 | 20.96 |

- **运行开销：**
  - E6 第二阶段每个场景 52–69 min，其中生成 16–23 min；显存峰值 18.4–19.7 GB。
  - E5+pp 每个场景 15–17 min，包括测试帧、12 张静帧和一段横移 1 的视频；显存峰值约 13 GB。
  - 都是和其他任务共用 GPU 时测的。
- **轨迹附近：**
  - E6 比 E5 略差：PSNR 均值 −0.19 dB，val041 最多（−0.65 dB）；LPIPS +0.025。
  - E5+pp 更差：PSNR −1.1 dB，LPIPS +0.05。
  - 逐帧后处理对保真度的损害比蒸馏大：PSNR 的下降约为 E6 的 6 倍，LPIPS 的上升约为 2 倍。
- **横移视角（目测对比拼图和网页 3DGS 模式，没有定量）：**
  - val056 横移 1–2 m：E6 的路面暗斑和拖影明显减少，车道虚线更成形，人行道、树篱更清楚；树冠有涂抹感。
  - val041（雨天、车多的高速）：E6 减少了车辆拖影，但整体偏雾偏暗。E5+pp 路面更干净，却凭空画出了深色斑块。
  - 没有跨相机评测，这些"更清晰"的内容对不对无法验证（OPEN_QUESTIONS 28）。
- **网页：**3DGS 模式可以在 E5 / E6 之间切换，视角不动；每个实验每个场景显示 8 万个 Gaussian。另外，对比拼图多了 E5+pp、E6 两列，并附有每个场景第 0 轮和第 5 轮的 SDXL 目标样例。
- **代码版本：**
  - E6 / E5+pp 各运行记录的 commit 为 493fb1d、ca6de9f、fba0d27、68b9f08，都不带 `-dirty`。
  - 这几个 commit 之间，`dashrecon/gen`、`dashrecon/train`、`scripts/{train_ggds,postprocess_frames,render_lateral,train_gs}.py` 以及上游目录没有任何改动（`git diff` 为空）。
- **事后修正（2325562）：**
  - 原来的行为：`disparity_image` 在视角里没有网格像素时，静默返回全零的条件图，违反 AGENTS 第 0 节第 7 条。
  - 现在改为报错。
  - 上面所有运行用的都是旧行为。没有记录是否真的遇到过这种视角；新视角都取自训练帧附近，前方总有路面网格，推测没有遇到。

---

## L. Phase 3 修正：相机自标定执行记录（2026-09-25，D13）

**探索（val039，GT 只用于诊断）**
- 增量式 SfM（COLMAP mapper）在前向行驶视频上初始化失败：199 帧只注册 2 帧，畸变参数发散（RADIAL 的 k1 = 6.1；SIMPLE_RADIAL 的 f = 16223）。
- 全局式 SfM（GLOMAP，pycolmap 4.2 的 `global_mapping`）注册了全部帧，下面用它。
- 在 GLOMAP 前先跑 `calibrate_view_graph`（从基础矩阵估焦距）：val056、val087 的焦距与不跑时相同到 0.3%，不采用。
- MapAnything 在去畸变图像上的两种输入（val039）：
  - 只给内参：焦距对了，但轨迹仍然抖动（路径长 / 起终点距离 1.84），Sim(3) 轨迹误差 8.65 m，尺度为真值的 0.72。
  - 给内参和 SfM 位姿（非度量）：轨迹平滑（1.007），但输出位姿与 SfM 位姿相差 0.37 单位（Sim(3) 残差），轨迹误差 0.74 m（SfM 自身 0.09 m），尺度为真值的 0.51。
  - 所以：位姿直接用 SfM；深度用"内参 + 位姿"输入下的 MapAnything 输出；尺度取所有稀疏观测上"MapAnything 深度 / SfM 深度"的中位数。

**设置**（`scripts/run_calib.py`、`dashrecon/pose/calib.py`）
- 原始（畸变）FRONT 图像，SIFT 每张 8192 个特征，Phase 4 的动态和天空像素上不提特征。
- 序列匹配：相邻 20 帧加平方间隔，不做回环检测。
- 所有帧共用一台 COLMAP RADIAL 相机（f、k1、k2），主点固定在图像中心。GLOMAP 随机种子 0。
- 去畸变：用同一个 K，同样大小。畸变是桶形，去畸变后几乎所有像素都采样自原图内部；val094 的上下边缘有 1.4 px 越界，用边缘像素复制填充，允许最多 2 px，越界量写进 meta。
- 去畸变后的图像按 processed 目录结构放在 `data/dashrecon/_undistorted/calib-glomap/<idx>/images/<t>_0.jpg`。下游直接把它当作 `--processed_root`，训练时是 `data.data_root`。这个目录里只有前视图像，误读其他文件会直接报错；guard 只允许原始 split 或这个目录作为图像来源。

**自标定结果**（正式跑，commit 599eacb；GT 只用于诊断，`data/dashrecon/diagnostics/phase3_pose_glomap.json`）

| 场景 | 注册帧 | f 估计 / GT | k1, k2（GT 约 0.04–0.05, −0.33 至 −0.35） | 路径长 / 起终点距离（GT） | 重投影误差 | 尺度 估计 / GT | 原先 MapAnything：f 比值 / 抖动 / 尺度 |
|---|---|---|---|---|---|---|---|
| val056 | 197/197 | 0.860 | −0.022, −0.135 | 1.001 (1.001) | 0.41 px | 0.43 | 0.54 / 1.07 / 0.50 |
| val039 | 199/199 | 0.984 | 0.025, −0.268 | 1.000 (1.000) | 0.45 px | 0.58 | 0.71 / 1.86 / 0.63 |
| val041 | 198/198 | 0.955 | −0.003, −0.215 | 1.000 (1.000) | 0.43 px | 0.49 | 0.55 / 1.24 / 0.39 |
| val087 | 198/198 | 0.923 | −0.021, −0.188 | 1.001 (1.001) | 0.52 px | 0.66 | 0.74 / 1.09 / 0.65 |
| val094 | 198/198 | 1.041 | 0.040, −0.317 | 1.011 (1.017) | 0.44 px | 0.79 | 0.88 / 2.48 / 0.69 |

- 探索跑（同样设置）另外算了 Sim(3) 对齐后的轨迹误差：3–24 cm，轨迹长 162–219 m。
- **MapAnything 深度与 SfM 的一致性：**所有稀疏观测上"MapAnything 深度 / SfM 深度"的比值，偏离中位数的中位值是 5.7–8.9%。MapAnything 自己预测的焦距与给定内参相差 1–3%，主点相差 7–9 px（在 518×336 的深度网格上）；所以只用它的深度，内参和位姿仍用 SfM 的。
- **尺度：**MapAnything 深度给出的度量尺度仍然偏小（真值的 0.43–0.79），见 OPEN_QUESTIONS 15。
- 轨迹抖动完全消失。
- 焦距误差从 12–46% 降到 4–14%。畸变 k2 只估出真值的 40–95%：在直路上，焦距和径向畸变可以互相抵消一部分（val056 最明显）。GLOMAP 即使固定随机种子也不完全确定，val094 三次运行的焦距在 2011–2145 之间。见 OPEN_QUESTIONS 33。

**原流水线的跨相机检查（D15，`scripts/eval_cross_camera.py`，每 5 帧一次，FRONT_LEFT + FRONT_RIGHT，共 80 张/场景）**

格式为重叠区覆盖率 / PSNR（颜色对齐后）/ SSIM / LPIPS：

| 场景 | E3 | E4 | E5 | E6（SDXL 蒸馏） |
|---|---|---|---|---|
| val056 | 0.84 / 15.39 / 0.561 / 0.650 | 0.85 / 15.36 / 0.561 / 0.655 | 0.84 / 15.43 / 0.562 / 0.646 | 0.85 / 15.35 / 0.567 / 0.673 |
| val039 | 0.97 / 13.63 / 0.441 / 0.798 | 0.97 / 13.59 / 0.440 / 0.792 | 0.98 / 13.56 / 0.440 / 0.795 | 0.99 / 13.52 / 0.443 / 0.819 |
| val041 | 0.60 / 17.98 / 0.718 / 0.413 | 0.61 / 17.26 / 0.714 / 0.424 | 0.61 / 17.54 / 0.719 / 0.410 | 0.62 / 17.49 / 0.712 / 0.423 |
| val087 | 0.68 / 14.39 / 0.498 / 0.544 | 0.70 / 14.41 / 0.495 / 0.552 | 0.68 / 14.29 / 0.499 / 0.535 | 0.73 / 14.04 / 0.498 / 0.589 |
| val094 | 0.89 / 16.24 / 0.502 / 0.706 | 0.90 / 16.30 / 0.502 / 0.702 | 0.90 / 16.34 / 0.499 / 0.692 | 0.91 / 16.16 / 0.501 / 0.722 |

- 侧前视角的质量很差，而且四个实验几乎没有差别：初始化、网格正则、SDXL 蒸馏都改变不了它。E6 在 val087、val094 上的 LPIPS 还略差。这支持 D13 的判断：瓶颈在相机几何。
- 目测 val056 帧 100 的 FRONT_LEFT：车道线的位置和方向与真实图像大致吻合（相机放置正确），但房子和树几乎认不出来。

**E5c 结果（2026-09-25，5 个场景）**

左侧是留出帧（PSNR / SSIM / LPIPS），右侧是跨相机检查（颜色对齐后的 PSNR / SSIM / LPIPS，重叠区覆盖率）。E5c 的留出帧对照的是去畸变后的真值图，E5 对照的是原始畸变图（E5 要用针孔模型去拟合畸变图像，这本身是它的短板之一）。

| 场景 | E5 留出帧 | E5c 留出帧 | E5 跨相机 | E5c 跨相机 |
|---|---|---|---|---|
| val056 | 27.41 / 0.823 / 0.163 | **31.39 / 0.910 / 0.101** | 15.43 / 0.562 / 0.646, 0.84 | **16.43 / 0.543 / 0.589, 0.82** |
| val039 | 16.97 / 0.569 / 0.445 | **20.30 / 0.787 / 0.286** | 13.56 / 0.440 / 0.795, 0.98 | **14.13 / 0.440 / 0.727, 0.94** |
| val041 | 28.42 / 0.889 / 0.242 | **30.21 / 0.933 / 0.183** | 17.54 / 0.719 / 0.410, 0.61 | **18.00 / 0.728 / 0.381, 0.58** |
| val087 | 21.35 / 0.726 / 0.323 | **23.96 / 0.848 / 0.219** | 14.29 / 0.499 / 0.535, 0.68 | **15.50 / 0.507 / 0.462, 0.64** |
| val094 | 17.89 / 0.588 / 0.541 | **25.11 / 0.817 / 0.260** | 16.34 / 0.499 / 0.692, 0.90 | **17.53 / 0.506 / 0.595, 0.88** |
| 均值 | 22.41 / 0.719 / 0.343 | **26.20 / 0.859 / 0.210** | 15.43 / 0.544 / 0.616, 0.80 | **16.32 / 0.545 / 0.551, 0.77** |

- **留出帧：**所有场景都明显提升，均值 +3.8 dB，LPIPS 0.343 → 0.210。原来位姿最差的 val094 提升 7.2 dB，val039 提升 3.3 dB。原来最好的 val056 也提升了 4.0 dB，这部分应主要来自去畸变和共享内参。
- **跨相机：**也有改善，均值 PSNR +0.9 dB，LPIPS 0.616 → 0.551，但幅度远小于留出帧。侧前相机看到的很多是前视相机从没见过的区域，相机修好了也补不出来（OPEN_QUESTIONS 35）。
- **训练时间：**84–129 min，GPU 与其他任务共用。commit：141bed8, d40dd12, ea1fb16，都不带 -dirty。
- **E7（Fixer 蒸馏）的前 4 个场景：**相对 E5c，留出帧基本不变（−0.5 至 +0.2 dB）；跨相机 LPIPS 只下降 0.00–0.02，只有 val041 的 PSNR 提高了 0.7 dB。Fixer 能修轨迹附近的伪影，但侧前视角的主要问题是未观测区域，它补不了。完整结果等 val087 跑完补上。

---

## M. Phase 8 换用 Fixer（2026-09-25，D14）

**环境**（`envs/setup_fixer.sh`，`.venvs/fixer`）
- 官方只支持 NGC 容器 `cosmos-predict2-container:1.2`，匿名拉取被拒（需要 NGC 登录），所以改用 uv 搭环境：torch 2.6 cu124（驱动 555 可用）；cosmos-predict2 1.0.9 和 megatron-core 0.10.0 都用 `--no-deps` 安装（与官方 Dockerfile 相同；它们的训练依赖 tensorstore 在这里编不过）。
- flash-attn 用官方预编译包：Qwen 文本编码器模块在导入时断言它存在，但 Fixer 并不运行文本编码器。transformer_engine 2.2.0 用系统 CUDA 12.1 针对这个 torch 编译。
- 运行时要把 venv 里 `nvidia/*/lib` 加进 `LD_LIBRARY_PATH`（transformer_engine 通过 ctypes 加载 cuDNN）；`fixer_client.py` 启动 worker 时会设好。
- Fixer 源码 nv-tlabs/Fixer @ b39dfca（Apache-2.0）；权重 `nvidia/Fixer` @ ca20a25（NVIDIA Open Model License，不需要登录）。

**推理**
- VAE 编码（不加噪声），DiT 在 σ = 250/1000 上一步去噪，再解码；输入 RGB 归一化到 [−1, 1]，bf16。
- A6000 上 960×640 每张约 0.12 s，显存峰值 2.2 GB。SDXL 方案是每张约 4 s、14 GB。
- 直接用渲染的 960×640 作输入：长宽都是 16 的倍数，和默认的 1024×576 效果相近，而且不改变宽高比。
- 效果（E5 在 val056、val041 上横移 1–2 m 的渲染）：路面拖影和暗斑被清掉，车道线变清楚；被掩掉车辆留下的鬼影变成了看起来合理的停放车辆（编造的内容）；有一处右侧虚线被修成了实线（语义改变）。

**与训练进程的交接**：Fixer 需要 torch 2.6 + cosmos，drivestudio 训练用 torch 2.4.1 + gsplat 1.3.0，不能放进同一个 venv。`dashrecon/gen/fixer_worker.py` 作为常驻子进程运行在 `.venvs/fixer` 里，与训练进程通过 stdin/stdout 上的 JSON 行和 .npy 文件交换图像（AGENTS §9：模块之间只通过磁盘文件交接）。

**E7（`scripts/train_fixer.py`）**：从 E5c 续训 6000 步，照 Difix3D+ 的渐进式做法。
- 每 250 步重建一次新视角池：每 4 个训练帧取 1 个，横移范围从 [0.25, 0.75] 线性扩大到 [1.5, 2.5] 个场景单位，偏航不超过 3°。
- 用当前模型渲染这些视角，交给 Fixer 修复，修复结果作为目标。
- 其余与 E6 相同：只在网格覆盖或不透明的像素上算，冻结 Sky 和 Affine，学习率预热，关闭不透明度重置。
- loss 权重：渲染与 Fixer 目标之间的 LPIPS-VGG 约 0.15，L1 约 0.015（val056 冒烟测试）。所以取 gen_w 0.5、lpips_w 0.2，让生成 loss（约 0.02）与真实帧的 rgb + ssim loss（约 0.025）相当。

**E5c+fx**（`scripts/postprocess_frames.py --generator fixer --init_exp E5c`）：逐帧对照，与 E5+pp 的做法相同。

**结果（2026-09-25，5 个场景）**

留出帧格式为 PSNR / SSIM / LPIPS（对照去畸变后的真值图），跨相机格式为 PSNR / LPIPS。E5c+fx 是逐帧后处理，没有自己的 3D 模型，所以没有跨相机结果。

| 场景 | E5c 留出帧 | E7 留出帧 | E5c+fx 留出帧 | E5c 跨相机 | E7 跨相机 |
|---|---|---|---|---|---|
| val056 | 31.39 / 0.910 / 0.101 | 31.63 / 0.923 / 0.104 | 30.98 / 0.899 / 0.084 | 16.43 / 0.589 | 16.43 / 0.582 |
| val039 | 20.30 / 0.787 / 0.286 | 20.19 / 0.788 / 0.286 | 20.37 / 0.771 / 0.266 | 14.13 / 0.727 | 14.14 / 0.716 |
| val041 | 30.21 / 0.933 / 0.183 | 29.71 / 0.924 / 0.183 | 29.86 / 0.927 / 0.152 | 18.00 / 0.381 | 18.67 / 0.361 |
| val087 | 23.96 / 0.848 / 0.219 | 24.33 / 0.853 / 0.214 | 24.12 / 0.843 / 0.194 | 15.50 / 0.462 | 14.79 / 0.489 |
| val094 | 25.11 / 0.817 / 0.260 | 24.64 / 0.823 / 0.240 | 25.24 / 0.809 / 0.220 | 17.53 / 0.595 | 17.24 / 0.588 |
| 均值 | 26.20 / 0.859 / 0.210 | 26.10 / 0.862 / 0.205 | 26.11 / 0.849 / 0.183 | 16.32 / 0.551 | 16.25 / 0.547 |

- **E7 与 E5c 基本持平：**留出帧 -0.10 dB，跨相机 LPIPS 0.551 → 0.547；各场景有正有负，val087 的跨相机变差（0.462 → 0.489）。Fixer 蒸馏在轨迹附近修掉了一些伪影，但侧前视角的主要问题是未观测区域，它补不了（OPEN_QUESTIONS 35）。
- **E5c+fx（逐帧 Fixer）：**留出帧 PSNR 基本不变（-0.08 dB），LPIPS 在 5 个场景都下降（均值 0.210 → 0.183），每个场景 1–2 min。这与 SDXL 的逐帧后处理 E5+pp 相反（E5+pp 让 PSNR 降了 1.1 dB）。
- **编造的内容：**val039 这条街停满了车，训练时全被掩掉（OPEN_QUESTIONS 18），E5c 里是模糊的车影；Fixer 把它们修成了干净的车，但颜色是编的（真值中的红车和深色车变成了蓝车，见网页 val039 的对比拼图）。
- **网页：**自标定页面 https://claude.ai/artifact/SzEh7vBbofCdSWA2nQr3sC Version 2 新增了 E5c / E7 的 3DGS 切换、E5c / E5c+fx / E7 的对比拼图、E7 的 Fixer 目标样例（渲染 | 目标）和跨相机表。

---

## N. Phase 9：未观测区域的生成式补全（2026-09-25，D16–D18，E8）

用户要求（2026-09-25）：先只在 val056 上迭代，效果满意后再扩展到 5 个场景。val039 只用来打通流程。

**环境**（`envs/setup_wan.sh`，`.venvs/wan`）：torch 2.6 cu124、diffusers 0.40.0。
- 权重：`Wan-AI/Wan2.1-VACE-1.3B-diffusers`（补洞）；`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ 0fad780 的 transformer（整帧重绘）。都是 Apache-2.0。
- T2V 的文本编码器与 VACE 的是同一个 UMT5-XXL（只是存成 fp32），VAE 权重完全相同（按文件哈希和 config 核对过），所以只额外下载 5.7 GB 的 transformer。

**流程**（每个相机运动一轮，`scripts/run_phase9.sh`）：
1. `render_views.py`（主 venv）：把每帧训练好的 FRONT 相机（含 CamPose 修正）按 `dashrecon.gen.views.ViewMove` 移动和转向，渲染 RGB、深度和空洞掩码。
   - 空洞 = 没有训练视角在相近分辨率下看到过的像素，天空除外。判定方法：把渲染深度反投影成点，投到每隔一帧的训练视角里，要求落在该视角没被动态或天空掩掉的像素上、与该视角渲染深度的相对误差 < 10%、距离不超过 3 倍。然后做开运算、去掉小连通块、再膨胀。
   - 不用不透明度判空洞：未观测区域里残留着训练没约束过的 Gaussian，渲染出来是不透明的。val039 上，按不透明度只能找出 0.2% 的像素，按可见性是 13%。
   - 被掩掉的动态像素和天空像素不算"看到过"：那些位置的 Gaussian 没有光度约束，会留下车辆鬼影（val039）和漂在天上的灰块（val056）。
   - 3D 记忆：之前各轮的补全视角，用当前模型渲染后也算作观测者，这样同一区域不会被重复生成（`--memory_dirs`）。
2. `fill_views.py`（wan venv）：Wan2.1-VACE-1.3B 做"视频 + 掩码"补全（白 = 生成）。
   - 816×544，每段 81 帧，段间重叠 16 帧；从第二段起，重叠帧用上一段的结果作条件（掩码全黑）。
   - 50 步 UniPC，flow_shift 3，CFG 5；生成的像素羽化 4 px 后贴回全分辨率渲染图。
   - 每段约 11 分钟，一条 200 帧的轨迹约 33 分钟，显存峰值约 37 GB。
3. `depth_views.py`（mapanything venv）：MapAnything 以已知内参和位姿估计补全帧的深度，再对齐到渲染深度。
   - 每个空洞连通块单独缩放：用块外 20 px 环带里至少 2 个训练视角看到过的像素，取渲染深度 / MapAnything 深度的中位数。
   - 只用一个整帧尺度的话，val039 的空洞边界处有 13–30% 的深度台阶；按环带对齐后，环带误差的中位数在 val039 上是 8.7%，val056 上是 5.1%。
4. `train_fill.py`（主 venv，E8）：
   - 在最新一轮每 3 帧的空洞像素上，每隔 3 个像素反投影出新的 Background Gaussian：颜色取补全图，各向同性尺度取像素足迹的 0.7 倍，不透明度 0.5，按半个足迹做体素去重。新 Gaussian 直接拼到初始 checkpoint 的张量后面。
   - 续训 3000 步。每步照常训练一个真实帧（初始 run 的全部 loss），外加一个补全视角：
     - L1 在空洞处权重 1、其他地方 0.1；LPIPS-VGG 的目标图在空洞外换成当前渲染（detach），权重 0.2；两项合起来乘 0.5；
     - 空洞处加逆深度 L1，权重 0.1。
   - 补全视角冻结 Sky 和 Affine，按该视角重算光线；关闭不透明度重置，学习率预热 300 步（checkpoint 里没有优化器状态）。已过 stop_split_at，不再增密。

**评测**（`eval_cross_camera.py --cams 1 2 3 4 --region_ref <E5c run>`）：
- 加入 SIDE_LEFT / SIDE_RIGHT（1920×886，按 FRONT 的渲染宽度、保持各自宽高比渲染；Affine 是逐像素的，索引图要跟着渲染尺寸改）。
- 新增"未观测区域"指标：用参考 run（E5c）的渲染深度和同样的空洞判定，在每个侧视角里找出从没被训练视角看到过的像素，去掉 GT 动态物体。所有 run 都在同一组像素上评，颜色对齐沿用重叠区的拟合。
- 与 D15 的旧表（只有 FRONT_LEFT / FRONT_RIGHT，写在 `cross_camera/`）分开存放，写在 `cross_camera_p9/`。

**其他**：`view_gs.py` 增加 right / up / yaw / pitch 滑块；`vis_freeview.py` 把多个 run 在同样的相机运动下并排渲染。

**第一轮结果（2026-09-25，右移 1.5 + 右转 15°）**
- val039（只用来打通流程）：生成 53 万个新 Gaussian；留出帧 20.08 / 0.784 / 0.285，E5c 是 20.30 / 0.787 / 0.286。
- val056（主场景）：

| val056 | E5c | E8 第一轮 |
|---|---|---|
| 留出帧 PSNR / SSIM / LPIPS | 31.39 / 0.910 / 0.101 | 31.46 / 0.916 / 0.098 |
| 4 个侧相机，未观测区域 PSNR / LPIPS（E5c 定义的同一组像素，约占 44%） | 16.61 / 0.353 | 16.86 / 0.352 |
| FRONT_RIGHT 未观测区域 | 16.86 / 0.295 | 17.26 / 0.288 |
| SIDE_RIGHT 未观测区域 | 17.44 / 0.389 | 17.86 / 0.389 |
| 4 个侧相机，重叠区 PSNR / LPIPS（各 run 自己的重叠区：E5c 覆盖 0.81，E8 0.86） | 17.58 / 0.616 | 17.36 / 0.651 |

- 已观测部分没被破坏；补过的方向（右侧相机）在未观测区域提高约 0.4 dB；没补的方向（左移、升高）与 E5c 相同，这是预期的，每轮只补一条轨迹。
- 重叠区指标变差，是因为 E8 的重叠区多出了生成内容（覆盖率 0.81 → 0.86），两者不是同一组像素；比较要看未观测区域那一行。
- 漏检：天空里的灰块和彩色碎块不在掩码里（训练视角的天空像素被算成"看到过"）。已修（`e4c6b31`），从下一轮起生效。
- 一轮只补一条轨迹，要覆盖自由视角需要多轮；每轮约 50 分钟（补全 33 分钟）。

**Wan 重绘（`scripts/repaint_views.py`，用户问到，2026-09-25）**
- 做法：Wan2.1-T2V-1.3B 视频到视频（SDEdit），816×544，81 帧。val056 第一轮的补全帧 40–120，strength 0.3 / 0.5 / 0.7，每次 2.5–5 分钟。
- 结论：**不能让画面变清晰**，strength 越高只会越改内容。
  - 0.3：结构保持，但牌子上的字变成乱码；
  - 0.5：树变成棕榈；
  - 0.7：房子变成仓库、路牌变成汽车。
- 用拉普拉斯方差量高频细节（960×640）：

| 画面 | 高频细节 |
|---|---|
| 真实 FRONT | 116–173 |
| 真实图缩到 816×544 再放大 | 57–81 |
| Wan VAE 在 816×544 往返一次 | 82 → 67（960×640 下 159 → 121） |
| E5c 在原轨迹上的渲染 | 43–60 |
| 第一轮补全帧（重绘的输入） | 34–53 |
| Wan 重绘输出 | 22–32 |

  - strength 0.3 在 flow_shift 3 下起始噪声已是 σ = 0.56，输出细节仍比输入少；
  - 1.3B / 480p 模型的"真实感"在运动、光照、布局上，逐像素细节比我们的渲染还低。
- **另一个发现：3DGS 在原轨迹上只有真实图像约 1/3 的高频细节**（留出帧 PSNR 却有 31 dB）。发虚首先是重建的问题，不只是外推的问题（OPEN_QUESTIONS 38）。

---

## O. E9：在重建网格上复现 LSD-3D 的外观生成（2026-09-25，用户决定）

**用户决定**：尝试生成，用保真换真实感，复现 LSD-3D 的后半部分，不做第一步的场景生成；扩散模型选 (a)：冻结的基础 SDXL + 深度 ControlNet，不微调，不加 IP-Adapter。

**LSD-3D 后半部分（arXiv 2508.19204 原文核对；代码未公开，项目页写 "Code (tba)"）**
- 在网格面上初始化 2DGS（180–220 万个，上限 400 万个）。
- GGDS 6000 步：每步 DDIM 反演到 t、去噪 5 步、L1 + LPIPS；t ~ U(t_min, t_max)，训练中逐步降低下界；加网格法向 / 视差正则、TV loss、SGLD、环境贴图。H100 上约 2 小时。
- 延迟渲染：输出视频时，每帧的高斯渲染编码成轻微加噪的潜变量，再由扩散模型去噪。原文说这一步让路面和树的高低频纹理都达到照片级。
- SDXL 先在 Waymo 上微调，再以网格视差为条件（原文 "we first finetune a LDM … on the desired image distribution"），数据量和步数没有给。
- 评测只看真实感（FID / FD-DINOv2 / FVD），分已见视角和新视角。

**E9 的做法（`scripts/train_lsd.py`）**：
- 从 E5c 出发（网格初始化的 3DGS，外观已拟合真实帧），在 E6 GGDS 基础上改为：
  - 自由视角池：每 4 个训练帧取 1 个，右移 U(−2.5, 2.5)、升高 U(0, 1.5)、偏航 U(−30°, 30°)、俯仰 U(−10°, 5°)，20% 的概率不移动；每 200 步整池换新的移动并重新生成目标（每张约 2.5 s）；
  - t ~ U(t_lo, 0.85)，t_lo 从 0.7 线性降到 0.3；权重 w(t) = sqrt(ᾱ)；
  - 默认**没有真实帧 loss**（`--real_w 0`）；
  - TV loss 权重 0.01（论文没给数值）；
  - 重新打开增密，持续到续训后第 3000 步。
- 其余同 E6：只在网格覆盖或不透明的像素上计算，天空和空白保持原渲染；冻结 Sky 和 Affine；网格深度 / 法向 loss；关闭不透明度重置；学习率预热。
- 延迟渲染：`vis_freeview.py --deferred_t`，逐帧用同一个 SDXL 重画。
- 没有复现的部分：SDXL 的 Waymo 微调（AGENTS §2）、2DGS、SGLD、生成的环境贴图。

---

## P. 用生成模型补齐左、右、后方的"虚拟相机"（2026-09-25，用户方向）

**用户判断**：目前最大的问题是只有前视相机。想用视频模型（或能控制角度的图像模型）从前视视频补出左侧、右侧甚至后方的画面，一点点往外补。

**为什么可行**：车在前进，侧面的东西在更早的帧里曾出现在前视相机的斜前方。所以转向后的视角里，远处和斜方向的内容很多是"看到过"的，3DGS 能渲染；真正没见过的主要是紧挨车身的近处和物体的背面。验证用 GT 的 SIDE_LEFT / SIDE_RIGHT（偏航约 90°）：E5c 在侧相机视角里有 44% 的像素从没被看到过（N 节）。

**三条路线：**
1. **沿用 Phase 9，只做纯转向**（主线）：3DGS 按目标角度渲染，由渲染决定视角；VACE 只补空洞；之前各轮作为 3D 记忆，逐步外扩：±30° → ±60° → ±90°。
   - 每轮约 35 分钟（隔帧渲染，补全约 22 分钟）；6 轮约 3.5 小时。
   - 这就是"能控制角度、一点点补齐"，而且不同角度来自同一个 3D 模型，内容一致。
   - 后方（±120°、180°）等侧面效果确认后再做。
2. **ReCamMaster**（对照，`scripts/recam_views.py`）：开源版基于 Wan2.1-T2V-1.3B，输入原视频加相机轨迹，整帧生成。
   - 先用预设轨迹（左 / 右摇约 20°、左 / 右绕行）在 val056 的真实 FRONT 帧上试。
   - 自定义轨迹要解决它的尺度与我们场景尺度的换算，暂不做。
3. **Cosmos-Transfer1-7B-Sample-AV-Single2MultiView**（NVIDIA，记录，暂不做）：前视视频 → 5 / 6 路环视视频（有 Waymo 后训练版本），57 帧、576×1024。
   - 但每路都要 HD map 或 LiDAR 控制视频，GT 不能用（AGENTS §10）；可以考虑把我们自己重建的点云投到虚拟相机里当"LiDAR"控制。
   - 7B 模型，单卡显存需求未知。

**E9 结果（2026-09-25，val056）**
- **第一次（`--real_w 0`，增密开）：发散。** 到第 15 轮（续训第 3000 步），渲染变成饱和的色条，路面变绿，天空布满品红和白色笔触；高斯数 68 万 → 280 万，参与 loss 的像素 68% → 97%。停掉，保留在 `results/_failed/E9_realw0/`。
  - 原因：没有真实帧锚定时，基础 SDXL（没在 Waymo 上微调）每一轮都把上一轮的伪影当作内容继续放大；新视角上的增密长出的漂浮物变得不透明后进入 loss 掩码。
- **第二次（`--real_w 1 --densify_steps 0 --t_max 0.8`）：不发散，但以保真换来的真实感有限。**

| val056 | E5c | E8 第一轮 | E9 |
|---|---|---|---|
| 留出帧 PSNR / SSIM / LPIPS | 31.39 / 0.910 / 0.101 | 31.46 / 0.916 / 0.098 | 30.29 / 0.904 / 0.156 |
| 4 个侧相机，未观测区域 PSNR / LPIPS | 16.61 / 0.353 | 16.86 / 0.352 | 16.53 / 0.366 |

  - 目测（`results/_vis_p9/e9/`，与 E5c、E8 并排）：车道线、灌木、护栏、路牌杆更清楚；但未观测区域原有的彩色碎块被 SDXL 画成了大片饱和的绿色，升高视角下路面也偏绿。
  - 这是第一次的问题减弱后的样子：垃圾区域在 loss 掩码内（不透明度 > 0.5），基础 SDXL 会把它们越画越"实"，颜色也会漂。
  - LSD-3D 没有这两个问题，推测是因为它的场景从干净的网格生成（没有残留的漂浮物），而且用的是 Waymo 微调过的 SDXL。
- **结论：** 在我们的约束下（基础 SDXL、不微调），GGDS 的真实感收益抵不过颜色漂移和垃圾放大。若要继续，至少要：
  - 生成 loss 只作用在网格覆盖的像素上（不含"不透明但未被观测"的区域）；
  - 加颜色统计约束（相对 E5c 渲染）；
  - 未观测区域先由 Phase 9 补全，再做 GGDS。
  
  延迟渲染尚未评估。

**P 节试验记录（2026-09-26 凌晨，val056）**
- **ReCamMaster 预设轨迹**（`results/_vis_p9/recam_val056/`，第 40–120 帧；每条约 17 分钟，17 GB）：画面很真实（832×480），但 4 条预设（左 / 右摇、左 / 右绕）都停在第一帧的位置，只转约 20°，车"不再往前开"，内容主要按第一帧编造。所以加了 `--custom_moves`：沿 Phase 3 估计的 FRONT 轨迹前进，同时逐渐转向（结果见下）。
- **GEN3C**（`scripts/gen3c_fill.py`，`results/E8/val056/views_gen3c/yaw45_ramp20/`）：
  - 用法：以真实 FRONT 帧起步，偏航 20 帧内过渡到 45°，把 3DGS 渲染（被看到过的像素）当作 3D 缓存。
  - 197 帧用 2 段、58.7 分钟，显存峰值 30 GB（全部卸载）。
  - 结果：不比 VACE 好。输出发灰、比输入的渲染还模糊；空洞处是糊状纹理；被判为"看到过"的漂浮物（竖直白条）被照搬。
  - 可能原因：GEN3C 训练时的缓存是由清晰的真实帧按深度 warp 出来的，模糊的 3DGS 渲染超出了它的输入分布；低显存模式下没有文本条件（prompt encoder 被关掉）；10 fps 与训练时的 24 fps 不同。
  - 更对路的用法：把最近几帧真实 FRONT 图像按深度 warp 到侧视角作为缓存（GEN3C 本来的用法，侧面的东西在更早的帧里被斜着看到过），3DGS 渲染只作补充。改动较大，列为后续。
- **ReCamMaster 沿行驶轨迹转向**（`--custom_moves yaw=30`，`results/_vis_p9/recam_val056_drive/drive_yaw30.mp4`）：第一帧正常，之后很快糊成一片没有结构的色块。81 帧里自车前进约 39 个场景单位，远超它训练时的相机运动（合成场景里几米的平移和绕行）。开源 1.3B 版无法重拍行车视频，这条路线放弃。第二条（yaw=-30）中途停掉。
- **今晚的主线**回到 Wan VACE：E8 第 1–6 轮，偏航 ±30° → ±60° → ±90°，隔帧，逐轮以 3D 记忆外扩；之后做 4 个侧相机评测和转向视频。

**E8 转向补全（2026-09-26 凌晨，val056，第 1–6 轮）**
- 做法：在第 0 轮（右移 1.5 + 右转 15°）之后，依次原地转向 +30°、−30°、+60°、−60°、+90°、−90°。
  - 隔帧渲染，99 视角 / 轮；Wan VACE 补全；每轮以之前各轮为 3D 记忆。
  - 深度：第 1–4 轮用 MapAnything；第 5–6 轮改用 MoGe-2（给定视场角），因为 MapAnything 在生成的侧视帧上预测的焦距偏离给定内参 27%（60°）和 76%（90°），深度不可信（`depth_views.py --backend moge`）。
  - 每轮约 37 分钟（补全 22、蒸馏 12）；高斯 68 万 → 140 万。
- 每轮空洞比例（均值 / 最大）：

| 轮 | r0 右 1.5 + 15° | r1 +30° | r2 −30° | r3 +60° | r4 −60° | r5 +90° | r6 −90° |
|---|---|---|---|---|---|---|---|
| 空洞 | 0.104 / 0.672 | 0.040 / 0.199 | 0.137 / 0.363 | 0.118 / 0.505 | 0.134 / 0.562 | 0.075 / 0.357 | 0.114 / 0.695 |

  右侧的空洞小，是因为第 0 轮（右侧）的补全作为记忆起了作用。
- **跨相机检查，未观测区域**（E5c 定义的同一组像素，PSNR / LPIPS）：

| | E5c | 对照：E5c 多训 6000 步 | E8 第 0 轮¹ | E8 第 6 轮 |
|---|---|---|---|---|
| FRONT_LEFT | 14.07 / 0.280 | 14.16 / 0.278 | 14.30 / 0.278 | 14.71 / 0.269 |
| FRONT_RIGHT | 16.77 / 0.302 | 16.75 / 0.309 | 17.26 / 0.288 | 17.24 / 0.285 |
| SIDE_LEFT | 18.06 / 0.460 | 17.92 / 0.458 | 18.10 / 0.458 | 18.64 / 0.451 |
| SIDE_RIGHT | 17.44 / 0.389 | 17.35 / 0.390 | 17.86 / 0.389 | 17.87 / 0.390 |
| 4 个相机 | 16.56 / 0.356 | 16.52 / 0.357 | 16.86 / 0.352 | **17.12 / 0.349** |

  ¹ 第 0 轮的数字用的是修正天空像素之前的空洞掩码（`e4c6b31` 之前），其余三列都用修正后的掩码。E5c 用两种掩码分别是 16.61 和 16.56：未观测比例到小数点后 3 位都相同，只有 FRONT_LEFT / FRONT_RIGHT 低约 0.1 dB。修正后的掩码存在 `results/E5c/val056/cross_camera_p9/unseen/`，U3 也用它。
  - 对照组（`scripts/train_refine.py --densify_steps 0`，只多训练、不生成）在未观测区域没有提升，所以 E8 的 +0.56 dB 来自生成补全。4 个相机都有提升，FRONT_LEFT 和 SIDE_LEFT 最多（+0.64、+0.58）。
- **留出 FRONT 帧**：E8 第 6 轮 32.35 / 0.930 / 0.092；E5c 31.39 / 0.910 / 0.101；对照 32.32 / 0.930 / 0.093。留出帧的提升几乎全部来自多训练的步数，不是生成内容。已观测部分没有被破坏。
- **目测**（`results/_vis_p9/e8_turns/`，E5c 与 E8 在 ±45°、±90° 并排）：
  - E8 清掉了 E5c 转向后满屏的彩色碎片，换成了看起来合理的墙、树篱、路缘；
  - 但 ±90° 的侧视仍是模糊、糊状的纹理，离照片级还远。
  - 另外，大角度时空洞掩码会漏掉一部分垃圾高斯（OPEN_QUESTIONS 39）。
- **容量诊断的对照结果**：
  - 增密：32.03 / 0.927 / 0.091；
  - 不增密、同样多训 6000 步：32.32 / 0.930 / 0.093。
  
  所以留出帧的提升主要来自多训练，增密只让 LPIPS 略好。原轨迹上"发虚"的主因仍是相机 / 位姿（OPEN_QUESTIONS 38）。

---

## Q. 方法上界 U3：drivestudio StreetGS，3 个前向相机 + LiDAR + 真值（2026-09-26，用户决定，oracle）

**用户决定**："我想测量方法的上界……先不做消融，直接测量上界。"动机：其他方法的效果看起来很好，而我们复现的结果一般，需要知道差距来自单目设定还是来自实现。

**做法（只在 val056 上）**：
- drivestudio 自带的 `configs/streetgs.yaml`，按上游发表时的用法运行：
  - 3 个前向相机（FRONT、FRONT_LEFT、FRONT_RIGHT），960×640；
  - Waymo 真值位姿、内外参和畸变（`undistort: True`）；
  - LiDAR 初始化（80 万点）和深度监督；
  - 真值框生成的 RigidNodes（`only_moving: true`：val056 没有运动车辆，停放车辆作为背景重建，不掩掉）；
  - 天空、Affine、CamPose 与上游默认一致；每隔 10 帧留出 1 帧，30000 步。
- **这是 oracle 实验**：违反 AGENTS §2 的"只用 FRONT、不读 GT、不用标定"，结果只作上界参考。SIDE_LEFT / SIDE_RIGHT 不参与训练，仍是真正的外推评测。
- 需要补的数据：
  - `scripts/extract_lidar.py`：本地 v1.4.3 validation 没有 scene flow，上游 `--process_keys lidar` 会读 flow 失败（C5）。脚本复用上游的转换函数，flow 写 0、flow 类别写 −1（数据集自己的"无 flow 标注"值）；drivestudio 只在评测里用 flow。val056：197 帧，每帧约 16 万点。
  - `scripts/sky_masks_processed.py`：用 Phase 4 的 SegFormer 生成 drivestudio 格式的天空掩码（相机 0 / 1 / 2 的天空比例 0.34 / 0.20 / 0.23）。
- 可比的评测：
  - `scripts/eval_front_heldout.py`：只评 FRONT 留出帧。drivestudio 的 metrics.json 对所有相机取平均，3 相机和单相机的结果不可比。
  - `eval_cross_camera.py --save_unseen / --unseen_dir`：U3 在真值世界系里，不能按 E5c 重新渲染来判定未观测区域，改为读取 E5c 存下的掩码，在同一张真实侧相机图像的同一组像素上评。
  - `views.front_image_index`：drivestudio 的多相机图像按帧优先排列，Phase 9 的脚本和查看器改为按 FRONT 取图。

**结果（2026-09-26，val056）**

| | E5c（单目，自标定） | E8（E5c + 转向补全 7 轮） | U3（30k 步） | U3-long（90k 步，调度 ×3） |
|---|---|---|---|---|
| FRONT 留出帧 PSNR / SSIM / LPIPS（`eval_front_heldout.py`） | 31.39 / 0.910 / 0.101 | 32.35 / 0.930 / 0.092 | 26.57 / 0.814 / 0.216 | 25.83 / 0.794 / 0.157 |
| FRONT 留出帧清晰度（渲染 / 真实的拉普拉斯方差） | 0.68 | 0.70 | 0.59 | 0.59 |
| 训练图像 PSNR（训练末期） | 33–34 | — | 29–30 | 29–30 |
| SIDE_LEFT 未观测区域 PSNR / LPIPS | 18.06 / 0.460 | 18.64 / 0.451 | 19.60 / 0.390 | 19.28 / 0.394 |
| SIDE_RIGHT 未观测区域 PSNR / LPIPS | 17.44 / 0.389 | 17.87 / 0.390 | 20.40 / 0.321 | 20.06 / 0.315 |

- 未观测区域是 E5c 定义的同一组像素（`--unseen_dir`）。FRONT_LEFT / FRONT_RIGHT 是 U3 的训练相机，不列。
- **自由视角：上界明显更好。** SIDE 相机上比 E8 高 1–2.5 dB。转向 45° / 90° 的渲染（`results/_vis_p9/u3_turns/`）里是完整的房子、停放车辆、护栏和灌木；E5c 和 E8 在这些角度是糊状纹理和碎片。差距来自传感器覆盖：左右前视相机直接看到两侧，LiDAR 提供几何。E8 的生成补全只追回了这段差距的一小部分（SIDE_RIGHT：差距约 3 dB，E8 补了 0.43 dB）。
- **原轨迹：上界反而更差，而且不是训练量的问题。** U3-long 把每个相机的训练步数提到与 E5c 相同，训练图像 PSNR 仍停在约 29.5；E5c 能拟合到 33–34。drivestudio 自己的 test 指标（26.3，3 个相机平均）与 `eval_front_heldout.py` 一致，排除了评测错误。
- 推测原因：drivestudio 对所有相机都用"每帧一个自车位姿 + 固定外参"。而 Waymo 的相机是卷帘快门，各相机的曝光时刻相对这个位姿也有偏移；val056 车速约 11 m/s，几十毫秒的偏差就是几十厘米。我们的 E5c 位姿是 SfM 直接对着 FRONT 图像拟合的，与图像更一致。还没有验证。
- 所以 U3 是"覆盖面"的上界，不是"原轨迹画质"的上界。

**U1：只用 FRONT + 真值位姿 + LiDAR（2026-09-26，用户选 A，用来检验真值位姿）**
- 设置与 U3 相同，但只用 `dataset=waymo/1cams`，960×640，30000 步。

| val056 | E5c（单目，自标定） | U1（FRONT + 真值 + LiDAR） | U3（3 相机 + 真值 + LiDAR） |
|---|---|---|---|
| 训练图像 PSNR（末期） | 33–34 | 35.0 | 29.5 |
| FRONT 留出帧 PSNR / SSIM / LPIPS | 31.39 / 0.910 / 0.101 | 30.46 / 0.891 / 0.099 | 26.57 / 0.814 / 0.216 |
| FRONT 留出帧清晰度 | 0.68 | 0.64 | 0.59 |
| 4 个侧相机未观测区域 PSNR / LPIPS | 16.56 / 0.356 | 16.48 / 0.344 | 23.17 / 0.237 |

- **真值 FRONT 位姿没有问题**：U1 对训练图像拟合得最好（35.0）。所以 U3 在原轨迹上变差，原因在于多个相机合在一起：
  - drivestudio 对所有相机用"每帧一个自车位姿 + 固定外参"，各相机的曝光时刻和卷帘快门都没有建模，三个相机对同一批高斯给出互相矛盾的约束；
  - CamPose 的学习率只有 1e-5，修不过来。
- **原轨迹上，我们的单目流程（E5c）与单相机 oracle（U1）相当**：PSNR 高 0.9 dB，LPIPS 基本相同，清晰度略高。所以原轨迹上的"发虚"不是实现问题，真值位姿和 LiDAR 也解决不了。
- **只加 LiDAR 不能改善侧视外观**：U1 在侧相机未观测区域和 E5c 一样（16.48 vs 16.56）。U3 的侧视优势来自左右前视相机的覆盖。
- **总结**：别人的效果"看起来好"，主要是因为多相机覆盖了两侧，而不是实现更好。要得到在原轨迹和侧视上都成立的上界，需要解决多相机之间的一致性（逐相机的曝光时刻位姿、更强的逐相机位姿优化，或者多相机联合自标定）。

---

## R. 单目补全向上界 U3 追赶（2026-09-27，用户选"第一个"）

用户决定：推进单目补全，以 U3（3 相机 + LiDAR）的侧视质量为追赶目标。按顺序做三件事：修空洞掩码；GEN3C 改用真实帧 warp 作缓存；重跑转向补全，用 SIDE 相机与 U3 对比。

**1. 空洞掩码（OPEN_QUESTIONS 39）**
- **光度一致性**：一个点除了深度一致，还要求新视角的渲染颜色与观测视角**真实图像**在投影点的颜色接近（RGB 平均绝对差 < `rgb_tol`），才算被该视角看到过。训练视角用训练图像，记忆视角用补全帧。
  - 取值 0.10：在 E5c 右转 45° 上，空洞比例从 0.307 增加到 0.330，新增的主要是垃圾区域的边缘；个别细小的真实物体（"BUSES ONLY"牌子，渲染位置略偏）也会被标为空洞。
- **记忆视角只为自己补过的像素作证**：这是大角度下垃圾被保留的主因。之前各轮的补全帧里，只有空洞部分是生成的，其余是当时的渲染（包括垃圾），却被整帧当成"看到过"，于是垃圾一轮轮被认证下来。现在记忆视角只在自己的空洞掩码内有效。
  - 在 7 轮之后的 E8 上右转 60°、以全部 7 轮为记忆：残留的天空拖影和彩色碎块都被标成了空洞。
- 评测侧的未观测区域不变：`eval_cross_camera.py` 关掉光度检查，仍用 E5c 存下的掩码，之前所有数字保持可比。

**2. GEN3C 以真实帧 warp 作缓存（`render_views.py --src_offsets`，`gen3c_fill.py --buffers`）**
- GEN3C-Cosmos-7B 最多接受 2 个缓存（`frame_buffer_max = 2`）。对第 t 帧的目标视角，把 t−6 和 t−15 帧的真实 FRONT 图像，用 GEN3C 自己的 splatting 前向 warp 过去。
  - warp 用的深度是 run 在源相机处渲染的深度，与 3D 模型一致。
  - 车在前进，侧面的东西在更早的帧里曾在前方被斜着看到过；11 m/s 时，侧向 8 m、90° 方向的点大约在 15 帧前进入 FRONT 视场。
  - 两个缓存在 E5c 右转 45° 轨迹上各覆盖约 34% / 33% 的像素。
- 结果（`results/_diag/gen3c_v2/val056_yaw45b/`，197 帧 64 分钟，与 v1 并排）：
  - 明显好于 v1（3DGS 渲染作缓存）。v1 发灰、偏糊，还照搬竖直白条漂浮物；v2 颜色和结构都像真实街景（树、树篱、护栏、远处的房子），白条也没了。
  - v2 仍会编造内容：阴沉的天空、远山、一辆红车；细小结构（路牌、廊架）会丢失。与 U3 相比仍有明显差距。
- 另外，近处树篱这类植被在各视角下渲染深度不一致，深度检查会把它们标为空洞（右转 45° 的第 145–155 帧 90% 以上都是空洞），与颜色检查无关。这些区域会被重新生成。
- 下一步（第 3 项）：E10，从 E5c 出发，GEN3C v2 转向补全 ±30° → ±60° → ±90°，隔帧，MoGe-2 深度，修正后的空洞掩码；用 SIDE 相机与 E8、U3 对比。

**3. E10：GEN3C v2 转向补全，与 E8、U3 对比（2026-09-27，val056）**
- 做法：从 E5c 出发，6 轮：+30°、−30°、+60°、−60°、+90°、−90°。
  - 隔帧渲染（99 视角 / 轮），704×1280，20 个视角内从 FRONT 位姿过渡到目标角度；
  - GEN3C 的两个缓存是 t−6 和 t−15 帧真实图像的 warp；MoGe-2 深度；修正后的空洞掩码。
  - 每轮约 52 分钟（GEN3C 约 29 分钟）；高斯 68 万 → 约 430 万。
  - 偏航 90° 那一轮有 4 / 99 帧完全没有被看到过的像素，这些帧不写深度（`depth_views.py --min_ref_px`），只提供颜色监督。

| 未观测区域 PSNR / LPIPS | E5c | E8（Wan VACE，7 轮） | E10（GEN3C v2，6 轮） | U3（上界） |
|---|---|---|---|---|
| FRONT_LEFT | 14.07 / 0.280 | 14.71 / 0.269 | 13.58 / 0.274 | —（训练相机） |
| FRONT_RIGHT | 16.77 / 0.302 | 17.24 / 0.285 | 17.75 / 0.276 | —（训练相机） |
| SIDE_LEFT | 18.06 / 0.460 | 18.64 / 0.451 | 18.30 / 0.439 | 19.60 / 0.390 |
| SIDE_RIGHT | 17.44 / 0.389 | 17.87 / 0.390 | 17.72 / 0.376 | 20.40 / 0.321 |
| 4 个相机 | 16.56 / 0.356 | 17.12 / 0.349 | 16.81 / **0.340** | 23.17 / 0.237 |
| FRONT 留出帧 PSNR / LPIPS / 清晰度 | 31.39 / 0.101 / 0.68 | 32.35 / 0.092 / 0.70 | 31.70 / 0.103 / 0.71 | 26.57 / 0.216 / 0.59 |

- **E10 与 E8 各有长短。**
  - LPIPS：3 / 4 个相机都是 E10 最好，平均 0.340 对 0.349；
  - PSNR：E10 平均低 0.3 dB，FRONT_LEFT 甚至低于 E5c。
  - 可能原因：GEN3C 编造的内容（阴沉的天空、树篱、车辆）在感知上更像真实街景，但逐像素与真实图像对不上。
- **目测**（`results/_vis_p9/e10_turns/`，与 E5c、E8、U3 并排）：
  - E10 清掉的漂浮物比 E8 多，两侧是颜色自然的树篱、树、土地，而不是 E8 的灰色糊状纹理；
  - 但蒸馏回 3D 后仍然糊。各轮生成的内容互相不完全一致，蒸馏时被平均掉，与 E6、E7 的规律相同。
- **单目方法的根本限制：** 左转 90° 时 U3 看到的是一栋房子和一辆停着的白色面包车，E10 补出的是树篱。紧挨车身的那片区域 FRONT 从来没看到过，任何生成模型都只能编。
- **结论：** 修正掩码加真实帧 warp 后，感知指标改善，但与上界的差距（约 6 dB / 0.1 LPIPS）主要来自"根本没被观测到"，生成补全只能让它看起来合理，不能让它正确。要得到真实的侧面内容，只能靠更多观测，例如 Mapillary 的多次通行（§14）。


## S. 现成模型优先的一致性修正路线（2026-09-27，用户授权）

**用户决定**：取消微调和训练的禁令，但应优先使用已有模型跑通；只有现有方法达不到成果时，训练和微调才可作为补充。按本次分析提出的路线开始，并同步维护 AGENTS.md。

**执行顺序**：跨轮生成记忆 → 几何验收 → 生成/蒸馏分离诊断 → 完整现成模型基线 → 固定协议的评测扩展。先在 val056 短轨迹上用已安装的 GEN3C-Cosmos-7B；新实验 E11 单独保存，保留旧结果。不启动网络训练或微调。

**实现诊断（尚待消融验证）**：`scripts/run_phase9.sh` 传给 GEN3C 的两个缓存是 `real0 real1`，历史生成只参与空洞判断和最终拟合，没有进入跨轮生成缓存。MoGe-2 是逐帧深度，边界尺度对齐不等于跨视角几何一致。`train_fill.py` 将各轮目标同时拟合；目标矛盾是模糊的候选原因，不能仅凭现象断言已经定位。

**评测解释修正**：R 节 U3 的四相机平均包含训练相机 FRONT_LEFT/RIGHT，因此“约 6 dB / 0.1 LPIPS”不代表公平的外推差距。按现有相同未观测区域协议，E10 对 U3 的 SIDE_LEFT/RIGHT 差距约为 1.3/2.7 dB。U1 与 E5c 接近也未排除共同实现、相机模型和分辨率限制；多相机不同步/卷帘快门仍是待验证假设。

**训练/微调准入**：先记录现成方法和修复后基线、明确未达目标、排除集成错误，再写明适配数据及评测隔离、验收指标与算力预算。优先最小适配；不以微调替代相机/深度/缓存一致性修复。现有权重来自其他数据集的预训练不属于本项目自行训练；仍需记录模型来源和潜在数据重叠。

**验收原则**：短轨迹既保留真实 FRONT，又比较生成重投影一致性和蒸馏前后细节；无支持像素标未知，不算验证成功。小样本只能定位问题，不能声称全局质量或闭环能力已提升。静态/单目/GT 隔离维持，未新增云 GPU、侧相机输入或动态重建授权。


### S1. E11 跨轮缓存实现与诊断协议

- 新增 `dashrecon/gen/consistency.py` 和 `scripts/validate_generated_views.py`：以相邻不同相机的生成 RGB-D 做重投影检查；被遮挡/出界是未知，前方自由空间矛盾和颜色不符是冲突。默认至少两票支持、冲突比例不超过 25%；只把原始空洞内通过检查的像素作为生成记忆。这只是自洽性，不能证明隐藏内容真实。
- `gen3c_fill.py --cache_policy consistent`：使用已安装 GEN3C 官方 `forward_warp`，第一个缓存按覆盖率选择真实 FRONT，第二个缓存以互补真实观测为先，剩余空缺才能写入经过验证的历史生成。按时间邻近候选的实际投影覆盖挑选记忆；真实观测重叠处严重矛盾的候选拒收。真实缓存排除预测动态/天空掩码与留出帧。
- `--out_dir` 将输出写到新目录并链接只读输入，禁止覆盖已有生成帧；`cache.json` 记录逐帧来源、覆盖、冲突、留出源帧排除和 commit。`--require_memory` 防止无记忆情况下静默宣称完成了记忆实验。
- 发现旧 `real0 real1` 路线的历史参考帧可能落在 FRONT 留出时刻。新路线排除它们；已有 E10 因而只能作为历史参考，不能把新旧差值纯归因于缓存改进。公平消融使用相同 `consistent` 模式，分别不给/给记忆；固定输入相机、种子及 121 帧模型窗口；先做两组相同 8 步的接线消融（不能代表论文质量），通过后才扩为标准 35 步。
- 官方 `gen3c_dynamic.py` 明确要求 N*120+1 帧；本次保留模型原生 121 帧窗口，不私改结构为短窗。先做一段/一个方向的诊断，避免未验证即展开多轮大角度训练。
- 几何合成测试覆盖相机平移、遮挡、深度冲突、颜色冲突、无证据拒收；与已有自由视角测试合计 10 项通过。真实数据验收与生成实验待运行，暂不宣称画质改善。


### S2. 几何验收与蒸馏诊断接入

- val056/E10/+30° 的 99 帧 RGB-D 初检：原始空洞中 12.67% 像素通过至少两帧支持及冲突阈值，49.47% 没有可检验证据；剩余为支持不足或冲突。阈值见 `results/E11/val056/validation/r0_yaw30/validation.json`。此比例受估计深度、遮挡和纹理影响，不是生成正确率。
- `train_fill.py --validation_dirs ... --seen_w 0`：只在通过检查且深度有效的位置增补高斯；生成 RGB/深度 loss 使用置信度；原始 FRONT loss 继续锚定观测部分。无可靠生成点则显式停止，不把未知像素默认视为可靠。旧模式仍可复现。
- `diagnose_distillation.py` 在完全相同相机和分辨率比较直接生成与 3D 拟合，另渲染偏航 +5° 的未监督视角，保存图像及空洞区清晰度/拟合误差。清晰度只是辅助，不把噪声锐化当作成功。
- E11 首次消融复用 E10 的相机/渲染/生成记忆用于定位缓存问题。这些历史产物可能间接包含旧留出帧信息，因此只作为实现诊断，不宣称是完全无泄漏的质量基准；最终基准需从 E5c 按新协议重新生成全部记忆。
- 现成新基线核查：UniWorld-View 官方建议 >=60GB GPU；CogNVS 有可直接使用的 inpainting 权重，但官方说明未经测试时微调质量较低，完整微调配置需要至少 5 张 48GB A6000。当前先完成已安装 GEN3C 的对照，不自行训练，也不把受限配置当作论文复现。来源：https://github.com/PKU-YuanGroup/UniWorld-View 、https://github.com/Kaihua-Chen/cog-nvs （2026-09-27 查阅）。

## T. 单目场景补全分析与局部蒸馏验证（2026-09-27，用户授权）

用户要求“记录你的分析。按照你的规划进行验证”。完整分析、官方来源、模型适用性和固定验证协议见 [SCENE_COMPLETION_ANALYSIS.md](SCENE_COMPLETION_ANALYSIS.md)。目标允许隐藏内容合理猜测，但要求在不同视角保持同一静态场景。

此次先执行 Phase 9 修正任务 3：利用已有缓存/几何检查做直接生成、同相机 3D 拟合、未监督视角的分离诊断；不把任务 1/2 的未验收部分标为完成。先用 E10 标准步数结果定位，再对 E11 的两臂做等预算局部拟合，生成器不微调。

状态修正：E11 磁盘上的 8 步生成/深度/几何检查已完成（commit `7c4dc54`，run_status.json）。real-only/memory 自洽验收比例 12.6768%/22.0166%，未知比例 27.2236%/13.7532%，历史生成记忆平均覆盖仅 0.3599%。这些是有旧产物污染可能的实现诊断，不是正确率或最终画质改善证据。局部 3D 拟合及分离诊断于 2026-09-28 完成，见 T2；完整质量验收仍未通过。

复现：工作区存在用户未跟踪的 `tmpidea.md`，不修改、不提交。长实验使用提交后的独立干净 worktree，通过链接读取数据和环境、写入结果，保留用户工作区原状。

诊断实现：`run_e11_probe.py` 固定生成视角 40–64（25 个视角）作局部监督，两臂各 3000 步、seed=0，从 E5c 出发。`diagnose_distillation.py` 保存原始空洞/通过区/共同通过区的 RGB 误差、拉普拉斯方差与相关性，保存未监督 +5° 与横移 +0.5 场景单位的图像及往返视频。共同掩码必须对应完全相同相机与空洞；无支持区域返回未知。往返只作固定场景重现检查，不当成独立几何证据。E10 的三个角度先用标准 35 步生成结果检查，E11 的 8 步结果仅用于缓存接线/拟合诊断。

核查发现：`dashrecon/gen/views.py::training_observers` 当前颜色支持使用训练相机的**渲染 RGB**，不是 R 节所写的真实 RGB（其模块注释已说明原因）。本次保留输入/掩码避免混入另一变量，记录该差异；它仍可能让几何/渲染共同错误自我支持，应另做消融。

2026-09-28 验证工具准备完成：主环境运行 `python -m pytest -q tests/test_distillation_metrics.py tests/test_generation_consistency.py tests/test_views.py`，13 项通过（日志 `results/_logs/e11-probe-tests.log`）。相关 Python 编译与 `git diff --check` 通过。诊断视频仅为展示缩放；全部指标在原分辨率 PNG 上计算。

运行中修正：上游 `ScenePixelSource.propose_training_image` 用 Python `random.random/choice` 采样真实 FRONT，而旧 `train_fill.py` 仅固定 Torch 与局部 NumPy RNG。第一轮在拟合完成前主动停止，保存 `results/E11/val056/probe_20260928/ABORTED.json`；其拟合不能用于固定种子的两臂对照，E10 固定相机的渲染诊断仍可参考。新增统一种子入口与真实帧采样回归测试，修正后从 E5c 重跑两臂到新的 `_seeded` 目录，不续接未完成 checkpoint。固定随机流不保证 CUDA 浮点归约逐位确定。

种子修正后的 14 项测试通过（`results/_logs/e11-probe-seeded-tests.log`），包含直接调用上游真实帧采样器的重复性检查。训练保存每步真实/生成视角采样序列，runner 必须确认两臂的序列完全相同后才标记完成。

### T2. 局部验证完成：颜色可拟合，细节与自由视角质量仍不足

报告：[E11_PROBE_RESULTS.md](E11_PROBE_RESULTS.md)。正式输出 `results/E11/val056/probe_20260928_seeded/`，代码 `e120bbc`，阶段全部成功、两臂采样完全相同，累计 30.03 分钟。E10 的 +60°/+90° 空洞区细节方差比 0.414/0.181，相关性 0.049/0.028。E11 共同通过区拟合后 PSNR real-only/memory 32.488/32.265，方差比 0.268/0.334，相关性 0.263/0.220；memory 方差更高但匹配细节较低，未显示明确质量优势。PSNR 目标是合成图，不是真值。

局部生成记忆覆盖仅 0.461% 全图、13/25 帧有记忆；原始空洞约 24.7%，共同通过仅 6.84% 空洞。掩码来自 E10/+60° 历史轮次，而本轮以 E5c 初始化，不能代表首轮完整外扩。实际深度为 MoGe-2（旧 train_fill 模块注释称 MapAnything，与已保存 depth.json 不一致，本次将注释改为“估计深度”）。这些限制均保留；不把 8 步接线结果当作模型上限或无泄漏基准。

FRONT 历史留出 PSNR 31.391 → 32.057/32.153，未明显退化；无纯真实帧继续训练对照，不能归因于记忆。两臂 PyTorch 显存峰值 10.31/10.34 GiB。固定场景往返首尾差为 0，仅为可重复渲染检查；未监督视角仍明显碎片化。Phase 9 质量未通过，仅任务 3 的此次局部分离诊断完成。

下一项优先验证单/少量可靠视角的表示密度与拟合上限，再分离联合深度与跨视角冲突；随后从 E5c 重生成小步长、高记忆覆盖的正式 35 步基准。暂不因本结果启动生成网络微调，也不扩大到新场景。此路线后续仍需独立道路/碰撞几何与尺度验收，不能把视频观感当作可测试资产。

2026-09-28 用户请求查看可视化：为 E5c 与两臂局部模型提供独立 8081 交互入口。自动审批拒绝了默认监听 0.0.0.0 的启动请求，理由为可能暴露私有场景；改用已核实的 `ViserServer(host=...)` 参数，`view_gs.py` 新增 `--host` 且默认 127.0.0.1，使用本机或 SSH 转发访问。保留原 8080 服务，精确对照图/视频继续由报告链接提供。

## U. E12：行驶轨迹与关键位置扫视的等预算诊断（2026-09-28）

用户询问为何随车转向，而非每个位置都扫 0–60°，随后授权“开始”。GEN3C 不要求相机随车运动；开头缓转用于真实 FRONT 第一帧与连续视角，原地扫视同样满足。原地旋转可增加角度覆盖，但不能单独提供深度视差；多个位置之间需要共用场景假设并进行平移验证。当前行驶方案尚无更优证据。

本次只推进 Phase 9 修正任务 3 的采样诊断。配置 `configs/dashrecon/E12_sweep_probe.yaml`；先在 val056、frame 96–112，保留旧模型。两臂都从 E5c 重渲染输入，原生 121 帧、GEN3C 35 步/seed=1、25 个监督视角、3D 优化 3000 步/seed=0。

- drive：估计 FRONT 位姿沿 frame 96–112 连续插值，前 20 个间隔从 0° 转到 60°，随后保持相对 60°。
- sweep：frame 96 固定中心 36 帧 0→60°；6 帧平移到 frame 104；固定中心 36 帧 60→0°；6 帧平移到 frame 112；固定中心 37 帧 0→60°。这是三个扫视联合的一次视频推理，共享真实 RGB-D 缓存和序列条件，避免逐段独立猜测。位置坐标不表示真实时间；生成动态仍可能污染静态假设。
- 两臂均使用真实 FRONT 候选 t、t−6、t−15，按覆盖选择两个缓存，排除真实留出源帧及动态/天空；不读取旧 E10/E11 生成记忆。第一帧为训练源 frame 96 的真实 FRONT。检查完成后的 RGB-D 留存为下一批可用记忆，本次先比较一次局部回写，不声称多轮外扩已完成。
- 新采样 CLI 使用磁盘 JSON，FRONT 位姿与 K 来自 E5c/CamPose，无 GT。连续位置的平移线性插值、旋转 SLERP；SciPy `Rotation.from_matrix` / `Slerp` 用法已对照上游 `utils/camera.py`，GEN3C 参数沿现有已核实调用。
- 两臂统一新增 `translated` 几何诊断：参考相机与查询相隔 >=1 场景单位，参考中心彼此也相隔 >=1 单位，最多 4 个方向/距离匹配参考；逐像素视差 >=0.5°，仍需至少两票支持及冲突 <=25%。纯旋转不晋升为几何证据。此诊断独立保存，不改变已有评测协议；通过率不得与旧 offset 协议直接比较。
- 同相机生成/拟合及未监督 +7°、横移 +0.5 场景单位分开保存；另外用相同 E5c 相机在 3 个位置、5 个角度及横移对照两个 3D 模型。报告真实 FRONT 保真、生成自洽及拟合细节，不把不同合成目标的绝对 PSNR 当作隐藏真值质量。
- 初始几何/表示/学习率保持相同；本次只变采样方式，未同时加入高斯稠密化、网络微调或联合深度优化。物理尺度仍未验收，所有平移阈值明确使用场景单位。

运行前提交代码并在新干净 worktree 启动。为释放 48GB 单卡显存可临时停止本项目两个查看器，保存启动参数，实验后恢复 loopback 服务；不触碰其他项目 GPU 进程。CPU 几何测试和运行日志保存在 results，结果总结完成后独立 commit。


### U1. E12 对照完成：三个锚点扫视没有显示明确优势

报告：[E12_SWEEP_RESULTS.md](E12_SWEEP_RESULTS.md)，输出 `results/E12/val056/probe_20260928/`。独立干净代码 `81882ed`，两臂全部成功；121 帧、标准 35 步生成，25 个监督视角、各 3000 步优化，真实 FRONT 与监督池索引的随机流完全相同，累计 105.26 分钟。19 项相关测试通过，242 张生成 PNG、两组诊断视频、共同 30 相机的 RGB/深度和输入哈希已核对。此处只完成任务 3 的采样诊断，完整自由视角场景仍未验收。

- drive / sweep：原始空洞占全图 43.22% / 23.97%，空洞中通过新增平移/视差检查 8.73% / 3.11%，未知 46.18% / 57.66%。角度分布不同，较少输入空洞不能代表 3D 回写减少了更多空洞；45–60° 分组通过比例也为 9.12% / 3.69%。全部比例都是估计 RGB-D 自洽，不是隐藏内容正确率，不与旧 E11 邻帧协议直接排名。
- sweep 真实缓存覆盖较高（34.89% / 27.57%，drive 为 15.89% / 8.69%），但通过检查的生成点更少：spawn 5,469 / 31,895 个。固定位置旋转没有被重复计作独立深度支持，三段之间的估计深度、遮挡与生成假设仍可能冲突；本次未定位具体责任。
- 各自通过区拟合 PSNR 达 33.878 / 36.634，细节方差比 0.353 / 0.450，细节相关性 0.247 / 0.121。目标和掩码不同，不能用绝对 PSNR 排名；完整空洞的改进仅为 14.490→15.399 / 16.747→17.004。共同相机的 60° 与横移仍明显模糊、碎片化，扫视没有显示明确质量优势。不能由此否定更密位置或更强几何约束下的扫视。
- FRONT 历史整图留出指标为 32.009 / 32.065 dB，LPIPS 0.0990 / 0.0981，相对 E5c 无明显退化。无“仅继续真实帧训练”对照，不能归因于生成；仍不是延后的完整 Phase 1 协议。
- 本次模型为 GEN3C-Cosmos-7B（Cosmos-Predict1），未换成 Cosmos Dream、未微调。生成 Torch allocated 峰值均 32.78 GiB，3D 优化 10.33 / 10.28 GiB；生成环境 Torch 2.6+cu124，主环境 2.4+cu121/gsplat 1.3。所有源图实际为 FRONT，排除 cache 的留出源帧 90/100/110；未读取旧 E10/E11 生成记忆，来源与哈希保留。

代码/文档解释：当前 `hole_mask` 按真实 FRONT 观测支持而非仅低 opacity 判未知区域（R 节已更改），因此“空洞”也包含不透明的无约束高斯。未通过区域不接受生成监督，旧失真几何不会自动消失。`fill.json` 的 `buffers=[render]` 是 consistent 分支未使用的旧 CLI 默认项；实际两个真实 warp 以 `cache.json` 和 coverage 为准。诊断 CLI 的历史输入提示沿用旧文本，不代表 E12 使用过 E10/E11 图像，实际 provenance 见 `input_audit.json`。

下一项保留任务 3 的拟合上限诊断：固定支持充分的单/少量视角，比较表示密度/有限分裂，再分离多视角深度及目标冲突；采样可组合关键位置扫视与更密的同方向平移，并按真实/验证记忆投影覆盖安排。暂不扩大到每个 frame×多角度、侧后视、多场景或生成网络微调。完整生成记忆循环、几何质量和可测试资产验收仍未完成。

E12 完成后恢复本项目查看器，均仅监听 127.0.0.1，HTTP 200；8081 提供 E5c、E11 real-only/memory、E12 drive/sweep 五个模型，生成模型显式标 gen。原 8080（E5c/E8/E10/U3）恢复，其他项目 GPU 进程未触碰。查看器标签功能独立提交 `9b7adf7`，未改变在 `81882ed` 上运行的实验代码。历史结果未覆盖；只 commit、不 push。

### U2. 用户左右 90° 检查：E11 / E12 自由视角质量失败

2026-09-28 用户明确反馈 E11、E12 drive / sweep 在左右 90° 均很差。此前局部画面的主观认可不能代表完整场景通过；任务执行完成不代表场景质量通过。E11 两臂从 E5c 只拟合历史 +60° 的局部 25 视角，E12 只更新 frame 96–112、0→+60°；均没有左右 90° 补全，也不是继承 E10 全部轮次的场景。已有 +60° 画面也失败，因此不以未覆盖为全部解释，不将增加角度数量视为已验证修复。

本次仍只推进任务 3 的失败诊断：固定 E5c 相机，三个位置、五个偏航（−90/−60/0/+60/+90°），串行渲染五模型，不重新生成或训练。扩展现有比较 CLI 的模型数、角度与标签，旧 E12 默认网格保持不变；独立保存，不改变历史评测协议。范围、复现命令和待验证变量见 [SIDEVIEW_FAILURE_ANALYSIS.md](SIDEVIEW_FAILURE_ANALYSIS.md)。完整 Phase 9 质量继续未通过。

固定相机复查已完成：代码 `39796c2`、干净独立 worktree，65.99 秒、Torch allocated 峰值 3.88 GiB。75 张 RGB、75 个有限值深度数组、15 张五列图和五个 checkpoint 哈希已核对。目测三个位置的左右 90° 均失败，负偏航大体保留 E5c 的失真片状结构；正偏航只有局部新增内容，+60° 也未达到可用质量。没有 GT 分数、没有新生成或训练，既不接受场景质量，也不以此断言某个生成模型达到能力上限。保留原模型及查看器，报告内提供左右 90° 对照图。首次因 PATH 缺少 venv bin 而无法找到 Ninja 的输出/日志独立保留，显式补齐环境后同代码重跑，无静默降级。

可视化身份澄清：用户提到 8090 并追问先前认可版本。当前 8090 实际是 HUGSIM 发布场景 `100613054308_0_200`，与 val056/E11 不同；E11 对照位于 8081。原/现 8081 的 E11 checkpoint 路径一致，修改时间在原启动之前，当前哈希与侧视复查一致；查看器仅增加标签功能，渲染和相机算法未变。8081 默认 E5c、frame=0、yaw=0，重启恢复默认；未记录历史 GUI 选择，不能直接认定用户当时看的分支/视角。E8/E10 轨迹最大 ±90°，本项目尚无这些模型的专门 180° 后视补全。详情并入失败诊断报告，只读核对未改服务或训练输入。

## V. E13：surrounding camera 补全试验（2026-09-28，用户明确要求）

用户要求试一次 surrounding camera 补全，授权此次扩大到两侧和后方；U2 的质量失败仍保留，不把此试验当作局部质量已经通过。此次只推进 Phase 9 的一个环视补全实验，先用已安装的冻结 GEN3C-Cosmos-7B，不微调网络、不新增真实相机、不使用 HUGSIM、GT 或 Waymo 标定，不覆盖旧结果。

计划和复现见 [E13_SURROUND.md](E13_SURROUND.md)，配置 `configs/dashrecon/E13_surround.yaml`，结果 `results/E13/val056/surround_20260928/`。三个位置 frame 96/104/112，共享 FRONT 自标定相机中心和内参，以每 45° 一台虚拟相机覆盖八个方向；各相机共中心，不冒充真实车辆环视标定。空间量使用估计场景单位，不宣称真实米。

两轮分别生成 0→+180° 与 0→−180°。每轮三个位置各 37 帧扫视（角度步长 5°），中间两段各 5 帧平移，合为一个 121 帧视频、GEN3C 标准 35 步/seed=1。每轮固定 41 个 RGB-D 监督视角、4000 步高斯优化/seed=0；保留真实 FRONT loss 和此前通过验证的监督，第一轮更新后再渲染第二轮。预计约两小时，实际时长/显存逐阶段记录。为了给生成器留显存，暂时暂停本项目 8080/8081；完成后恢复，8090 的 HUGSIM 继续运行。

与旧代码差异及必要修正（不是只改变采样的因果消融）：

- `depth_views.py` 原边界对齐在参考像素不足时整帧深度置零，后方难以提升到 3D。新增显式 `scene_global` 选项：用 count≥2 的已支持估计几何像素和 MoGe-2 度量深度先求统一尺度，第二轮固定复用；无任何尺度锚点报错。所有未观测深度仍是候选先验，必须过平移重投影检查。默认 `boundary` 和历史产物不变，不把有深度等同于验收通过。
- 当前记忆缓存按 frame 最接近的三个索引选候选。同一 frame 有许多朝向时会选到前方、错过后方。新增显式 `pose` 策略，按朝向/中心距离排序，再实际 warp 六个候选并选覆盖；只使用历史原始空洞中 confidence>0 的 RGB-D。真实 warp 优先，第二轮 `--require_memory` 显式拒绝完全没有有效记忆的接线。
- 几何检查沿用 E12 的中心分离≥1 场景单位、像素视差≥0.5°、至少两票支持、depth 10%、RGB 0.12、冲突≤25%。额外拒收与已验证历史记忆矛盾的新像素；记忆不计作新的独立视差支持，未知不冒充通过。检查属于生成自洽，不是真实隐藏几何的正确率。
- 密集初始化：所有选定目标都可 spawn、像素 stride=2；高斯分裂和 opacity reset 仍关闭，真实区域并非冻结。此前模糊可能还受旧不透明失真高斯遮挡影响，本轮不凭外观任意删除真实几何；不能将结果归因于单一变更。
- 查看器增加可选初始 frame/yaw，默认值仍为旧的 0/0。E13 新入口默认选最终模型、frame=104，避免把未补的位置或默认 E5c 误当作环视结果。

验收记录分开：直接生成、同相机 3D、未监督 yaw +7°/横移 +0.5、第一轮内容在第二轮后的保留情况；最终固定三位置 × 八方向（±180 重复作回看检查）× 两横移共同相机网格，与 E5c 和第一轮模型比较。完整 Phase 1/GT 几何及仿真资产验收仍延后，不用合成目标 PSNR 代替环视质量。

运行中输入修正：首轮 `37a774f`、源 offsets=[0,6,15] 的 cache 预检显示 frame104/yaw180 静态真实覆盖=0，yaw135 仅 0.083%。源 frame104/98 的点完全不在后视 frustum；frame89 有 22,073 个投影点，但动态/天空掩码后没有可用静态 warp。最近 15 帧不是完整的已观测历史。主动停止未完成的第一轮生成，保留原目录和 ABORTED.json，不把它计为质量对照。正式配置扩大为 [0,6,15,30,60,90] 的真实 FRONT 候选；按投影覆盖选缓存，仍剔除留出帧/动态/天空，每轮先只构建缓存预检再启动生成网络。新输出 `results/E13/val056/surround_20260928_history/`，原定两轮、35 步生成和 4000 步优化不变。


### V1. 首轮结果与缓存有效帧修正

首轮原生 121 帧、35 步生成完成，推理主体 28.79 分钟。直接侧/后视已有明显重影和糊纹理；单一场景尺度 1.7744 的 MoGe 深度在不少视角近似平面，尺度锚点比值 p5–p95=0.618–10.959。初检空洞通过率仅 0.11284%，未知 64.44859%；三个位置的 45/90/135/180° 精确目标通过像素均为零。只有 7 帧有通过像素，其中大部分在 155°，没有被每三帧的训练网格选中；首轮只新增 838 个高斯。不能用后续真实 FRONT loss 改善代表环视补全成功。

据此发现记忆候选接线缺陷：frame96/104/112 的后视最近六个候选全部置信度为空，挤掉了已经通过检查的 155° 帧。`pose` 候选池改为先依据 validation 的 accepted_pixels 排除空帧，再按方向/中心排序，最后仍按实际投影覆盖选来源；跨轮冲突检查使用同一有效帧筛选。历史默认 `frame` 排序保持原行为。没有降低几何门槛、把无支持像素设为通过或修改冻结生成权重。新增“六个空的后视帧不能排挤有效斜视帧”的回归测试。

为避免新子任务遇到未提交代码，暂缓此次调度器的下一阶段启动，正在运行的首轮训练继续；修正先在 /tmp 开发和测试，再复制、commit、同步到工作树、恢复队列。首轮重建产物来自 ec7b4b7，后续阶段使用修正提交，各阶段自己的 meta 保留实际 commit；最终根 meta 增补 round_code_commits 和 CODE_UPDATE.json，不能再将整次运行描述为单一冻结提交。修正不改变首轮无记忆的生成/深度/自检查结果；正式复现从修正提交干净启动可统一使用该提交。

### V2. 文件审核格式修正

完整环视生成、两轮高斯优化及共同相机渲染已完成，审核首次在 opacity/000.npy 报错。核对 `render_views.py:138`：它直接保存渲染器的 `(H,W,1)` opacity；深度和 count 保存 `(H,W)`。两轮 opacity 数值有限、非负且不超过 1，产物没有损坏。修正审核的逐类精确形状，并增加 opacity≤1 检查；不挤压/改写产物、不放宽深度或置信度检查。重新从已提交的干净分析工作树执行完整审核；第一次失败日志单独保留。此修正仅影响审核，不影响实验模型或生成结果。

## W. 两个诊断：侧视垃圾的来源（A）与生成帧的联合深度（B）（2026-09-28，用户决定"两个都做，等 E13 结束，check 后继续"）

背景：E13 第 1 轮 0.11% 的空洞像素通过验收，只新增 838 个高斯，拟合后的侧视与 E5c 几乎相同；E11 / E12 侧视复查里，E5c 的竖直白条和彩色碎片在 5 个模型中位置都不变。两个假设，各做一个只读诊断，都只在 val056 上、E13 结束后运行，不训练、不生成：

- **A. 侧视垃圾来自 FRONT 从未约束过的高斯**（`scripts/diagnose_floaters.py`，`dashrecon/gen/floaters.py`）。
  - E5c 的配置用 `near_randoms: 100000`、`far_randoms: 100000` 初始化了随机位置、**随机颜色**（`torch.rand`，`models/trainers/scene_graph.py`）的高斯；被表面挡住的那些在训练视角里权重接近 0，得不到梯度，颜色保持随机。另外 SH 高阶项在未见过的方向上外推，也会给出饱和颜色。
  - 逐个高斯统计：训练视角（只算动态、天空之外受光度 loss 约束的像素）的混合权重总和（support，像素单位，用梯度技巧求出）；沿最近训练相机视线的拉长比（needle）；DC 颜色饱和度。
  - 在内存里改参数、同一 checkpoint 渲染固定相机（frame 96 / 104 / 112 × 偏航 −90 / −60 / 0 / 60 / 90）：原样、只用 DC 颜色、去掉低 support、去掉 needle、三者合并；另渲 support 和 needle 的热力图。每个变体同时算 FRONT 留出帧 PSNR / LPIPS，看清理对已观测视角的代价。
  - 判读：如果去掉低 support 高斯后碎片消失、FRONT 几乎不变，就说明补全之前必须先清掉这层；清掉后露出的是空洞（可补）还是新的垃圾，决定下一步。
- **B. 验收几乎全拒，是因为逐帧单目深度彼此不一致**（`scripts/run_depth_probe.py`；`depth_views.py` 另存 MapAnything 返回的位姿并记录它与给定位姿的偏差）。
  - 同一组 E13 第 1 轮生成帧、同一套验收参数（从 E13 自己的 `validation/r0` 读取：平移参考、基线 ≥ 1、视差 ≥ 0.5°、深度 10%、颜色 0.12、≥ 2 票、冲突 ≤ 25%），只换深度：MoGe-2 / MapAnything（给定内参和位姿，121 帧一次联合推理）× 全局尺度 / 逐空洞边界尺度，共 4 组。
  - 报告空洞内通过 / 未知 / 冲突比例（总体和按 |偏航| 分组），以及对齐后深度与已观测区域渲染深度的偏差。通过比例只说明生成 RGB-D 自洽，不说明隐藏内容正确。
  - 注意：MapAnything 只把给定位姿当条件，深度与它自己输出的位姿一致；偏差记在 `given_pose_agreement`，偏差大时通过率的提升不能直接归因于"联合深度"。
