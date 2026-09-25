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
