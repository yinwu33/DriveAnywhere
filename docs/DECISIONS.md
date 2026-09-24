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
