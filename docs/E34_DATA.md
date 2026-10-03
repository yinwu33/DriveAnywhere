# E34 训练数据：驾驶领域修复模型的训练对

E34（docs/PLAN_AFTER_E30.md）在 Waymo training 上微调 NVIDIA Fixer，让它学会把"只用 FRONT 重建、在侧相机位置渲染出来的画面"修成真实的侧相机图像。用户已按 DECISIONS S 批准训练。本文件只记录**训练数据**的构造；微调由主线完成。

状态：**进行中**（试点 2 段完成，批量运行中）。统计数字在最后一节，跑完后更新。

## 1. 数据划分与隔离

- 候选：`data/waymo_train_day_moving.txt` 的 428 段（白天、行驶 ≥ 60 m 的 training 段）。
- 排除：`data/dashrecon/scene_selection/revisits.json`（D-M1）里任何一对中出现的段，共 43 段，其中 29 段在候选里，包括 MT1 的 mt1a（368）、mt1b（662），以及和 val056 / val087 共享道路的 training 段。5 个开发场景是 validation 段，不在候选里（脚本里断言过），永远不用。
- 第一批 100 段：按地点比例分配（最大余数法），每个地点内用 `numpy.random.default_rng(0)` 抽取，再整体打乱，所以列表的任意前缀在地点上也是分散的：
  - phx 42、sf 44、other 14（排除后候选 166 / 175 / 58）。
- 结果写在 `data/e34/segments.json`（候选数、排除的段及原因、选中的 100 段及处理顺序）。
- 训练 / 测试按段划分（`scripts/e34_manifest.py --test_every 20`：每 20 个通过的段里第 20 个是 test），同一段不会同时出现在两边。

## 2. 每段的流程

所有步骤都是独立的命令行脚本，输入输出都是磁盘文件；`scripts/e34_batch.py` 只负责调度和记录。

| 步骤 | 脚本 | 环境 | 说明 |
|---|---|---|---|
| 下载 | `e34_fetch.py` | 任意 | `rclone cat --tpslimit 8 ... \| tar -x`，不落盘 tar。图像进 `data/e34/raw/<idx>/images/`；GT 的 `extrinsics/ intrinsics/ ego_pose/` 移到单独的 `data/e34/gt/<idx>/`，只有评测侧诊断读它。GT 框掩码、objects 不解包 |
| 掩码 | `e34_masks.py` | masks | 与 §13.4 相同的模型和参数（Grounding DINO + SAM 2.1，SegFormer-B5）。FRONT 每帧，4 个目标相机每 2 帧。见 §3 |
| 联合自标定 | `e34_calib.py` | sfm | 5 个相机的原始图像放进同一个 GLOMAP，每个相机一个 RADIAL 相机（相机内所有帧共享），不加刚体 rig 约束。见 §3 |
| 去畸变 | `e34_undistort.py` | sfm | FRONT：用 FRONT 的 K 去畸变，帧重新编号成 0..N-1，单独成一棵只有 FRONT 图像的树；掩码按同一映射变换。目标相机：用各自估计的 K 和畸变去畸变，缩放到 576×1024，生成损失掩码 |
| GT 诊断 | `e34_diag_calib.py` | sfm | **评测代码**，读 GT，只用于报告标定质量，不参与任何决定 |
| FRONT 深度 | `run_pose.py --camera_dir` | mapanything | D13 的做法：给定 FRONT 的内参和 SfM 位姿，MapAnything 补稠密深度，SfM 位姿按稀疏点深度比缩放并转到重力对齐的世界系 |
| a：点云对 | `e34_points_pairs.py` | main | FreeVS 式，见 §4 |
| b：3DGS 对 | `run_fusion.py`、`run_mesh.py`、`e34_train_gs.py`、`e34_gs_pairs.py` | main / nksr | Fixer 的 Cross reference：只用 FRONT 训练 E5f 式 3DGS，渲染到每个目标相机，见 §4 |
| 收尾 | `e34_finalize.py` | 任意 | 每段的 `meta.json`，保留标定，删除中间产物 |
| 清单 | `e34_manifest.py` | 任意 | `data/e34/manifest.json` 和 Fixer 格式的 `fixer_points.json`、`fixer_gs.json` |

重建（深度、融合、网格、3DGS）只读 FRONT 去畸变树和由它导出的产物，GT 隔离检查（`dashrecon/train/guard.py`）在训练前运行，图像根目录只允许这棵 FRONT 树。目标相机的图像只在去畸变那一步被读取，用来生成目标图和掩码。

## 3. 联合自标定

- **为什么不用 GT 外参**：按简报，侧相机的位姿和内参全部由图像估计，不读 Waymo 标定、自车位姿。
- **匹配**（`dashrecon/e34/calib.py`）：
  - 相机内：顺序匹配，FRONT 重叠 20 帧（同 D13），目标相机重叠 10 个采样帧（即 20 帧），另加二次方跨度；
  - 相机间：按名字相邻的四对（FRONT–FRONT_LEFT、FRONT–FRONT_RIGHT、FRONT_LEFT–SIDE_LEFT、FRONT_RIGHT–SIDE_RIGHT），A 相机每隔 2 张取一张，和 B 相机时间相差 ≤ 10 帧的所有图像配对。哪些图像看到同一处，交给特征匹配和几何验证决定。
- **不加 rig**：Waymo 各相机不是同时触发的（随激光雷达旋转依次曝光），按图像独立估计位姿，每张图像就在它实际拍摄的位置。
- **SIDE 相机的主点**（试点中发现）：GLOMAP 把所有相机的主点固定在图像中心。SIDE 相机的 1920×886 图像是传感器的裁切，真实主点的 cy 约 250（中心是 443）。主点固定时，重建用约 5° 的俯仰加偏心畸变来吸收这个偏差：wt646 上 SIDE → FRONT 的相对旋转误差 5.3–5.4°（评测侧 GT 诊断）。所以在 GLOMAP 之后加一次 bundle adjustment：SIDE 两台相机的焦距、畸变、主点放开，其余相机内参固定，位姿和点都放开（Huber 损失，同 GLOMAP 自己的 BA）。效果（试点）：
  - wt646：SIDE cy 从 442 变到 268 / 270（GT 250 / 256），SIDE → FRONT 旋转误差 5.3 / 5.4° → 0.57 / 0.66°，FRONT_LEFT / RIGHT 不变（0.35°）；
  - wt683：SIDE cy 260 / 273（GT 238 / 263），SIDE 的重投影误差 0.75 → 0.49 / 0.46 px，各相机相对 FRONT 的旋转误差 0.3–0.9°。
  - 只对 SIDE 放开主点：FRONT_LEFT / RIGHT 也放开时它们的误差反而变大（0.35° → 0.7°），因为它们的真实主点离中心只有约 10 px。
- **通过条件**（简报）：FRONT 最长连续注册段 ≥ 90% 的帧，且 FRONT 去畸变越界 ≤ 2 px；每个目标相机注册 ≥ 80% 的图像。不通过就跳过并记录原因，不回退到 GT。如果通过率低于约 70%，停下来报告。
- FRONT 只在最长连续注册段上重建（帧重新编号，`meta.json` 里的 `frame_offset` 记原始帧号）。

## 4. 训练对

两种类型，都在 576×1024（Fixer 的 `resize_576x1024`，不保持长宽比）：

- **目标图**：真实侧相机图像，用该相机自己估计的 K 和畸变去畸变（同一个 K，越界处复制边缘像素），INTER_AREA 缩小，JPEG 95。
- **损失掩码**（PNG，255 = 不参与损失）：该相机的 Grounded-SAM-2 动态掩码（输入里不可能有这些物体），以及去畸变时采样到原图之外的像素。缩小时只要有一个源像素被掩就算被掩。
- **输入 a，点云**（`e34_points_pairs.py`）：
  - FRONT 的 MapAnything 深度，去掉天空、动态物体、置信度低于 30 分位、深度 ≥ 60 的像素（与 Phase 5 相同的规则），颜色取去畸变 FRONT 图像；
  - 时间窗口按行驶距离而不是帧数：FRONT 相机在当前帧之后 40 / 之前 10 个世界单位（MapAnything 单位，约为米的 0.8–0.9）内的帧，最多 40 帧均匀抽取。车侧的内容是 FRONT 在 10–30 m 之前看到的，按帧数开窗会随车速变化。试点里 ±10 帧窗口时 SIDE 相机的覆盖率只有 0.00–0.02，改成按距离后为 0.21–0.32；
  - 点按"深度网格一个像素在世界里的大小"投影到目标相机上的足迹溅射（最接近的奇数宽度，1–5 px），z-buffer 取最近的点，没有点的地方是黑色。
- **输入 b，3DGS**（`e34_train_gs.py`、`e34_gs_pairs.py`）：与 `train_gs.py` 的 E5f 完全相同的设置（NKSR 网格初始化、网格深度 / 法向正则、flatten 和最大轴正则，30000 步，960×640），只关掉训练结束时的视频渲染；渲染时用目标相机自己的内参，位姿是联合标定的位姿经 `run_pose.py` 同样的缩放和世界系变换（`dashrecon/e34/geometry.py`），不做 CamPose 修正；天空模型按目标相机的光线重算。
- 每 2 帧一个目标（目标相机在标定时就是每 2 帧取一张）。

## 5. 产物与磁盘

```
data/e34/
├── segments.json            选段
├── manifest.json            所有训练对（相对路径）、划分、每段摘要、按类型和相机的计数
├── fixer_points.json        Fixer PairedDatasetV2 格式（绝对路径），多一个 "mask" 键，原版加载器会忽略
├── fixer_gs.json
├── batch_points.json / batch_gs.json   批处理状态（失败的段、每步耗时）
├── diag/<sid>_calib_gt.json 评测侧 GT 诊断
└── pairs/<sid>/
    ├── target/<t>_<cam>.jpg、mask/<t>_<cam>.png、points/<t>_<cam>.jpg、gs/<t>_<cam>.jpg
    ├── targets.json、points.json、gs.json      每张图的参数、覆盖率
    ├── calib/                   联合标定（每个相机的 camera.json、frames.txt、SfM 位姿、世界系位姿），不含 COLMAP 模型
    └── meta.json                每步耗时、注册统计、重投影误差、各步 commit、训练对计数
```

中间产物（原始图像、原图掩码、COLMAP 模型、FRONT 去畸变树、深度、融合点云、网格、3DGS checkpoint）在一段写完训练对后删除；处理顺序里前 30 段中通过的段保留重建输入，等 b 用完再删。日志：`results/_logs/e34_<sid>.log`（每段所有步骤）、`results/_logs/e34_batch_<phase>.log`（调度）。

## 6. 复现

```bash
export CUDA_HOME=$HOME/cuda-12.6 TORCH_CUDA_ARCH_LIST=9.0 HF_HUB_OFFLINE=1   # 不设 PYTORCH_CUDA_ALLOC_CONF，见 §7
# 选段
.venvs/main/bin/python scripts/e34_select.py --candidates data/waymo_train_day_moving.txt \
    --revisits data/dashrecon/scene_selection/revisits.json --num 100 --seed 0 --out data/e34/segments.json
# a：100 段（可中断，重新运行会从已完成的步骤继续）
.venvs/main/bin/python scripts/e34_batch.py --phase points --segments data/e34/segments.json --root data/e34 \
    --venvs .venvs --cuda_home $HOME/cuda-12.6 --log_root results/_logs \
    --remote onedrive:/Projects/P05_DriveAnywhere/waymo_processed/training --tpslimit 8 --max_raw 12 \
    --calib_jobs 4 --calib_threads 14 --keep_first 30 --gs_count 20 --min_free_gb 161 --max_failures 8
# b：前 20 个通过的段
.venvs/main/bin/python scripts/e34_batch.py --phase gs ...（同上的参数）
# 清单
.venvs/main/bin/python scripts/e34_manifest.py --segments data/e34/segments.json --pairs_root data/e34/pairs \
    --test_every 20 --prompt "remove degradation"
# 目测
.venvs/main/bin/python scripts/e34_contact_sheet.py --pairs_dir data/e34/pairs/wt683 --cams 1 2 3 4 \
    --times 60 140 --width 512 --out data/e34/vis/wt683.jpg
```

单步命令见各脚本的 docstring；批处理用的参数在 `scripts/e34_batch.py` 顶部。`--min_free_gb 161`（10⁹ 字节）即 `df -h` 的 150 GiB。手动在 GPU 上跑东西时先建 `data/e34/gpu.lock`，批处理的 GPU 通道会等它删掉。

## 7. 与简报的差别和遇到的问题

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 不能用**：这块卡是 vGPU（H100L-94C），所有 venv 第一次把张量放到 GPU 上就报 "CUDA driver error: operation not supported"。不设这个变量。
- **github.com 对本机返回 503**（10-03 下午起，api.github.com 和 HuggingFace 正常）：torch.hub 加载 MapAnything 用的 DINOv2 时，即使已有缓存也会先访问 github.com 判断默认分支，于是 MapAnything 全部加载失败。`dashrecon/pose/mapanything.py` 在加载模型期间把 `facebookresearch/dinov2` 直接解析到已缓存的 main（缓存不存在就报错），其他仓库不受影响。
- **MapAnything 显存**：在共享的 H100 上它按空闲显存自动选 dense head 的 minibatch，峰值 46.6 GB，超过本任务 40 GB 的上限。`run_pose.py --minibatch_size 16` 固定 minibatch 后峰值 26.2 GB；深度在两边都有效的像素上逐像素相同，只有少数帧边缘约 0.5% 像素的有效性不同。
- **掩码在 GPU 上做后处理**（`dashrecon/e34/masks.py`）：SAM 掩码的上采样和圆盘膨胀原来在 CPU 上（膨胀一张 1920×1280 要 0.2 s）。改到 GPU 后与原后端逐像素相同（fp32，30 张图核对）。另外开了 TF32：动态掩码约 3e-5、天空约 5e-6 的像素不同，SegFormer 快 2.7 倍。GPU 空闲时每张图 0.36 s，和主线共用时 1.0–1.4 s。
- **原图掩码变换到去畸变图像**，而不是在去畸变图像上重新分割（L 节的做法）。去畸变只是重采样，变换后的掩码就是去畸变图像的掩码（只差最近邻取整），省掉一半分割时间。
- **点云窗口按行驶距离**而不是简报举例的 ±10 帧，原因见 §4。
- **SIDE 主点的额外 BA**，见 §3。

## 8. 统计

（批量跑完后填写：选中 / 通过 / 跳过及原因、每种类型和相机的训练对数、每步耗时、磁盘占用、示例图路径。）

试点（wt646、wt683，都在 SF 市区）：

| | wt646 | wt683 |
|---|---|---|
| 注册（FRONT / FL / FR / SL / SR） | 198 / 99 / 99 / 94 / 99 | 199 / 100 / 100 / 100 / 100 |
| 每个相机的重投影误差（FRONT / FL / FR / SL / SR） | 0.70 / 0.63 / 0.64 / 0.52 / 0.52 px | 0.61 / 0.61 / 0.60 / 0.49 / 0.46 px |
| 焦距 估计 / GT | 0.99–1.01 | 0.99–1.00 |
| 目标相机相对 FRONT 的旋转误差（中位数，FL / FR / SL / SR） | 0.37 / 0.34 / 0.61 / 0.68°（不做主点 BA 时 SL / SR 5.3 / 5.4°） | 0.91 / 0.31 / 0.74 / 0.45° |
| 目标相机位置误差（Sim(3) 对齐后，中位数） | 0.06–0.19 m | 0.15–0.32 m |
| GLOMAP（16 线程，含主点 BA） | 21.6 min | 20.2 min |
| 重建尺度：MapAnything 单位 / GT 米（评测侧 Sim(3)） | 0.89 | 0.83 |
| 点云对的平均覆盖率（FL / FR / SL / SR） | 0.53 / 0.50 / 0.22 / 0.21（v1 标定） | 0.59 / 0.35 / 0.32 / 0.21 |
