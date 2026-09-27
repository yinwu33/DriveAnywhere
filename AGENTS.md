# AGENTS.md — 单目前视（dashcam 式）街景重建原型

> 本文件是 coding agent 的工作说明。开始任何任务前先完整阅读。
> 遇到本文件与上游 repo 实际代码不一致时，**以代码为准，并在 `docs/DECISIONS.md` 记录差异**，不要猜测 API。

---

## 0. Agent 工作规则

1. 每次只推进一个 Phase 中的一个任务；完成后更新第 12 节的 checklist。
2. 任何外部库的函数名、参数、配置键，**必须先在源码或 README 中确认**再使用。不确定就停下来，在 `docs/OPEN_QUESTIONS.md` 记录问题。
3. 对上游 drivestudio 的修改保持最小化：新功能放在独立包 `dashrecon/` 中，上游只做必要的钩子改动，并在 `docs/UPSTREAM_PATCHES.md` 逐条记录。
4. 每个模块都要能通过命令行单独运行，输入输出都是磁盘文件（见第 5 节数据约定）。
5. 所有随机过程固定种子；所有实验结果写入 `results/`，格式见第 8 节。
6. 不要为了"跑通"而静默降级（例如检测到 GT 不可用就偷偷用 GT 替代）。任何回退都必须显式报错或打印警告。
7. 不要使用防御性代码，比如get(key, default), try except, 部分if else的防御性代码。如果某个条件可能失败，应显式检查并报错，而不是静默降级。
8. **每完成一个任务项就 commit**（代码、配置、文档一起），不要把多个任务攒到一起提交。只 commit 不 push。
   - 结果的 `meta.json` 会记录启动时的 commit；代码目录有未提交改动时记为 `-dirty`，无法凭 commit 复现。
   - 所以：长任务启动前工作区必须是干净的；任务运行期间要改代码，就先在仓库外开发，改完后"复制 + commit"一步完成，不让正在排队启动的任务碰到脏的工作区。

---

## 1. 项目目标

**目标**：从单个前视相机视频重建静态街景（3D Gaussian Splatting），使得 ego 车辆可以在重建场景中沿原轨迹及小幅偏离轨迹渲染相机画面。最终用于 dashcam 视频；当前阶段用 Waymo Open Dataset 的 `FRONT` 相机模拟 dashcam，利用 Waymo 的真值逐模块定位误差来源。

**2026-09-25 目标扩展（DECISIONS D16、D17）**：重点是静态场景质量。要能从**自由视角**渲染（平移、升降、转向，不只是横移），原视频没拍到的区域由生成模型补全（Phase 9），补全内容在不同视角之间要一致。先在 Waymo 上打通，Mapillary 部署放到以后（D19，见第 14 节）。

**非目标（当前阶段明确不做）**：

- ~~不做任何生成式补全（扩散模型、视频生成、score distillation 等）。~~ **2026-09-24 范围变更（DECISIONS D10）：**Phase 8 开始允许生成式蒸馏和后处理。**2026-09-27 用户更新（DECISIONS S）：取消训练和微调的绝对禁令，但优先使用已有预训练模型跑通；只有现有方法经对照实验仍无法满足明确目标时，才允许训练或微调作为补充。生成、几何验证和重建可交替进行。
- ~~不重建未被观测的区域。~~ **2026-09-25 范围变更（DECISIONS D16）：**Phase 9 用生成模型补全未观测区域，默认冻结现成权重，训练/微调条件见 §2；补全内容必须标注为生成内容。
- 不重建动态物体（只掩掉，不建模）。
- 不做风格/外观迁移。
- 不做闭环仿真集成（后续阶段再考虑）。

---

## 2. 硬约束

| 约束 | 说明 |
|---|---|
| 算力 | 单张 NVIDIA RTX A6000（48GB），即本机实际硬件。 |
| 现成模型优先 | **已取消训练/微调的绝对禁令（2026-09-27，用户决定）**。先以已有模型和发布权重完成基线、集成修复和一致性验证；仅当有可复现对照证明现有方法达不到明确目标时，才用领域适配、LoRA、测试时微调或训练补足缺口。先在 `docs/DECISIONS.md` 写明不足、基线、目标、数据划分和预算，再实施最小必要训练。逐场景 Gaussian、相机和深度优化可正常进行。 |
| 训练输入 | 非 oracle 实验中，重建只允许使用 `FRONT` 相机图像。其他相机只用于评测。 |
| 真值隔离 | 非 oracle 实验中，重建流程**不得读取** GT 位姿、LiDAR、GT 3D 框。GT 只能被评测代码读取。见第 10 节。 |
| 不用标定 | 非 oracle 实验中**不得使用 Waymo 标定**：内参、畸变、外参、相机高度都不行。在原始（未去畸变）图像上重建，内参由模型估计。见 DECISIONS D3。 |

---

## 3. 技术栈与外部依赖

以下为候选，agent 在 Phase 0 需逐一确认：仓库地址、许可证、安装方式、推理显存、输入分辨率/帧数上限。确认结果写入 `docs/DEPENDENCIES.md`。

| 用途 | 候选 | 备注 |
|---|---|---|
| 基础重建框架 | **drivestudio**（OmniRe 官方代码库） | 主仓库，fork 后开发。确认其 Waymo 预处理、单相机训练配置、天空掩码、动态掩码的实现方式。 |
| 高斯光栅化 | gsplat | drivestudio 的后端；2DGS 备选。 |
| 位姿 + 点图（主选） | MapAnything | 直接输出度量尺度，优先尝试。 |
| 位姿 + 点图（备选） | VGGT、π3、MegaSaM | VGGT/π3 需另行恢复尺度；MegaSaM 针对动态视频。 |
| 单目度量深度（辅助） | MoGe-2、Metric3D v2、UniDepth | 用于尺度校准或对比。 |
| 动态物体分割 | Grounded-SAM-2 或 SAM 3（文本提示） | 类别：car, truck, bus, motorcycle, bicycle, person。 |
| 天空 / 路面分割 | drivestudio 自带天空掩码；路面用 Cityscapes 语义分割模型（如 SegFormer / Mask2Former） | 确认 drivestudio 使用的分割模型。 |
| 表面重建 | NKSR（nv-tlabs/NKSR，预训练模型） | 需要法向或传感器位置；大场景需分块。 |
| 点云处理 | Open3D | 降采样、离群点去除、法向估计。 |
| 位姿评测 | evo | Sim(3) Umeyama 对齐、ATE、RPE。 |
| 图像评测 | torchmetrics / lpips | PSNR、SSIM、LPIPS。 |

**已选定**（2026-09-24，见 `docs/DECISIONS.md` D5、D6 和 D 节）：
- 位姿和点图：MapAnything，使用 `facebook/map-anything-apache` 权重。
- 动态分割：Grounded-SAM-2。
- 天空和路面：HF transformers 版 SegFormer-B5 Cityscapes，一次推理同时输出天空（类 10）和路面（类 0）掩码。
- 环境工具：**uv**，不用 conda。

---

## 4. 仓库结构

在 drivestudio fork 中新增以下内容，不要重排上游目录：

```
<drivestudio-fork>/
├── AGENTS.md
├── docs/
│   ├── DECISIONS.md          # 设计决策与"代码与本文件不一致"的记录
│   ├── DEPENDENCIES.md       # Phase 0 的依赖确认结果
│   ├── OPEN_QUESTIONS.md     # 待确认问题
│   └── UPSTREAM_PATCHES.md   # 对上游的每一处改动
├── envs/                     # uv 环境脚本：setup_{main,waymo,mapanything}.sh、main-requirements.txt
├── dashrecon/
│   ├── __init__.py
│   ├── io.py                 # 第 5 节数据约定的读写函数（唯一入口）
│   ├── provenance.py         # meta.json 里的 git commit（代码有未提交改动时加 -dirty）
│   ├── scenes.py             # 开发场景列表与划分
│   ├── viewer/index.html     # Phase 3 网页可视化模板（DashRecon Pose Review），见第 13 节
│   ├── pose/                 # Phase 3：位姿 + 点图后端（可插拔）
│   │   ├── base.py           # 抽象接口
│   │   ├── world.py          # 重力对齐的世界系与尺度策略
│   │   ├── gt.py             # oracle：读取 Waymo GT（尚未实现）
│   │   ├── mapanything.py
│   │   └── vggt.py           # 等（尚未实现）
│   ├── masks/                # Phase 4：动态与天空掩码后端
│   │   ├── base.py
│   │   ├── gt_boxes.py       # oracle：GT 3D 框投影（尚未实现）
│   │   ├── sam_text.py       # Grounding DINO + SAM 2.1（transformers）
│   │   └── segformer.py      # 天空与路面（SegFormer-B5 Cityscapes）
│   ├── fusion/               # Phase 5：点云融合与清理
│   │   ├── backproject.py    # 深度反投影与投影
│   │   ├── consistency.py    # 多视图深度一致性过滤
│   │   ├── cleanup.py        # 降采样、离群点去除
│   │   └── road.py           # 路面平滑
│   ├── mesh/                 # Phase 7：NKSR
│   │   └── nksr_recon.py
│   ├── train/                # Phase 6：与 drivestudio 训练的对接
│   │   └── hooks.py
│   └── eval/                 # Phase 1：评测工具
│       ├── image_metrics.py
│       ├── cross_camera.py
│       ├── geometry.py
│       ├── pose_metrics.py
│       └── lateral_render.py
├── data/data/                # 符号链接 → Waymo perception v1.4.3 原始 tfrecord（共享盘，只读）
│                             #   training 无 LiDAR；validation 有 LiDAR；两者都不是 scene-flow 版本
├── configs/dashrecon/        # 实验配置（E0–E5）
├── scripts/                  # 每个 Phase 的命令行入口：scene_stats、run_pose、run_masks、run_fusion、run_mesh、vis_pose、vis_masks、vis_fusion、vis_mesh、diagnose_pose、build_viewer
├── tests/                    # 小型单元测试
└── results/                  # 实验输出（不入 git，只提交 summary）
```

---

## 5. 数据约定

**单位与坐标**：长度单位为米。相机坐标系统一使用 OpenCV 约定（x 右、y 下、z 前）。位姿为 camera-to-world 的 4×4 矩阵。如 drivestudio 使用其他约定，在 `dashrecon/io.py` 中统一转换，并在 `docs/DECISIONS.md` 记录。

**每个场景的中间产物目录**：

```
data/dashrecon/<scene_id>/<backend_tag>/
├── frames.txt            # 使用的帧索引（与 drivestudio 帧索引一致）
├── intrinsics.npy        # (3,3) float64；若逐帧不同则 (N,3,3)
├── poses_c2w.npy         # (N,4,4) float64
├── depth/{idx:06d}.npy   # (H,W) float32，米；无效值为 0
├── depth_conf/{idx:06d}.npy  # (H,W) float32，可选
├── mask_dynamic/{idx:06d}.png  # uint8，255 = 动态（需掩掉）
├── mask_sky/{idx:06d}.png      # uint8，255 = 天空
├── mask_road/{idx:06d}.png     # uint8，255 = 路面
├── points_fused.ply      # Phase 5 输出：xyz + rgb + normal
├── mesh_nksr.ply         # Phase 7 输出
└── meta.json             # 后端名、版本、参数、运行时间、显存峰值
```

`backend_tag` 示例：`pose-gt_depth-lidar`、`pose-mapanything_depth-mapanything`。所有读写只能通过 `dashrecon/io.py`。

---

## 6. 分阶段计划

每个 Phase 都有「产出」和「验收标准」。验收未通过不得进入下一 Phase。

> **当前执行顺序（DECISIONS D1、D8、D13、D16）：Phase 0 → 3 → 4 → 5 → 7 的网格重建（任务 1–3）→ 6（E3、E4、E5 一起训练）→ 8 → Phase 3 修正（相机自标定，E5c）→ 8（Fixer，E7）→ 9（生成式补全，E8）。**
> - 先搭非 oracle 的 FRONT 单目 pipeline。**Phase 1（评测工具）和 Phase 2（E0）延后。**
> - 延后期间，下文所有依赖 GT 的任务和验收（位姿指标、IoU、LiDAR Chamfer 等）都标为「延后」，改用 `docs/DECISIONS.md` E 节的临时验收标准。
> - Phase 1 任务 5（横向偏移渲染）不需要 GT，提前纳入 Phase 6。

### Phase 0：环境与数据

任务：

1. 仓库已 fork（`yinwu33/DriveAnywhere`）。用 **uv** 建环境：一个主训练 venv，另外每个冲突的后端各一个 venv（见 DECISIONS D 节）。记录 CUDA、PyTorch、gsplat 版本。
2. 从**本地 validation**（有 LiDAR，供以后评测）中选 **5 个开发场景**：至少 2 个以静态为主的街道、2 个有较多动态车辆、1 个有明显坡度或弯道。场景 ID 写入 `dashrecon/scenes.py`。
   - 预处理需要上游补丁 P1（文件列表参数化）。
   - `--process_keys` 不含 `lidar`：上游 lidar 处理要求 scene-flow 版本，而 pipeline 用不到 LiDAR。
3. 确认 drivestudio 中以下几点：`FRONT` 相机的索引；只用单相机训练的配置方式；天空掩码和动态掩码的来源；如何关闭动态物体建模、只训练静态背景。已在 DECISIONS B 节确认（C1、C7、C11）。
4. 完成 `docs/DEPENDENCIES.md`。

产出：可运行环境；5 个预处理完的场景；依赖确认文档。

验收：
- 各 venv 都能 import 各自的依赖；gsplat 1.3.0 CUDA 扩展编译通过，并能跑通一次光栅化；5 个场景的 FRONT 图像预处理完成。
- 原验收是"drivestudio 默认配置完整训练"。但默认配置强制读 LiDAR（C5），而开发场景不处理 LiDAR，所以完整训练的验证移到 Phase 6，在 P2 落地后进行。

### Phase 1：评测工具（原定先于所有实验；现延后，见 D1）

任务：

1. **插值视角评测**：训练时每隔 10 帧留出 1 帧 `FRONT` 图像作为测试帧（与 OmniRe 协议一致）。指标：PSNR、SSIM、LPIPS。动态区域和天空在计算时掩掉，并同时报告不掩的结果。
2. **跨相机外推评测**：用 `FRONT_LEFT`、`FRONT_RIGHT` 相机作为 held-out 偏离视角。
   - 重叠区域掩码：渲染累积不透明度 > 0.5 的像素（阈值可配置）。
   - 颜色对齐：在重叠区域上对渲染图做逐通道仿射拟合后再算 PSNR/SSIM；LPIPS 同时报告对齐前后。
   - 渲染这些视角时使用真值相机位姿（评测允许读取 GT）。对于估计位姿的实验，先用 Sim(3) 把估计轨迹对齐到 GT，再用对齐后的变换放置评测相机。
3. **几何评测**：把 LiDAR 投影到留出的 `FRONT` 帧，与渲染深度比较，范围 0–80 米。指标：AbsRel、RMSE、δ<1.25。点云与 mesh 对 LiDAR 聚合点云计算 Chamfer 距离和 precision/recall（阈值 0.2 米、0.5 米）。
4. **位姿评测**：Sim(3) 对齐后的 ATE、RPE（平移/旋转）、恢复的尺度比。
5. **横向偏移渲染**：沿相机 x 轴偏移 0.5 / 1.0 / 2.0 米，渲染整段轨迹视频，用于定性检查。
6. `scripts/summarize.py`：汇总 `results/` 下所有 `metrics.json`，生成 markdown 表格。

产出：`dashrecon/eval/*`，对应脚本与单元测试。
验收：用 drivestudio 默认输出跑通全部评测并生成汇总表。

### Phase 2：Oracle 上界（实验 E0；延后，见 D1）

任务：单 `FRONT` 相机，GT 位姿，LiDAR 初始化与深度监督，GT 框投影的动态掩码，只训练静态背景。

产出：5 个场景的 E0 结果。
验收：插值视角指标合理（可与 drivestudio 公开结果对照量级）；外推与横向偏移结果已记录。

### Phase 3：位姿与点图估计

任务：

1. 在 `dashrecon/pose/base.py` 定义接口：输入图像序列和（可选）内参；输出第 5 节格式的内参、位姿、深度、置信度。
2. 实现 `gt.py`（oracle）和至少一个估计后端（优先 MapAnything）。
3. 处理帧数与显存限制：若单次推理放不下全部帧，实现带重叠的分块推理，并用重叠帧做 Sim(3) 对齐拼接。记录分块边界处的误差。
4. 尺度策略（可配置），**默认 `model`**：
   - `model`：直接使用模型输出的度量尺度；
   - `camera_height`：用路面掩码拟合地平面，按已知相机离地高度校准。**本阶段不用**：D3 禁止读取标定，暂时没有高度来源，见 OPEN_QUESTIONS 11；
   - `oracle`：Sim(3) 对齐到 GT（仅用于上界分析，结果必须标注为 oracle）。**延后**，随 Phase 1 一起做。
5. 内参：由模型估计，可以是逐帧的。"使用标定内参"的选项**延后**（D3）。
6. 世界系：把估计位姿转换到"前-左-上"的度量世界系。上方向先取相机 −y 轴的平均，再用路面平面拟合细化；这是 drivestudio 的 AABB 和 EnvLight 所要求的（DECISIONS C6、D3）。

产出：各后端的中间产物；位姿评测结果**延后**。
验收：至少一个估计后端在 5 个场景上全部跑通。位姿指标**延后**；在此之前按 DECISIONS E 节的临时标准验收。

### Phase 4：动态与天空掩码

任务：

1. `gt_boxes.py`（oracle）：两种模式——`all`（掩掉所有物体）和 `moving`（仅速度超过阈值的物体，默认 0.5 m/s）。**延后**，随 Phase 2 一起做。上游的 `dynamic_masks/` 用的是轴对齐 2D 框和 1.0 m/s 阈值，与这里的定义不同，不能复用（C11）。
2. `sam_text.py`：文本提示分割，后端用 Grounded-SAM-2（D5），可配置掩码膨胀像素数（默认 5）。
3. 天空与路面掩码：用 HF transformers 版 SegFormer-B5 Cityscapes 一次推理同时生成（D6），取代上游基于 mmcv 的 `extract_masks.py`。
4. 统计每帧被掩掉的像素比例。与 GT `moving` 掩码的 IoU 统计**延后**。

产出：各掩码后端的输出与统计。
验收：SAM 掩码、天空和路面掩码在 5 个场景上跑通，像素比例统计已记录。

### Phase 5：点云融合与清理

任务（每一步可单独开关，便于消融）：

1. 反投影：用深度、位姿、内参生成逐帧点云，剔除动态和天空像素，以及置信度低于阈值的像素。
2. **多视图深度一致性过滤**：把每帧深度投影到前后 K 帧（默认 K=4），深度相对误差 < 5% 视为一致；至少在 2 帧中一致的点才保留。
3. 体素降采样（默认 5 厘米）和统计离群点去除。
4. 路面平滑：取路面掩码内的点，拟合平滑高度场（分段平面或 B 样条），将路面点投影到拟合曲面上。
5. 法向估计：用相机中心作为朝向参考统一法向方向。
6. 输出 `points_fused.ply`，不包含留出帧（D7）。对 LiDAR 计算 Chamfer 与 precision/recall 的部分**延后**。

产出：融合点云，以及每一步前后的点数和截图；几何指标**延后**。
验收：
- 原标准是"一致性过滤后的点云几何指标优于未过滤版本；如果没有，先排查再继续"，**延后**到评测补上之后。
- 在此之前按 DECISIONS E 节的临时标准验收。

### Phase 6：高斯重建集成（实验 E1–E4；当前只跑 E3、E4）

任务：

1. 在 drivestudio 中接入 `dashrecon` 产物：
   - 用 `points_fused.ply` 初始化；
   - 读取估计位姿和内参；
   - 用估计深度做逆深度 L1 监督（权重可配置）；
   - 用动态和天空掩码屏蔽光度损失。
   - 上游钩子见 `docs/UPSTREAM_PATCHES.md` P2–P4；位姿、深度和掩码的注入用 pixel source 子类实现（`data.pixel_source.type`）。
2. 关闭动态物体建模，只训练静态背景：新写一份只含 Background + Sky 的 yaml，放在 `configs/dashrecon/`（C1）。
   - 其余设置：CamPose、Affine 沿用上游默认开启（D4）；`test_image_stride: 10`（D7）；`undistort: False`（D3）；分辨率 960×640。
3. 训练时不得读取任何 GT（见第 10 节）；在训练入口加断言。
4. 按第 7 节实验矩阵运行。**当前只跑 E3、E4**；E1、E2 需要 GT，延后。
5. 沿原轨迹以及沿相机 x 轴横向偏移 0.5 / 1.0 / 2.0 m，渲染整段轨迹视频。这一项从 Phase 1 任务 5 提前而来。

产出：各实验的渲染结果；汇总表**延后**。
验收：
- E3、E4 在 5 个场景上完成训练和渲染（临时标准）。
- 原标准"E1–E4 完成且汇总表能看出每替换一个模块带来的下降"，**延后**到评测补上之后。

### Phase 8：生成式蒸馏与后处理（D10–D12，实验 E6、E5+pp）

任务：

1. 冻结的 SDXL base 1.0 + ControlNet depth（SDXL），条件为 NKSR 网格渲染的视差图；不训练、不微调（`dashrecon/gen/ggds.py`）。
2. E6（`scripts/train_ggds.py`）：从 E5 的 checkpoint 继续训练 6000 步。
   - 真实 FRONT 训练视角照常训练。
   - 偏离轨迹的视角用 GGDS 式生成的目标图，按 L1 + LPIPS 监督。这些视角取自训练帧，横移 ±0.5–2.5 m，偏航不超过 3°。
   - 目标图的生成：DDIM 反演到噪声等级 t，再去噪 5 步；每 1000 步整体重新生成一次；t 的上界从 0.7 线性退火到 0.4。
   - 天空和空白背景不参与生成 loss。
   - 实现细节和与 GGDS 的差别见 DECISIONS K。
3. E5+pp（`scripts/postprocess_frames.py`）：对 E5 的渲染结果逐帧做同样的去噪（t = 0.6），作为对照。
4. 生成内容在结果和网页中都要明确标注（`meta.json` 的 `generative: true`；网页列名旁的 gen 标记）。

验收（临时）：5 个场景跑通，并产出横向偏移的对比渲染。跨相机评测延后（D12）。

### Phase 9：未观测区域的生成式补全，面向自由视角（D16–D18，实验 E8）

目标：原视频没拍到的区域（横移后的视野之外、被停放车辆挡住的路面、转向后看到的侧面和背面等）用生成模型补全，并蒸馏回同一个 3D 模型，使得从任意合理视角渲染时，补全内容互相一致。

任务：

1. **目标视角**：不限于横移。在行驶走廊内定义一组平滑的相机轨迹：横移、升降、偏航和俯仰扫视（包括侧视和回看），并记录这些视角相对 FRONT 的偏离范围。尺度偏小的问题（OPEN_QUESTIONS 15）会影响"米"制定义，先用场景单位，并记录换算。
2. **空洞掩码**：沿目标轨迹渲染 RGB、不透明度和深度；空洞 = 不透明度低、又不是天空的像素（天空由 EnvLight 负责），适当膨胀。
3. **视频补全**：用 Wan2.1-VACE 做"视频 + 掩码"补全（D18）：先在本地用 1.3B 打通；14B 走本地显存卸载，或者经用户批准后走 API。视频按 81 帧一段、段间重叠。
4. **提升到 3D**：给补出的像素估计深度（单目深度，并在空洞边界与渲染深度对齐），在空洞处生成新的高斯，按置信度加权蒸馏（已观测区域不被改写）。
5. **3D 记忆，逐步外扩**：每补完一批轨迹就更新模型，下一批视角从更新后的模型渲染，已补的区域作为条件，只生成新的空洞（GEN3C / Difix3D+ 的思路），以保证不同视角补出的内容一致。
6. **评测**：跨相机检查扩展到 SIDE_LEFT / SIDE_RIGHT（偏航约 90°），用来检验大视角变化；另外报告空洞覆盖率。补全内容无法和真值逐像素对应，指标只能说明是否合理，所以要同时目测。
7. 生成内容在 `meta.json` 和网页中标注。

验收（临时）：5 个场景跑通；跨相机（含 SIDE 相机）指标相对 E5c / E7 有改善；自由视角的渲染里空洞明显减少，且不同视角看到的补全内容一致（目测）。

### Phase 9 修正：先验证跨视角一致性，再扩大生成（2026-09-27，DECISIONS S）

按以下顺序逐项执行，先在 val056 短轨迹上比较；不覆盖 E5c/E8/E10/U1/U3。

1. **跨轮生成记忆**：先用现成 GEN3C；将通过验证的历史生成 RGB + 深度实际 warp 到生成器缓存，而不只用于空洞判断或高斯 loss。真实观测优先，生成记忆不得冒充真实观测。按投影覆盖挑选参考帧，记录来源、覆盖率、冲突与退回情况。
2. **几何验收**：在生成结果进入场景之前检查跨视角深度重投影、遮挡和颜色一致性，记录有效支持与冲突。未经验证区域标为未知，拒绝或降低其监督权重；不能把缺乏观测记为通过。逐步加入联合深度优化和有依据的漂浮物清理。
3. **分开诊断生成与蒸馏**：同时输出直接生成帧、同视角 3D 拟合、未参与生成的新视角；固定输入、权重、种子和计算预算进行消融。短轨迹验证改善后才扩到大角度和更多场景。
4. **完整现成模型基线**：核对 UniWorld-View / CogNVS 的官方代码、权重、内存及输入条件，先运行可满足现有硬件条件的现成模型。资源不足必须记录，不以删掉关键组件后的结果代表原方法；未满足 §2 证据条件不得先行微调。
5. **评测提前**：分别报告真实可见区域保真、生成区域感知/几何/时间一致性、驾驶关键语义。U3 只在未参与训练的 SIDE_LEFT/RIGHT 上作外推对照；四相机平均值不得当作公平外推差距。全量 GT 几何评测仍与重建隔离。

本次只明确解除训练/微调的绝对禁令及阶段间交替优化限制；FRONT 单目输入、GT 隔离、当前静态场景范围和本机硬件约束保持。云资源/额外传感器/动态建模不因本路线自动启用。

### Phase 7：NKSR 表面重建（实验 E5，可选正则）

任务：

1. 安装 NKSR，使用预训练模型推理。输入 `points_fused.ply`，并提供传感器位置（每个点对应的相机中心）或法向。
2. 大场景分块重建，记录显存与耗时。
3. 输出 `mesh_nksr.ply`。对 LiDAR 做几何评测的部分**延后**。
4. E5（D9）：在网格面上初始化 Gaussian，训练时加入网格渲染的法向与深度正则，与 E4 对比。与 Phase 6 一起训练（D8）。

产出：mesh 与几何指标；E5 结果。
验收：mesh 能覆盖路面主体且无大面积穿洞；E5 与 E4 的对比已记录。

---

## 7. 实验矩阵

| ID | 位姿 | 初始化 / 深度监督 | 动态掩码 | 其他 | 性质 |
|---|---|---|---|---|---|
| E0 | GT | LiDAR | GT boxes（moving） | — | oracle 上界 |
| E1 | 估计 | LiDAR | GT boxes（moving） | — | 仅测位姿影响（半 oracle） |
| E2 | 估计 | 估计点图 | GT boxes（moving） | — | 位姿 + 深度影响（半 oracle） |
| E3 | 估计 | 估计点图 | SAM 文本分割 | — | **完全非 oracle** |
| E4 | 估计 | 估计点图 + Phase 5 清理 | SAM 文本分割 | — | 完全非 oracle |
| E5 | 同 E4 | 同 E4 | 同 E4 | NKSR 网格：在网格面上初始化 Gaussian，并用网格渲染的深度/法向正则（D9，参照 LSD-3D 的几何部分） | 完全非 oracle |
| E6 | 同 E5 | 同 E5 | 同 E5 | 从 E5 继续训练，对偏离轨迹的视角做 GGDS 式蒸馏：冻结的 SDXL + ControlNet depth，以网格视差为条件（D10–D12） | 完全非 oracle，**含生成内容** |
| E5+pp | 同 E5 | 同 E5 | 同 E5 | 不训练，对 E5 的渲染逐帧做同样的去噪（对照组） | 完全非 oracle，**含生成内容** |
| E4c、E5c | 自标定：GLOMAP 共享内参 + 径向畸变 + 位姿，去畸变图像（D13） | 同 E4、E5，但在去畸变图像上；深度来自给定内参和位姿的 MapAnything | 同 E4，但在去畸变图像上重新分割 | — | 完全非 oracle |
| E7 | 同 E5c | 同 E5c | 同 E5c | 从 E5c 继续训练，用 NVIDIA Fixer 修复的新视角渐进蒸馏（D14，Difix3D+ 做法） | 完全非 oracle，**含生成内容** |
| E5c+fx | 同 E5c | 同 E5c | 同 E5c | 不训练，对 E5c 的渲染逐帧用 Fixer 修复（对照组） | 完全非 oracle，**含生成内容** |
| E8 | 同 E5c | 同 E5c | 同 E5c | 从 E5c（或 E7）继续：自由视角轨迹上用视频模型补全未观测区域，以 3D 记忆逐步外扩并蒸馏（Phase 9，D16–D18） | 完全非 oracle，**含生成内容** |
| E9 | 同 E5c | 同 E5c | 同 E5c | 从 E5c 继续，复现 LSD-3D 的外观生成：自由视角上的 GGDS（冻结 SDXL + 深度 ControlNet，不微调），默认不用真实帧 loss，输出时可加延迟渲染（DECISIONS O） | 完全非 oracle，**含生成内容，以保真换真实感** |

注意：E1 使用估计位姿但 LiDAR 在 GT 世界坐标下，需要先把估计轨迹 Sim(3) 对齐到 GT，才能使用 LiDAR；该实验必须标注为半 oracle。

当前状态（2026-09-24）：
- **E0–E2 延后**（D1）。
- E3–E5 的"估计"位姿与内参都不使用标定（D3），CamPose、Affine 开启（D4）。
- E2→E3 之间还混有 GT `moving` 与"文本分割掩掉所有车辆"的差异，见 OPEN_QUESTIONS 5。

---

## 8. 结果记录

- 路径：`results/<exp_id>/<scene_id>/`
- 必须包含：
  - `metrics.json`：第 1 Phase 定义的全部指标，键名固定；
  - `config.yaml`：完整配置快照；
  - `meta.json`：git commit hash、各依赖版本、运行时间、显存峰值、是否使用 oracle 信息（布尔值 + 说明）；
  - `renders/`：留出帧渲染图、跨相机渲染图、横向偏移视频。
- `scripts/summarize.py` 生成 `results/SUMMARY.md`，按实验 × 场景列出指标及 5 场景均值。

---

## 9. 编码规范

- Python 3.10+，类型标注，函数写 docstring。
- 环境用 **uv** 管理：`uv python install`、`uv venv`、`uv pip install`，不用 conda。
  - 主训练、MapAnything（Python 3.12）、Grounded-SAM-2、NKSR、Waymo 预处理（TF 2.11，Python ≤3.10）各用一个独立 venv，模块之间只通过磁盘文件交接。
  - `dashrecon/io.py` 只依赖 numpy / PIL / json，保证能在所有 venv 中 import。
  - uv 不提供 CUDA toolkit，CUDA 扩展针对系统的 `/usr/local/cuda-12.1` 编译：torch 用 cu121 构建，安装时加 `--no-build-isolation`。
- 配置沿用 drivestudio 的配置系统；新增参数集中在 `configs/dashrecon/`。
- 禁止硬编码绝对路径；数据根目录通过环境变量或配置传入。
- 每个后端实现 `base.py` 中的抽象接口，通过配置名选择。
- `tests/` 中为 `io.py`、几何变换、Sim(3) 对齐、一致性过滤写小型单元测试（用合成数据，不依赖 Waymo）。
- 日志中打印每个阶段的耗时与显存峰值。

---

## 10. 禁止事项

1. **禁止 GT 泄漏**：非 oracle 实验中，重建流程不得读取 GT 位姿、LiDAR、GT 3D 框。特别注意 drivestudio 预处理产物中可能默认包含这些数据，训练入口必须显式断言不加载。评测代码可以读取 GT。
2. 禁止跳过现成模型基线而直接训练/微调。训练与微调仅按 §2 的证据条件作为补充；默认使用现成冻结权重。允许生成、几何校验与重建交替优化（DECISIONS S），生成内容必须明确标注。目标场景 FRONT 以外的相机仍只用于评测；适配数据不得包含评测场景的隐藏视角或 GT。
3. 目标场景的非 oracle 重建及测试时适配禁止使用 `FRONT` 以外的相机。其他训练场景上的领域适配按 §2 记录数据划分并隔离评测场景。
4. 禁止为了指标好看而更改评测协议；协议改动必须记录在 `docs/DECISIONS.md`，并对所有实验重跑。
5. 禁止在不确认的情况下假设第三方库 API。
6. **禁止在非 oracle 实验中使用 Waymo 标定**，包括内参、畸变系数、外参和相机离地高度（D3）。
7. 非 oracle 实验禁止读取上游预处理产出的 `dynamic_masks/`、`fine_dynamic_masks/`（两者都由 GT 框生成，C11），以及 `ego_pose/`、`extrinsics/`、`intrinsics/`、`lidar/`、`instances/`。

---

## 11. 已知风险与待确认问题

- 前向运动视差不足：估计位姿可能在长序列上出现尺度漂移，重点关注分块边界。
- 掩掉所有车辆会移除停放车辆，背景留下空洞；对比 `all` 与 `moving` 两种 GT 掩码可量化影响。
- Waymo 相机为卷帘快门，dashcam 同样存在该问题。**已确认 drivestudio 没有建模**（C13）。
- 跨相机评测中，侧前相机与前视相机的重叠有限，重叠掩码与颜色对齐方式会影响数值，需要固定下来并记录。
- NKSR 在观测稀疏区域可能产生鼓包或过度平滑。
- **镜头畸变**：不使用标定（D3），就是在原始畸变图像上按针孔模型重建。FRONT 的 k2≈-0.33，图像边缘的几何和光度会不一致，见 OPEN_QUESTIONS 12。
- ~~待确认：drivestudio 单相机配置的具体写法；关闭动态建模的方式~~：已确认，见 DECISIONS C1 和 C7。单相机配置是 `dataset=waymo/1cams`（`cameras: [0]`），FRONT=0。
- 待确认：MapAnything 在 A6000 48GB 上一次可处理的最大帧数与分辨率，见 OPEN_QUESTIONS 13。

---

## 12. 进度 Checklist

执行顺序为 0 → 3 → 4 → 5 → 7 的网格重建 → 6（E3、E4、E5 一起跑）；Phase 1、2 延后（D1、D8）。

- [x] 可执行性评估：结论见 `docs/DECISIONS.md`，待定问题见 `docs/OPEN_QUESTIONS.md`
- [x] Phase 0：环境与数据（2026-09-24：3 个 uv venv；5 个 validation 开发场景）
- [x] Phase 3：位姿与点图估计（2026-09-24：按临时标准验收；MapAnything 在 5 个场景跑通。内参、尺度、抖动问题见 OPEN_QUESTIONS 14–17）
- [x] Phase 4：动态与天空掩码（2026-09-24：按临时标准验收；Grounded-SAM-2 + SegFormer 在 5 个场景跑通，逐帧比例记在 meta.json；与 GT moving 的 IoU 延后。见 DECISIONS G）
- [x] Phase 5：点云融合与清理（2026-09-24：按临时标准验收；5 个场景都有逐步点数和截图/网页；"一致性过滤后几何指标更好"需要 LiDAR 评测，延后。见 DECISIONS H）
- [x] Phase 3 修正（2026-09-25，D13）：相机自标定 + SfM 位姿，轨迹抖动消失，焦距误差 4–14%；E5c 在 5 个场景完成，留出帧均值 +3.8 dB，跨相机 +0.9 dB（DECISIONS L）。尺度偏小（OPEN_QUESTIONS 15）、坡度（17）、焦距与畸变混淆（33）仍待处理
- [x] Phase 6：E3–E4（含横向偏移渲染）（2026-09-25：按临时标准验收；E3 / E4 / E5 在 5 个场景完成训练和 0 / 0.5 / 1 / 2 横移渲染，结果见 DECISIONS J）
- [x] Phase 7：NKSR 与 E5（2026-09-24：网格重建按临时标准验收，见 DECISIONS I；2026-09-25：E5 完成，与 E4 的对比记在 DECISIONS J。留出帧均值 E5 高 0.26 dB，主要来自 val094；几何评测延后）
- [x] Phase 8：生成式蒸馏与后处理（E6、E5+pp，D10–D12）（2026-09-25：按临时标准验收；E6 和 E5+pp 在 5 个场景完成，横移对比渲染和网页 E5 / E6 切换已产出。留出帧 E6 比 E5 低 0.19 dB，E5+pp 低 1.1 dB；横移视角的改善只有目测，跨相机评测延后。见 DECISIONS K）
- [ ] （延后）Phase 1：评测工具（2026-09-25：任务 2 的简化版"跨相机检查"已完成，D15，`scripts/eval_cross_camera.py`；其余延后）
- [x] Phase 8 换用 Fixer（D14）：E7 和 E5c+fx 在 5 个场景完成（2026-09-25，DECISIONS M）。E7 与 E5c 基本持平；E5c+fx 使留出帧 LPIPS 0.210 → 0.183；未观测区域仍然是空的，见 OPEN_QUESTIONS 35
- [ ] （延后）Phase 2：E0 oracle 上界；以及 E1、E2
- [ ] （延后）汇总报告 `results/SUMMARY.md`

- [ ] Phase 9：未观测区域的生成式补全，面向自由视角（D16–D18，E8）（2026-09-25：流程在 val039 打通；2026-09-26：val056 完成 7 轮（右移 + 15°、±30°、±60°、±90°），4 个侧相机的未观测区域 +0.5 dB，结果见 DECISIONS N。按用户要求先只在 val056 上迭代，满意后再扩展到 5 个场景）
- [ ] E9：在重建网格上复现 LSD-3D 的外观生成（DECISIONS O；只在 val056 上）（2026-09-26：不用真实帧时发散；锚定真实帧后局部更清晰，但颜色漂移、垃圾被放大，暂停）

- [x] Phase 9 修正任务 0：记录“现成模型优先、训练/微调仅补充已证实缺口”的用户授权（2026-09-27，DECISIONS S）。
- [ ] Phase 9 修正任务 1：跨轮生成缓存与短轨迹对照（E11；缓存/几何检查代码已实现，10 项相关测试通过，真实数据实验待验收）。
- [ ] Phase 9 修正任务 2：生成几何验收、来源/置信度与拒收机制。
- [ ] Phase 9 修正任务 3：直接生成 / 3D 拟合 / 未见视角分离诊断。
- [ ] Phase 9 修正任务 4：完整现成模型基线可运行性核查与实验。
- [ ] Phase 9 修正任务 5：固定协议评测与多场景扩展（短轨迹验收后）。

后续阶段（当前不做）：迁移到 Mapillary 的真实 dashcam 视频和多次通行融合（D19，计划见第 14 节）；闭环仿真集成。

---

## 13. 复现：Phase 0、Phase 3–8 与网页可视化

所有命令都在仓库根目录执行。

前提：
- `data/data` 链接到 Waymo perception v1.4.3（至少包含 `validation/`）；
- 已安装 uv；
- 有 CUDA 12.1 toolkit。

每个产物的 `meta.json` 或 `vis_params.json` 都记录了生成它的 `dashrecon_commit`；带 `-dirty` 后缀的结果无法仅凭 commit 复现。

### 13.1 环境（Phase 0）

```bash
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6 bash envs/setup_main.sh   # 主训练环境（Phase 6 用）
bash envs/setup_waymo.sh                                                          # Waymo 预处理
bash envs/setup_mapanything.sh                                                    # Phase 3
bash envs/setup_masks.sh                                                          # Phase 4
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6 bash envs/setup_nksr.sh  # Phase 7（NKSR 源码编译）
PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q                           # 单元测试
```

### 13.2 数据（Phase 0）

```bash
# 可选：重新统计 100 个 validation 段；选出的结果已写死在 dashrecon/scenes.py
CUDA_VISIBLE_DEVICES="" .venvs/waymo/bin/python scripts/scene_stats.py --raw_dir data/data/validation \
    --file_list data/waymo_val_list.txt --out data/dashrecon/scene_selection/validation_stats.json --workers 12
# 预处理 5 个开发场景（不处理 lidar）
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venvs/waymo/bin/python datasets/preprocess.py --data_root data/data/validation \
    --target_dir data/waymo/processed --dataset waymo --split validation --waymo_file_list data/waymo_val_list.txt \
    --scene_ids 56 39 41 87 94 --workers 5 --process_keys images calib pose dynamic_masks objects
```

### 13.3 位姿与点图（Phase 3）

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in val056 val039 val041 val087 val094; do
  # 主 run（DECISIONS F 节：facebook/map-anything）
  .venvs/mapanything/bin/python scripts/run_pose.py --scene_id $s --backend mapanything --model_id facebook/map-anything \
      --processed_root data/waymo/processed/validation --out_root data/dashrecon --scale model --max_views 300
  # 对比 run（Apache 权重），只用于诊断表
  .venvs/mapanything/bin/python scripts/run_pose.py --scene_id $s --backend mapanything --model_id facebook/map-anything-apache \
      --processed_root data/waymo/processed/validation --out_root data/dashrecon/_checkpoint_compare/map-anything-apache \
      --scale model --max_views 300
done
# 评测侧诊断（读 GT 位姿和标定，仅用于诊断）
.venvs/mapanything/bin/python scripts/diagnose_pose.py --processed_root data/waymo/processed/validation \
    --tag pose-mapanything_depth-mapanything --run cc-by-nc=data/dashrecon \
    --run apache=data/dashrecon/_checkpoint_compare/map-anything-apache --primary cc-by-nc \
    --out data/dashrecon/diagnostics/phase3_pose.json
```

### 13.4 动态、天空与路面掩码（Phase 4）

```bash
for s in val056 val039 val041 val087 val094; do
  .venvs/masks/bin/python scripts/run_masks.py --scene_id $s --processed_root data/waymo/processed/validation \
      --out_root data/dashrecon --detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large \
      --box_threshold 0.25 --text_threshold 0.25 --dilate_px 5 \
      --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_hw 1024 1536
done
```

### 13.5 点云融合与清理（Phase 5）

```bash
for s in val056 val039 val041 val087 val094; do
  .venvs/main/bin/python scripts/run_fusion.py --scene_id $s \
      --pose_dir data/dashrecon/$s/pose-mapanything_depth-mapanything --mask_dir data/dashrecon/$s/mask-gsam2_sky-segformer \
      --processed_root data/waymo/processed/validation --out_root data/dashrecon --test_stride 10 \
      --conf_percentile 30 --max_depth 60 --consistency_k 4 --consistency_rel 0.05 --consistency_min 2 \
      --voxel 0.05 --outlier_nb 20 --outlier_std 2.0 --road_cell 0.5 --road_sigma 2.0 --road_min_points 5 \
      --road_max_dz 0.3 --normal_knn 30 --rejected_sample 200000 --seed 0
done
```

做消融时加 `--skip consistency voxel outlier road` 中的任意几项，同时把 `--out_root` 换成别的目录，避免覆盖主结果。

### 13.6 表面网格（Phase 7 任务 1–3，D8）

```bash
export PATH=$PWD/.venvs/nksr/bin:/usr/local/cuda-12.1/bin:$PATH
for s in val056 val039 val041 val087 val094; do
  .venvs/nksr/bin/python scripts/run_mesh.py \
      --fusion_dir data/dashrecon/$s/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer \
      --detail_level 0.5 --mise_iter 1 --solver_tol 1e-4 --coverage_radius 0.15
done
```

### 13.7 3DGS 训练与渲染（Phase 6：E3、E4、E5）

```bash
export PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1
# E3 的初始点云：不做任何清理的反投影点云
for s in val056 val039 val041 val087 val094; do
  python scripts/run_fusion.py --scene_id $s --pose_dir data/dashrecon/$s/pose-mapanything_depth-mapanything \
      --mask_dir data/dashrecon/$s/mask-gsam2_sky-segformer --processed_root data/waymo/processed/validation \
      --out_root data/dashrecon/_e3_nocleanup --test_stride 10 --conf_percentile 30 --max_depth 60 --consistency_k 4 \
      --consistency_rel 0.05 --consistency_min 2 --voxel 0.05 --outlier_nb 20 --outlier_std 2.0 --road_cell 0.5 \
      --road_sigma 2.0 --road_min_points 5 --road_max_dz 0.3 --normal_knn 30 --rejected_sample 200000 --seed 0 \
      --skip consistency voxel outlier road
done
# 训练（每次训练前都会做 GT 隔离断言）→ results/<exp>/<scene>/；A6000 上可以同时跑两个
for e in E4 E5 E3; do for s in val056 val041 val087 val039 val094; do
  python scripts/train_gs.py --exp $e --scene_id $s --output_root results
done; done
# 沿原轨迹以及横向偏移 0.5 / 1 / 2（估计米）渲染 → results/<exp>/<scene>/renders/
for e in E3 E4 E5; do for s in val056 val039 val041 val087 val094; do
  python scripts/render_lateral.py --log_dir results/$e/$s --offsets 0 0.5 1 2 --still_frames 50 100 150 --fps 10
done; done
# 网页用的对比拼图和指标汇总 → results/_vis/
python scripts/vis_training.py --results_root results --exps E3 E4 E5 --frames 50 100 150 --offsets 0 0.5 1 2 \
    --cell_width 480 --processed_root data/waymo/processed/validation --out_dir results/_vis
```

### 13.8 生成式蒸馏与后处理（Phase 8：E6、E5+pp）

```bash
export PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1
bash envs/download_gen_weights.sh    # 一次性：SDXL、ControlNet depth、fp16 VAE → HF 缓存（9.6 GB）
export HF_HUB_OFFLINE=1
for s in val056 val039 val041 val087 val094; do
  # E6：从 results/E5/<scene> 续训 6000 步 → results/E6/<scene>/（含 gen/ 下每轮的目标样例）
  python scripts/train_ggds.py --scene_id $s --output_root results
  python scripts/render_lateral.py --log_dir results/E6/$s --offsets 0 0.5 1 2 --still_frames 50 100 150 --fps 10
  # E5+pp：E5 渲染逐帧细化 → results/E5pp/<scene>/（测试帧指标、静帧、横移 1 的视频）
  python scripts/postprocess_frames.py --scene_id $s --output_root results --offsets 0 0.5 1 2 \
      --still_frames 50 100 150 --video_offsets 1
done
# 网页用的对比拼图（E3–E6 共 5 列，所以单元格缩到 400 宽）和 E6 目标样例 → results/_vis/
python scripts/vis_training.py --results_root results --exps E3 E4 E5 E5pp E6 --frames 50 100 150 \
    --offsets 0 0.5 1 2 --cell_width 400 --gen_exp E6 --gen_views 2 \
    --processed_root data/waymo/processed/validation --out_dir results/_vis
```

E6 峰值显存约 19 GB，E5+pp 约 13 GB，都包含 drivestudio 训练器。

### 13.9 可视化数据与网页组装

```bash
for s in val056 val039 val041 val087 val094; do
  # 掩码叠加缩略图（每 10 帧一张）和逐帧掩掉像素比例
  .venvs/masks/bin/python scripts/vis_masks.py --scene_id $s --mask_dir data/dashrecon/$s/mask-gsam2_sky-segformer \
      --processed_root data/waymo/processed/validation --thumb_stride 10 --thumb_width 640
  # 点云、俯视图和网页数据；--mask_dir 给每个点加上类别（other / road / dynamic / sky）
  .venvs/mapanything/bin/python scripts/vis_pose.py --scene_id $s --scene_dir data/dashrecon/$s/pose-mapanything_depth-mapanything \
      --processed_root data/waymo/processed/validation --frame_stride 2 --conf_percentile 30 --max_depth 60 \
      --max_points 120000 --seed 0 --mask_dir data/dashrecon/$s/mask-gsam2_sky-segformer
  # 融合点云抽样（12 万点，含法向和路面标签）以及一致性检查剔除的点（3 万点）
  .venvs/main/bin/python scripts/vis_fusion.py --scene_id $s \
      --fusion_dir data/dashrecon/$s/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer \
      --max_points 120000 --rejected_points 30000 --seed 0
  # 网格简化到 10 万个面用于显示（完整网格保留在磁盘上）
  .venvs/main/bin/python scripts/vis_mesh.py --scene_id $s \
      --fusion_dir data/dashrecon/$s/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer --target_faces 100000
  # 网页 3DGS 模式用的 E5、E6 Gaussian（各 8 万个，去掉最大轴超过第 99.5 百分位的）
  for e in E5 E6; do
    .venvs/main/bin/python scripts/export_splats.py --log_dir results/$e/$s --scene_id $s \
        --max_splats 80000 --min_opacity 0.05 --max_scale_pct 99.5
  done
done
.venvs/mapanything/bin/python scripts/build_viewer.py --scene_root data/dashrecon --tag pose-mapanything_depth-mapanything \
    --diagnostics data/dashrecon/diagnostics/phase3_pose.json --page_title "DashRecon Pose Review" \
    --mask_tag mask-gsam2_sky-segformer --fusion --mesh \
    --training_dir results/_vis --results_root results --splat_exps E5 E6 --out_dir data/dashrecon/viewer/review
```

- 不加 `--training_dir`：网页里没有 Phase 6 的内容。
- 不加 `--mesh`：网页里没有网格。
- 再去掉 `--fusion`：没有 Phase 5 的内容。
- 再去掉 `--mask_dir` 和 `--mask_tag`：只剩 Phase 3。
- 不加 `--splat_exps`：网页里没有 3DGS 模式；只给 `E5`：没有 E5 / E6 切换。
- 显示用的抽样规模（原始 12 万点、融合 12 万点、剔除点 3 万点、网格 10 万个面、每个场景 E5 和 E6 各 8 万个 Gaussian）是为了让整个网页小于 Claude Artifact 单个版本 64 MB 的上限（实测约 61 MB）。

### 13.10 查看网页

- **本地**：直接用浏览器打开 `data/dashrecon/viewer/review/index.html`。也可以起一个静态服务器：
  ```bash
  cd data/dashrecon/viewer/review && python3 -m http.server 8000
  ```
  然后访问 http://localhost:8000。远程机器需先转发端口，例如 `ssh -L 8000:localhost:8000 <host>`。页面需要联网，才能从 CDN 加载 three.js r128 和字体。
- **在线（Claude Artifact，私有）**：https://claude.ai/artifact/VzZJhgdJapeiyjFPWHuPVM 。更新方式：在 Claude Code 中用 Artifact 工具发布 `data/dashrecon/viewer/review/index.html`，把 `config.js` 和 `scenes/*.js`（包括 `*.masks.js`、`*.fusion.js`、`*.mesh.js` 和 `training/*.jpg`）作为 supporting files，并传入上面的 URL，这样链接保持不变。

- **完整高斯场景（交互式，需要端口转发）**：`scripts/view_gs.py` 加载训练好的 run，渲染全部 Background 高斯（按视角计算颜色）加天空，比网页的 8 万个底色高斯完整。
  ```bash
  PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
      .venvs/main/bin/python scripts/view_gs.py --log_dirs results/E5/val056 results/E6/val056 --port 8080
  ```
  本地执行 `ssh -L 8080:localhost:8080 <host>` 后打开 http://localhost:8080。左侧面板：`run` 切换实验；`frame` 和 `lateral offset` 把相机放到某一帧的 FRONT 相机（可横移）；鼠标拖动旋转、右键平移、滚轮前后移动。

网页的工作方式：
- 模板 `dashrecon/viewer/index.html` 用 three.js 渲染点云、相机轨迹和视锥。
- 场景列表、诊断数字、可视化参数和掩码统计都来自 `build_viewer.py` 生成的 `config.js`，页面里没有硬编码的数字。
- 每个场景的数据按需加载：
  - `scenes/<id>.js`：点云。30 万个点，坐标量化为 uint16，颜色和类别为 uint8，都做了 base64 编码。
  - `scenes/<id>.masks.js`：掩码叠加缩略图和逐帧比例。
- 3D 视图的交互：
  - "Raw depth / Fused / Mesh / 3DGS"：在 Phase 3 的原始反投影点云、Phase 5 的融合点云、Phase 7 的 NKSR 网格和训练好的 Gaussian 之间切换。
  - 3DGS 模式下的 "E5 / E6 gen"：切换实验。视角保持不动，方便对比；E6 标 gen，因为含生成内容。每个实验第一次用到时才解码，同一场景内保留。
  - 底部的帧进度条：拖到某一帧，视角跳到该帧（3DGS 模式下是从该帧 FRONT 相机看出去，视场角取估计内参；其他模式是该帧后上方的跟车视角），该帧的视锥用强调色标出。"Play" 按 10 Hz 沿轨迹播放。为此 `vis_pose.py` 的网页数据存下每一帧的位姿和帧号（点云抽样仍按 `--frame_stride`）。
  - "Photo / Classes / Normals / Shaded"：分别按图像颜色、类别、法向（x→R、y→G、z→B）着色；"Shaded" 只用于网格，显示灰色光照下的形状。
  - "Hide dynamic"（仅原始点云）：隐藏动态物体的点。
  - "Rejected"（仅融合点云）：用红色显示被一致性检查剔除的点。
- 类别颜色取自 dataviz 参考调色板的 1–3 号色。用 `validate_palette.js` 在两种主题下做了全配对验证，均通过。

### 13.11 产物位置（均不入 git）

| 产物 | 路径 |
|---|---|
| 预处理后的场景 | `data/waymo/processed/validation/<idx>/` |
| Phase 3 数据（第 5 节约定） | `data/dashrecon/<scene_id>/pose-mapanything_depth-mapanything/` |
| Phase 4 掩码（第 5 节约定，独立的 backend tag） | `data/dashrecon/<scene_id>/mask-gsam2_sky-segformer/`：`mask_{dynamic,sky,road}/`、`meta.json`（含逐帧比例） |
| 点云可视化 | Phase 3 目录下的 `vis/`：`points_vis.ply`（可用 CloudCompare 或 MeshLab 打开）、`bev.png`、`viewer_data.js`、`vis_params.json` |
| 掩码可视化 | Phase 4 目录下的 `vis/masks_view.js` |
| Phase 5 融合点云（第 5 节约定） | `data/dashrecon/<scene_id>/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer/`：`points_fused.ply`（xyz、法向、rgb、`label` 0 = 其他 / 1 = 路面，可用 CloudCompare 打开）、`frames.txt`（用到的训练帧）、`meta.json`（逐步点数、路面平滑统计）、`diagnostics/consistency_rejected.ply`、`vis/fusion_view.js` |
| Phase 6 训练结果（第 8 节约定） | `results/<exp>/<scene_id>/`：`config.yaml`、`checkpoint_final.pth`、`metrics.json`、`meta.json`、drivestudio 的 `videos/` 和 `metrics/`、`renders/lateral_<offset>.mp4` 与 `renders/frames/`；训练日志在 `results/_logs/` |
| Phase 8 结果 | `results/E6/<scene_id>/`：同 Phase 6，另有 `ggds_params.json` 和 `gen/round<r>_view<k>.jpg`（渲染 / 网格视差 / SDXL 目标）；`results/E5pp/<scene_id>/`：`metrics.json`（只有 test）、`meta.json`、`renders/frames/`、`renders/lateral_1.mp4` |
| Phase 6 / 8 网页数据 | `results/_vis/`：`<scene>_<t>.jpg` 对比拼图、`<scene>_gen_r<r>_v<k>.jpg` E6 目标样例、`summary.json` |
| Phase 7 网格（第 5 节约定） | 同上的融合目录：`mesh_nksr.ply`（顶点带颜色，可用 MeshLab 打开）、`mesh_nksr.json`（参数、覆盖率、边界边、耗时）、`vis/mesh_view.js` |
| Apache 对比 run | `data/dashrecon/_checkpoint_compare/map-anything-apache/` |
| 诊断 JSON | `data/dashrecon/diagnostics/phase3_pose.json` |
| 组装好的网页 | `data/dashrecon/viewer/review/` |
| 相机自标定（D13） | `data/dashrecon/<scene_id>/calib-glomap/`：`camera.json`（共享 K、k1 k2）、`frames.txt`、`poses_c2w.npy`（SfM 坐标系和单位）、`sparse_obs.npz`、`meta.json`、`colmap/sparse/`；去畸变图像 `data/dashrecon/_undistorted/calib-glomap/<idx>/images/<t>_0.jpg` |
| 自标定流水线的 Phase 3–7 产物 | `data/dashrecon/<scene_id>/pose-glomap_depth-mapanything/`、`mask-gsam2_sky-segformer_img-glomap/`、`pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap/` |
| 跨相机检查（D15） | `results/<exp>/<scene_id>/cross_camera/`：`metrics.json`、`<t>_<cam>.jpg`（GT \| 颜色对齐后的渲染 \| 重叠区） |
| E7 / E5c+fx（D14） | `results/E7/<scene_id>/`（另有 `fixer_params.json`、`gen/round<r>_view<k>.jpg`：渲染 \| Fixer 目标）；`results/E5cfx/<scene_id>/` |

### 13.12 自标定流水线（D13）、跨相机检查（D15）与 Fixer（D14）

```bash
CUDA_HOME=/usr/local/cuda-12.1 bash envs/setup_sfm.sh     # pycolmap 4.2（CPU）
CUDA_HOME=/usr/local/cuda-12.1 bash envs/setup_fixer.sh   # NVIDIA Fixer（不用 NGC 容器），权重 nvidia/Fixer
U=data/dashrecon/_undistorted/calib-glomap
for s in val056 val039 val041 val087 val094; do
  # 自标定：GLOMAP，共享 RADIAL 相机，动态和天空像素不提特征 -> calib-glomap/ 与去畸变图像
  .venvs/sfm/bin/python scripts/run_calib.py --scene_id $s --processed_root data/waymo/processed/validation \
      --mask_dir data/dashrecon/$s/mask-gsam2_sky-segformer --out_root data/dashrecon --undistorted_root $U \
      --max_features 8192 --overlap 20 --seed 0
  # MapAnything：输入去畸变图像、共享内参和 SfM 位姿，补深度；SfM 位姿按稀疏点深度比缩放 -> pose-glomap_depth-mapanything/
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venvs/mapanything/bin/python scripts/run_pose.py --scene_id $s \
      --backend mapanything --model_id facebook/map-anything --processed_root $U \
      --camera_dir data/dashrecon/$s/calib-glomap --out_root data/dashrecon --scale model --max_views 300
  # 去畸变图像上的掩码（参数同 13.4）
  .venvs/masks/bin/python scripts/run_masks.py --scene_id $s --processed_root $U --out_root data/dashrecon \
      --detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large --box_threshold 0.25 \
      --text_threshold 0.25 --dilate_px 5 --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
      --seg_input_hw 1024 1536 --image_tag img-glomap
  # 融合与网格：13.5、13.6 的命令，换成 --pose_dir .../pose-glomap_depth-mapanything、
  # --mask_dir .../mask-gsam2_sky-segformer_img-glomap、--processed_root $U
done
export PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1
for s in val056 val039 val041 val087 val094; do
  python scripts/train_gs.py --exp E5c --scene_id $s --output_root results
  python scripts/train_fixer.py --scene_id $s --output_root results                     # E7
  python scripts/postprocess_frames.py --scene_id $s --output_root results --generator fixer --init_exp E5c \
      --offsets 0 0.5 1 2 --still_frames 50 100 150 --video_offsets 1                  # E5c+fx
  for e in E3 E4 E5 E6 E5c E7; do
    python scripts/render_lateral.py --log_dir results/$e/$s --offsets 0 0.5 1 2 --still_frames 50 100 150 --fps 10
    # 跨相机检查（评测代码，读 GT）：FRONT_LEFT / FRONT_RIGHT，每 5 帧一次
    python scripts/eval_cross_camera.py --log_dir results/$e/$s --gt_root data/waymo/processed/validation \
        --frame_stride 5 --alpha 0.5 --example_frames 50 100 150
  done
done
.venvs/mapanything/bin/python scripts/diagnose_pose.py --processed_root data/waymo/processed/validation \
    --tag pose-glomap_depth-mapanything --run glomap=data/dashrecon --primary glomap \
    --out data/dashrecon/diagnostics/phase3_pose_glomap.json
# 第二个网页（自标定流水线；两套数据放不进一个 64 MB 的 Artifact）：13.9 的可视化命令换成新目录，
# vis_pose / vis_masks 的 --processed_root 用 $U
.venvs/mapanything/bin/python scripts/build_viewer.py --scene_root data/dashrecon --tag pose-glomap_depth-mapanything \
    --diagnostics data/dashrecon/diagnostics/phase3_pose_glomap.json --page_title "DashRecon Calibrated Review" \
    --mask_tag mask-gsam2_sky-segformer_img-glomap --fusion --mesh --training_dir results/_vis_calib \
    --results_root results --splat_exps E5c E7 --out_dir data/dashrecon/viewer/calib
# 其中 results/_vis_calib 与 E5c / E7 的高斯：
#   export_splats.py --log_dir results/{E5c,E7}/$s --scene_id $s --max_splats 80000 --min_opacity 0.05 --max_scale_pct 99.5
#   vis_training.py --results_root results --exps E5c E5cfx E7 --frames 50 100 150 --offsets 0 0.5 1 2 --cell_width 480 \
#       --gen_exp E7 --gen_views 2 --processed_root $U --out_dir results/_vis_calib
```

在线（Claude Artifact，私有）：https://claude.ai/artifact/SzEh7vBbofCdSWA2nQr3sC （"DashRecon Calibrated Review"；原流水线仍是 13.10 的链接）。发布方式同 13.10。

### 13.13 Phase 9（E8）与 E9

先只在 val056 上跑（用户要求，2026-09-25）。

```bash
bash envs/setup_wan.sh   # Wan2.1-VACE-1.3B（补全）与 Wan2.1-T2V-1.3B transformer（重绘），HF_HUB_OFFLINE=1 运行
# E8：每个相机运动一轮（渲染 + 空洞 -> Wan 补全 -> 深度 -> 生成 Gaussian 并蒸馏），之前各轮作为 3D 记忆
bash scripts/run_phase9.sh val056 results/E5c/val056 results/E8/val056 0 1 vace right=1.5,yaw=15
# fill backend gen3c: GEN3C (envs/setup_gen3c.sh) instead of Wan VACE, rendered at 704 x 1280 with the move ramped in
# 已完成前 N 轮时从第 N 轮接着跑：first_round 设为 N，前面各轮的 move 原样列出
# 跨相机检查（评测代码，读 GT）：4 个侧相机 + E5c 定义的未观测区域 -> <run>/cross_camera_p9/
export PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1
for e in E5c E8 E9; do
  python scripts/eval_cross_camera.py --log_dir results/$e/val056 --gt_root data/waymo/processed/validation --frame_stride 5 \
      --alpha 0.5 --example_frames 50 100 150 --cams 1 2 3 4 --region_ref results/E5c/val056 --out_subdir cross_camera_p9
done
# Wan 重绘测试（strength 对比，wan venv）
.venvs/wan/bin/python scripts/repaint_views.py --views_dir results/E8/val056/views/r0_right1.5_yaw15 --source filled \
    --start 40 --strengths 0.3 0.5 0.7 --still_frames 50 70 100
# E9：LSD-3D 式 GGDS
python scripts/train_lsd.py --scene_id val056 --output_root results
# 自由视角对比视频；--deferred_t 给最后一个 run 加 SDXL 延迟渲染列
python scripts/vis_freeview.py --log_dirs results/E5c/val056 results/E8/val056 --labels E5c E8 \
    --moves right=1.5,yaw=15 right=-1.5,yaw=-15 up=1.5,pitch=-10 yaw=45 --still_frames 50 100 150 --cell_width 480 \
    --fps 10 --out_dir results/_vis_p9
# 交互式查看（right / up / yaw / pitch 滑块）
python scripts/view_gs.py --log_dirs results/E5c/val056 results/E8/val056 results/E9/val056 --port 8080
```

---

## 14. Mapillary 部署计划（D19，记录，暂不做）

先用 Waymo 的干净数据把整条流程打通，再迁移到 Mapillary。迁移时要处理的事项：

1. **多次通行融合**：同一路段往往有多段视频（不同时间、不同车道）。静态背景一致、动态物体不同，可以互相补齐遮挡区域，改善换道视角；这是唯一能让"补出来的内容是真实的"的办法，可与 Phase 9 的生成补全互补（参考 MTGS 一类多趟重建工作）。需要先解决跨视频的位姿对齐和光照、季节差异。
2. **相机**：同一设备的多段视频共用一套内参和畸变，可以联合自标定，缓解直路上焦距与畸变的混淆（OPEN_QUESTIONS 33）；dashcam 常见广角、鱼眼和卷帘快门，畸变模型可能要从 RADIAL 换成 OPENCV / 鱼眼，卷帘快门可能需要建模（3DGUT）。
3. **尺度**：没有标定和 LiDAR，度量尺度需要单独解决（OPEN_QUESTIONS 15），例如相机安装高度先验、车道宽度先验或更准的单目度量深度。
4. **动态与停放车辆**：Mapillary 视频里的停放车辆更多，按运动分割（保留停放车辆）对静态场景质量影响很大（OPEN_QUESTIONS 18）。
5. **数据与许可**：Mapillary 图像为 CC-BY-SA；当前流水线中 MapAnything（CC-BY-NC）、NKSR、SegFormer 限非商业使用（OPEN_QUESTIONS 26）。

---

## 变更记录

- 2026-09-24：按可执行性评估和用户决定修订了以下各节，详情与代码引用见 `docs/DECISIONS.md`。
  - §2：硬件改为 A6000；新增"不用标定"。
  - §3：写入已选后端和 uv。
  - §4：数据版本 v1.4.3，写明 LiDAR 情况。
  - §6：执行顺序；Phase 0 改用 validation 场景和 uv；GT 相关任务标为延后。
  - §7：当前状态。
  - §9：uv 多环境约定。
  - §10：第 6、7 条。
  - §11：已确认项和畸变风险。
  - §12：checklist 顺序。
- 2026-09-24：Phase 0、Phase 3 完成（§12）；§4 补上新增的文件；新增 §13（Phase 0 / Phase 3 / 网页可视化的复现命令、查看方式和产物位置）。网页源码在 `dashrecon/viewer/`，数据由 `scripts/vis_pose.py`、`scripts/diagnose_pose.py`、`scripts/build_viewer.py` 生成。
- 2026-09-24：Phase 4 完成（§12）；§4 列出 masks 后端；§13 重排为 Phase 3 → Phase 4 → 可视化 → 查看网页 → 产物位置，网页输出目录改为 `data/dashrecon/viewer/review`。
- 2026-09-24：Phase 5 完成（§12）；§4 列出 fusion/backproject.py 和新脚本；§13 新增 13.5（Phase 5 命令），可视化命令加入 `vis_fusion.py` 和 `build_viewer.py --fusion`。
- 2026-09-24：D8、D9（先网格后 3DGS；E5 在网格上初始化并正则）；Phase 7 网格重建完成；§13 新增 13.6（网格命令），显示抽样规模下调到 64 MB 以内，`build_viewer.py --mesh`；§6、§7 更新 E5 定义与执行顺序。
- 2026-09-24：Phase 8 代码完成（`dashrecon/gen/`、`scripts/train_ggds.py`、`scripts/postprocess_frames.py`，DECISIONS K）；§6 Phase 8 按实现改写（t 上界退火、横移 0.5–2.5、天空不参与生成 loss）；§13 新增 13.8（Phase 8 命令），原 13.8–13.10 顺延；13.9 的显示抽样规模改为实际值，并补上 `export_splats.py` 和 `--splat_exp`。
- 2026-09-25：Phase 6、Phase 7（E5）完成（§12），结果记在 DECISIONS J。
- 2026-09-25：网页 3DGS 模式加入 E5 / E6 切换（`build_viewer.py --splat_exps`），每个实验每个场景 8 万个 Gaussian（§13.9、§13.10）。
- 2026-09-25：网页加入帧进度条（§13.10）；`vis_pose.py` 的网页数据改为存每一帧的位姿和帧号。
- 2026-09-25：Phase 8 完成（§12），结果记在 DECISIONS K。
- 2026-09-25：D13–D15：相机自标定流水线（E4c、E5c）、跨相机检查、Fixer（E7、E5c+fx）；§7 新增这些实验；§12 更新；§13 新增 13.12 与产物位置。
- 2026-09-25：§0 新增第 8 条（用户要求）：每完成一个任务项就 commit；长任务启动前工作区必须干净。
- 2026-09-25：§13.10 新增交互式高斯查看器（`scripts/view_gs.py`）。
- 2026-09-25：Phase 8 的 Fixer 实验（E7、E5c+fx）完成；自标定页面加入 3DGS、对比拼图和跨相机表（§13.12）。
- 2026-09-25：D16–D19（用户决定）：§1 目标扩展到自由视角和未观测区域补全，非目标删去"不重建未被观测的区域"；§6 新增 Phase 9；§7 新增 E8；§10 第 2 条放宽到 Phase 9；§12 新增 Phase 9；新增 §14 Mapillary 部署计划（暂不做）。
- 2026-09-25：Phase 9 流程打通（val039），val056 第一轮完成（DECISIONS N）；用户要求先只在 val056 上迭代。新增 E9（用户决定：复现 LSD-3D 的外观生成，以保真换真实感，基础 SDXL 不微调，DECISIONS O）；§7 新增 E9；§12 更新；新增 §13.13。

- 2026-09-27：用户取消训练/微调绝对禁令，规定现成模型优先、仅用训练补足经验证的缺口；更新 §1/§2/§10，新增 Phase 9 一致性修正路线与 checklist（DECISIONS S）。
