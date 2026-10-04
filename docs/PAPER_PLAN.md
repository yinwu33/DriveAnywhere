# 论文计划：基线与评测协议

用户 2026-10-05 要求："收集 baseline 方法，作为一篇 paper 需要对比 baseline，设定 evaluation 方法"。本文档是方案，不是结果；真正启动的对比实验仍在 [EXPERIMENTS.md](EXPERIMENTS.md) 登记。文献条目见 [papers/paper.md](papers/paper.md)，本次新增的在"候选（论文基线与评测）"一节。

## 1. 论文的设定与要支撑的主张

**设定**：输入是一段单目前视行车记录仪视频（Waymo `FRONT` 模拟），不用 GT 位姿、标定、LiDAR、3D 框；输出一个静态 3D 街景，在评测范围（横移 ±3 m、转向 ±45°、升高 0–2 m，DECISIONS Z）内可渲染，用于 ego 仿真。未观测区域允许生成，但必须标注。

**候选主张**（每条都要有对应的表或图）：

| 主张 | 依据（现有） | 论文里需要的对比 |
|---|---|---|
| C1 单目非 oracle 输入下，评测范围内的质量好于现有方法 | E43：D-Z1 全部 KID 0.042 | 第 3 节的全部基线，统一输入 |
| C2 停放车辆被当成动态掩掉是单目驾驶重建的一个主要误差源；多视角深度一致性能分开停放和运动物体 | E41：FRONT +2.5 dB，D-Z1 每组都好；val041 运动车覆盖率不变 | 掩全部车辆 / 不掩 / GT 运动框 / 我们的，四种掩码对比 |
| C3 路面固定在表面上，横移时车道线不碎 | E42 系列：侧前相机路面 SSIM +0.008…+0.015 | NTL-IoU（车道线投影一致性）+ 路面 SSIM，对比自由高斯 / AutoSplat 式约束 |
| C4 车辆用以真实截图为条件的物体级生成补全，比场景级视频生成更真实 | E43：KID 每组下降；但车辆像素保真度下降 | 场景级生成基线（Difix3D+、GEN3C、ViewCrafter、Lyra）对比；保真度与真实感分开报 |
| C5（如果 E45 成功）表面约束消除侧视漂浮物 | 待做 | 漂浮物率（第 4 节 M7） |

主张必须和局限一起写：生成内容可信但不是真实（E43 车辆像素误差 0.171 → 0.178），45° 以外和远处仍靠通用生成，风格偏插画（用户 10-04）。

## 2. 数据与划分

| 用途 | 数据 | 为什么 | 状态 |
|---|---|---|---|
| 开发集 | Waymo val039 / 041 / 056 / 087 / 094（本机已有） | 全部调参都在这里做过 | 已有 |
| **测试集（主表）** | Waymo validation 再选 15–20 段，开发期间不看 | 多相机给侧视真值，LiDAR 给几何，地图给车道线，3D 框给 NTA-IoU 和评测用的动态掩码 | **需下载，要用户同意**（第 6 节） |
| 真实横移真值 | [EUVS](https://ai4ce.github.io/EUVS-Benchmark/)（多趟 / 多车 / 多相机，CC BY-SA 4.0，HF 下载）；[Para-Lane](https://nizqleo.github.io/paralane-dataset/)（真实平行车道扫描，填表申请） | Waymo 只有侧前相机（约 ±45°、横移很小），没有"相邻车道"真值 | 待用户决定选哪个 |
| 大视角变化（可选） | [MV2](https://arxiv.org/abs/2608.12442)（汽车 + 滑板车 + 无人机同步轨迹，ECCV 2026） | 远超评测范围的压力测试 | 可选 |
| 可控横移（可选） | [XLD](https://3d-aigc.github.io/XLD/)（CARLA 仿真，1–4 m 横移真值） | 无真实感但偏移量精确可控 | 可选 |
| 部署泛化（定性 + 无参考指标） | 真实行车记录仪片段（Mapillary 或用户自己的） | 证明不依赖 Waymo 相机 | 以后 |

**纪律**：测试集场景名在跑第一个方法之前固定并提交；评测代码在跑基线前冻结（打 tag），之后改评测代码要重跑所有方法。

## 3. 基线

所有基线统一用**同一输入**：只用 FRONT 图像；位姿和内参要么用该方法自己的估计，要么用我们的 GLOMAP 结果（两种都报时注明）；需要 LiDAR 的方法用估计深度或融合点云代替，标注"LiDAR→估计深度"。需要 3D 框或 HD 地图做条件的方法不能公平运行，只作为"oracle 条件参考"放在表的下半部分。

### A. 逐场景优化，无生成（必须）

| 方法 | 输入要求 | 适配 | 代码 | 优先级 |
|---|---|---|---|---|
| 3DGS（gsplat） | 位姿 | 我们的 GLOMAP 位姿 + 融合点云初始化 | 有 | 必须 |
| OmniRe / StreetGS（drivestudio，只训静态） | 位姿、LiDAR、框 | = 我们的 E5c（FRONT、估计位姿、估计深度、掩全部车辆）；这是最直接的"现有驾驶方法在单目下"的数字 | 有（本仓库） | 必须 |
| PVG / S³Gaussian | 位姿、LiDAR（无框） | LiDAR→估计深度 | 有（drivestudio 含 PVG） | 应该 |
| 2DGS 或 PGSR | 位姿 | 表面型高斯，对漂浮物的对照 | 有 | 应该 |
| InstantSplat | 无位姿（MASt3R） | 视频抽帧；长序列可能要分段 | 有 | 应该 |
| [LongSplat](https://arxiv.org/abs/2508.14041) | 无位姿长视频 | 直接跑 | 有 | 应该 |
| CF-3DGS | 无位姿 | LongSplat 报告长视频会 OOM，作为"跑不了"记录 | 有 | 可选 |
| 单目 GS-SLAM（[GeoGS-SLAM](https://arxiv.org/abs/2607.11184) 或 VINGS-Mono / MonoGS） | 无标定单目 | GeoGS-SLAM 在 Waymo 无标定设定下报过结果 | 待查 | 可选 |
| 摄影测量管线（用户 10-04 的问题） | 图像 | 没有 Metashape 许可，用 COLMAP MVS + OpenMVS 纹理网格；另加我们的 E39（融合点云直接当面元） | 有 | 应该 |

### B. 前馈（应该有一个）

| 方法 | 输入要求 | 适配 | 代码 | 优先级 |
|---|---|---|---|---|
| [AnySplat](https://dl.acm.org/doi/10.1145/3763326)（ACM TOG） | 无位姿、无标定多视图 | 零样本，抽关键帧 | 有 | 应该 |
| [DrivingForward](https://arxiv.org/abs/2409.12753) / STORM / [WorldSplat](https://arxiv.org/abs/2509.23402) | 多相机环视 | 只给 FRONT 是否能跑待查；跑不了就在相关工作里讨论 | 部分有 | 可选 |

### C. 生成先验外推（必须有 2–3 个）

| 方法 | 输入要求 | 适配 | 代码 | 优先级 |
|---|---|---|---|---|
| Difix3D+ | 一个 3DGS | 在基线 A 的 3DGS 上跑官方渐进蒸馏 + 后处理（我们的 E7 / E5cfx 已有） | 有（已在用） | 必须 |
| GEN3C 蒸馏 | 3D 缓存 | 我们的 E20（只有 GEN3C，没有后续步骤）就是这个基线 | 有（已在用） | 必须 |
| ViewCrafter | 点云 | 用融合点云渲染作条件 | 有 | 应该 |
| Lyra | 视频扩散 → 3DGS | 单目视频输入 | 有（NVIDIA） | 应该 |
| [FaithFusion](https://arxiv.org/abs/2511.21113)（CVPR 2026） | 3DGS + 扩散 | 输入要求待查（是否需要 LiDAR） | 有 | 应该 |
| StreetCrafter | LiDAR 条件 | LiDAR→融合点云渲染，标注 | 有 | 可选 |
| ReconDreamer / DriveDreamer4D / [GA-Drive](https://arxiv.org/abs/2602.20673) | 3D 框、HD 地图、LiDAR | 条件依赖 GT：只作 oracle 条件参考，或引用论文数字（同样的 Waymo 场景和 4 m 横移协议） | 前两者有 | 参考 |
| FreeVS、PRG-Fusion、DriveExplorer | — | 代码待查 | 待查 | 参考 |

### D. 上界（参考行）

- Oracle：OmniRe / StreetGS，GT 位姿 + LiDAR + 3D 框 + 全部相机（我们的 U3）；以及只用 FRONT + GT 位姿 + LiDAR（U1 / E0）。
- 说明 oracle 上界在侧视同样有空洞（U3 在转 45° / 90° 时由侧相机直接看到，所以完整）。

### E. 我们的消融

E5f（起点）→ +E41 运动掩码 → +路面面元（E42d）→ +车辆资产（E43）→ +场景级生成（E44 / E35 链）→ 视角屏蔽、补天空各自去掉一项。每一步同一张表。

## 4. 评测指标

现有工具在括号里；"新"是要写的。

| 编号 | 维度 | 指标 | 区域 / 视角 | 工具 |
|---|---|---|---|---|
| M1 | 插值保真 | PSNR / SSIM / LPIPS，含天空与不含天空 | FRONT 每 10 帧留出 1 帧（OmniRe 协议） | `eval_front_heldout.py` |
| M2 | 外推保真（真实真值） | 逐通道仿射颜色对齐后 PSNR / SSIM / LPIPS；**见过 / 没见过分开**；逐类别（路面、车、植被、建筑、天空） | Waymo FRONT_LEFT / FRONT_RIGHT（约 ±45°）；EUVS / Para-Lane 相邻车道 | `eval_cross_camera.py`、`diagnose_error_classes.py`；**新**：没见过的区域改用 LiDAR 定义（侧相机 LiDAR 点在任何训练 FRONT 帧里都不可见），与具体方法无关；**新**：EUVS / Para-Lane 加载器 |
| M3 | 评测范围内的真实感 | KID / FID（对同场景真实侧相机图像池）；横移轨迹视频的 FVD | 横移 ±1 / ±2 / ±3 m、转向 ±20° / ±45°、升高 1 / 2 m，分组报告 | `eval_view_range.py`（D-Z1）；**新**：FVD |
| M4 | 驾驶语义一致性 | NTA-IoU（静止车辆的 GT 框投影 vs 检测框）、NTL-IoU（HD 地图车道线投影 vs 车道线检测）；语义分割的路面 / 车辆 mIoU（对 LiDAR + 地图投影） | 同 M3 的移动视角，以及 ReconDreamer 协议（每帧 0.1 m、40 帧共 4 m 横移）以便和已发表数字对照 | **新**（DriveDreamer4D 仓库 2025-01 放出了 NTA-IoU / NTL-IoU 代码） |
| M5 | 生成内容的多视角一致性 | MEt3R；OneSceneEval 的 COLMAP 版本（主）；时间闪烁（相邻帧光流 warp 误差） | 移动视角视频 | **新** |
| M6 | 几何 | 渲染深度 vs LiDAR：AbsRel、δ<1.25（FRONT 留出帧和侧相机）；表面 vs 聚合 LiDAR 的 Chamfer、precision / recall（0.2 / 0.5 m） | Sim(3) 对齐后 | **新**（AGENTS Phase 1 任务 3，一直延后） |
| M7 | 漂浮物 / 半透明 | **漂浮物率**：侧相机里渲染不透明度 > 0.5 且渲染深度 < LiDAR 深度 × 0.8 的像素比例（用户 10-05 的"炫光"）；半透明率 | 侧相机 + M3 视角 | 半透明率已有；**新**：漂浮物率 |
| M8 | 位姿 | ATE、RPE（平移 / 旋转）、尺度比 | Sim(3) 对齐 | **新**（Phase 1 任务 4） |
| M9 | 生成占比 | 渲染像素里来自生成高斯的比例（我们的 `generated_ranges`） | M3 视角 | **新**（我们自己的方法；其他方法报"不适用"） |
| M10 | 代价 | 重建时间、峰值显存、渲染 FPS | — | meta.json 已记录大部分 |
| M11 | 人评（可选） | 两两偏好（2AFC），横移 / 转向视频，每对随机左右 | 10–20 人 × 30 对 | **新**，需要用户安排参与者 |

**报告规则**：
- 主表：M1、M2（侧相机，没见过区域单列）、M3（±3 m、±45°、+2 m）、M4、M7。几何、位姿、代价放补充表。
- 每个数字给场景平均和标准差；方法间差异用逐场景配对的 bootstrap 置信区间。
- 保真（M1、M2）和真实感（M3、M11）分开：生成方法可以在真实感上赢、在保真上输（E43 就是这样），两类都要显示。
- 失败（OOM、位姿发散）单独记数，不从平均里静默去掉。

## 5. 工作量与算力估计（单张 H100）

| 项 | 每场景 | 15 个测试场景 |
|---|---|---|
| 我们的完整方法（融合 → E41 → 路面 → 车辆 → 生成链） | 约 6–7 h | 约 100 h |
| A 类 5 个基线 | 约 0.5–1.5 h 每个 | 约 75 h |
| C 类 3 个生成基线 | 约 2–5 h 每个 | 约 150 h |
| 评测（全部指标） | 约 0.5 h 每方法 | 约 70 h |

合计约 400 GPU 小时，单卡连续跑约 2.5 周。可以先做 5 个测试场景的完整表（约 1 周）确认协议，再扩展。要写的评测代码：LiDAR 定义的未观测区域、NTA / NTL-IoU、漂浮物率、几何和位姿评测、FVD、MEt3R / OneSceneEval、EUVS 加载器，约 3–4 天。

## 6. 需要用户决定

1. **测试场景数量和来源**：Waymo validation 再下载 15–20 段（DECISIONS 里扩展场景需要用户同意；数据从哪里下载、用谁的凭证）。
2. **真实横移真值**：EUVS（直接下载，CC BY-SA）还是 Para-Lane（填表申请）还是两个都要。
3. **投稿目标**（会议 / 截稿时间）：决定基线做到哪一层（A + C 必须项约 1.5 周，全部约 3 周）。
4. **oracle 条件的方法**（ReconDreamer 等）：只引用论文数字，还是用 GT 条件跑一遍作参考行。
5. **人评**：是否做、谁参与。

## 7. 来源

- EUVS：[arXiv 2412.05256](https://arxiv.org/abs/2412.05256)、[项目页](https://ai4ce.github.io/EUVS-Benchmark/)
- MV2：[arXiv 2608.12442](https://arxiv.org/abs/2608.12442)
- Para-Lane：[arXiv 2502.15635](https://arxiv.org/abs/2502.15635)、[项目页](https://nizqleo.github.io/paralane-dataset/)
- XLD：[arXiv 2406.18360](https://arxiv.org/abs/2406.18360)、[项目页](https://3d-aigc.github.io/XLD/)
- WayveScenes101（离轴相机留出评测）：[arXiv 2407.08280](https://arxiv.org/abs/2407.08280)
- GA-Drive（NTA-IoU / NTL-IoU / FID，4 m 横移协议）：[arXiv 2602.20673](https://arxiv.org/abs/2602.20673)
- ReconDreamer：[arXiv 2411.19548](https://arxiv.org/abs/2411.19548)、[代码](https://github.com/GigaAI-research/ReconDreamer)；DriveDreamer4D：[arXiv 2410.13571](https://arxiv.org/html/2410.13571v3)、[代码](https://github.com/GigaAI-research/DriveDreamer4D)
- FaithFusion：[arXiv 2511.21113](https://arxiv.org/abs/2511.21113)；PRG-Fusion：[arXiv 2609.06948](https://arxiv.org/abs/2609.06948)
- LongSplat：[arXiv 2508.14041](https://arxiv.org/abs/2508.14041)；InstantSplat：[arXiv 2403.20309](https://arxiv.org/html/2403.20309v4)；CF-3DGS：[arXiv 2312.07504](https://arxiv.org/abs/2312.07504)
- AnySplat：[ACM TOG](https://dl.acm.org/doi/10.1145/3763326)；DrivingForward：[arXiv 2409.12753](https://arxiv.org/abs/2409.12753)；WorldSplat：[arXiv 2509.23402](https://arxiv.org/abs/2509.23402)
- GeoGS-SLAM：[arXiv 2607.11184](https://arxiv.org/abs/2607.11184)；GGS：[arXiv 2409.02382](https://arxiv.org/abs/2409.02382)
