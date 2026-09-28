# 参考文献

本目录收集对 DashRecon（单目前视视频 → 可自由视角渲染的静态街景）有参考价值的论文（用户要求，2026-09-28）。

- **PDF**：放在本目录，文件名是 `<arXiv 编号>_<简称>.pdf`。PDF 不入 git，原因是体积（45 篇约 876 MB），且每篇有各自的 arXiv 许可。用 `bash docs/papers/fetch.sh` 按 `papers.tsv` 重新下载。
- **核对**：编号和标题都在 2026-09-28 用 arXiv API 核对过。下面的描述只依据论文摘要或正文，以及本项目已经记录的实测结果（注明出处）。
- **维护**：新实验参考了新论文时，在同一次提交里补条目并更新"按实验查"。
- **状态**：
  - **已用**：已在实验里复现或直接使用；
  - **借鉴中**：部分思路已用到实验里；
  - **候选**：还没用，值得试；
  - **不适用**：前提条件（LiDAR、多相机、训练预算等）当前不满足。

## 按实验查

| 实验 | 参考 | 用到了什么 |
|---|---|---|
| 全部 | [OmniRe](#omnire) | drivestudio 框架（关闭动态建模，只训练 Background + Sky） |
| Phase 3、D-B1、E14 | [MapAnything](#mapanything) | 位姿 + 深度；给定内参和位姿时的联合深度 |
| E4c / E5c | [GLOMAP](#glomap) | 自标定（共享内参 + 径向畸变 + 位姿） |
| E5 / E5c、D-A2 / D-A3、E14 | [NKSR](#nksr)、[LSD-3D](#lsd-3d) | 网格；在网格上初始化 + 网格深度 / 法向正则；用离网格距离找垃圾高斯 |
| E6、E5pp | [LSD-3D](#lsd-3d) | GGDS（冻结 SDXL + 深度 ControlNet，不微调） |
| E7、E5cfx | [Difix3D+](#difix3d) | NVIDIA Fixer 渐进蒸馏 / 后处理 |
| E8 | [Wan](#wan)、[VACE](#vace)；思路来自 [ReconDreamer](#recondreamer)、[ViewCrafter](#viewcrafter) | Wan2.1-VACE-1.3B 补空洞；逐轮外扩 |
| E9 | [LSD-3D](#lsd-3d) | 自由视角 GGDS 复现 |
| DECISIONS P 的尝试 | [ReCamMaster](#recammaster) | 虚拟侧视重拍（行驶轨迹上失败） |
| U1 / U3 | [Street Gaussians](#street-gaussians) | oracle 上界（drivestudio 的 StreetGS 实现） |
| E10–E14 | [GEN3C](#gen3c)、[MoGe-2](#moge-2) | 3D 缓存条件的视频生成；逐帧度量深度（E10–E13） |
| E5f | [MTGS](#mtgs)、[2DGS](#2dgs)、[PGSR](#pgsr) | flatten 与大尺寸惩罚（贴表面的高斯） |
| D-A3、E14 | [VoD-3DGS](#vod-3dgs)、[MTGS](#mtgs) | 视角相关的不透明度（我们的观测锥屏蔽是手工版本）；每轮生成当作一次"虚拟经过"，外观单独建模 |
| 候选（Phase 9 修正任务 4） | [CogNVS](#cognvs) | "可见像素靠重建、隐藏像素靠视频补全" + 测试时微调 |
| 候选（E14 之后） | [FaithFusion](#faithfusion)、[RGE-GS](#rge-gs)、[DriveX](#drivex)、[ReconDreamer++](#recondreamer-1) | 像素级不确定性加权；筛选一致的生成内容；伪真值随模型逐步更新；地面几何冻结、只优化外观 |
| 候选（Mapillary，§14） | [MTGS](#mtgs)、[3DGUT](#3dgut)、[WildGaussians](#wildgaussians) | 多次经过融合；畸变与卷帘快门；外观嵌入 |
| 计划 D-E1（评测升级） | [MEt3R](#met3r)、[OneSceneEval](#onesceneeval)、[ReconDreamer](#recondreamer) | 不依赖真值的多视角一致性；基于 COLMAP 的一致性；NTA-IoU / NTL-IoU |
| 计划 E16（现成基线） | [Lyra](#lyra) | GEN3C + 3DGS 解码器，从单目视频直接生成 3DGS |
| 计划 E17（先几何后外观） | [InfiniCube](#infinicube)、[InfiniVerse](#infiniverse)、[SEM-ROVER](#sem-rover)、[LSD-3D](#lsd-3d)、[Lyra 2.0](#lyra-20) | 体素、占据或网格代理保证一致性，生成器只负责外观；历史帧检索 |
| 计划 E18（测试时适配） | [CogNVS](#cognvs)、[World from Motion](#world-from-motion) | 在本场景视频上自监督微调补全模型 |
| 计划 D-D1（蒸馏改进） | [DriveX](#drivex)、[FaithFusion](#faithfusion)、[FreeFix](#freefix) | 伪真值渐进更新；逐像素置信度 |
| 需要云 GPU | [Cosmos-Predict2.5 / Transfer2.5](#cosmos-predict25)、[Lyra 2.0](#lyra-20)、[Voyager](#voyager) | 显存或架构超出本机 A6000 |

## 条目

### 已用

#### LSD-3D
*LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding*，arXiv [2508.19204](https://arxiv.org/abs/2508.19204)（2025-08），`2508.19204_LSD-3D.pdf`
- **做了什么**：先生成代理几何和环境表示，再用学到的 2D 图像先验做分数蒸馏（GGDS），得到几何准确、可做因果新视角合成的大尺度 3D 驾驶场景，可用地图布局做条件。代码未公开（项目页写 "Code (tba)"，DECISIONS O）。
- **已借鉴**：
  - 几何部分：在 NKSR 网格面上初始化高斯，用网格渲染的深度和法向做正则（D9 → E5 / E5c，当前所有实验的基线）。
  - 外观部分：GGDS（E6、E9）。LSD-3D 先在 Waymo 上微调了 SDXL 再冻结；我们没有微调（D10），E6 比 E5 留出帧低 0.19 dB，E9 不锚定真实帧时发散、锚定后偏色（DECISIONS K、O）。
- **仍可借鉴**：以保真换真实感的整体路线，以及用地图布局控制生成。如果以后允许做领域适配（DECISIONS S 的准入条件），"在驾驶数据上适配扩散先验"是它和我们差距最大的一环。
- **状态**：已用。

#### Difix3D+
*Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models*，arXiv [2503.01774](https://arxiv.org/abs/2503.01774)（2025-03），`2503.01774_Difix3D+.pdf`
- **做了什么**：单步图像扩散模型 Difix 去掉新视角渲染里欠约束区域的伪影。两个用途：重建阶段清理伪训练视角后蒸馏回 3D；渲染时做实时增强。
- **已借鉴**：E7（NVIDIA Fixer 按 Difix3D+ 的做法渐进蒸馏）和 E5cfx（逐帧后处理）。E7 与 E5c 基本持平；E5cfx 使留出帧 LPIPS 0.210 → 0.183；未观测区域仍然是空的（DECISIONS M，OPEN_QUESTIONS 35）。
- **结论**：修复型模型能改善观感，但补不出没见过的内容。
- **状态**：已用。

#### GEN3C
*GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control*，arXiv [2503.03751](https://arxiv.org/abs/2503.03751)（2025-03），`2503.03751_GEN3C.pdf`
- **做了什么**：用 3D 缓存引导视频生成。缓存是对种子图像或已生成帧预测深度得到的点云；生成新帧时以缓存在新相机下的渲染为条件，模型不必记住生成过的内容，也不必从相机位姿推断结构。
- **已借鉴**：E10–E14 的生成器（GEN3C-Cosmos-7B，冻结）。实测：
  - 缓存用真实 FRONT 帧 warp 明显好于用 3DGS 渲染（DECISIONS R 第 2 步）；
  - 代码里最多 2 个缓存（`frame_buffer_max = 2`）；
  - 缓存覆盖不到的地方会编造内容（远山、红车）。
- **仍可借鉴**：论文里"已生成帧的深度也进入缓存"，对应我们的跨轮生成记忆（E11–E14）。目前记忆的覆盖率还很低（E11 为 0.46%）。
- **状态**：已用。

#### Wan
*Wan: Open and Advanced Large-Scale Video Generative Models*，arXiv [2503.20314](https://arxiv.org/abs/2503.20314)（2025-03），`2503.20314_Wan.pdf`
- **做了什么**：开源视频基础模型（DiT，新 VAE，最大 14B）。
- **已借鉴**：Wan2.1-T2V-1.3B 做 SDEdit 重绘测试（没有变清晰，DECISIONS N）；E8 使用的是它上面的 VACE。
- **状态**：已用。

#### VACE
*VACE: All-in-One Video Creation and Editing*，arXiv [2503.07598](https://arxiv.org/abs/2503.07598)（2025-03），`2503.07598_VACE.pdf`
- **做了什么**：统一的视频生成与编辑框架，用 Video Condition Unit 统一参考、编辑、掩码输入，支持"视频 + 掩码"的补全。
- **已借鉴**：E8 的空洞补全（Wan2.1-VACE-1.3B）。侧视未观测区域 +0.5 dB（16.56 → 17.12），但画面糊（DECISIONS N）。
- **状态**：已用。

#### ReCamMaster
*ReCamMaster: Camera-Controlled Generative Rendering from A Single Video*，arXiv [2503.11647](https://arxiv.org/abs/2503.11647)（2025-03），`2503.11647_ReCamMaster.pdf`
- **做了什么**：以输入视频为条件、在新相机轨迹下重新渲染同一动态场景；训练数据是用 UE5 合成的多相机同步视频。
- **已借鉴**：DECISIONS P 的虚拟侧视尝试。预设轨迹画面真实，但车停在第一帧不动；沿行驶轨迹转向时画面崩成色块（81 帧里车前进约 39 个场景单位，超出它训练时见过的相机运动）。
- **状态**：已用，不适合驾驶轨迹。

#### MapAnything
*MapAnything: Universal Feed-Forward Metric 3D Reconstruction*，arXiv [2509.13414](https://arxiv.org/abs/2509.13414)（2025-09），`2509.13414_MapAnything.pdf`
- **做了什么**：前馈 Transformer，输入一张或多张图像，可选内参、位姿、深度等几何输入，直接回归度量尺度的几何和相机；表示为深度图、局部射线图、相机位姿和一个度量尺度因子。
- **已借鉴**：
  - Phase 3 的位姿与深度；E5c 用它给 GLOMAP 位姿补稠密深度。
  - D-B1：给定内参和位姿、对生成帧联合推理的深度，让验收通过率在 E13 生成帧上从 0.11% 提高到 3.6%，在 E12 drive 上从 8.7% 提高到 20.2%；E14 第 0 轮达到 25.4%。
- **注意**：给定位姿时，输出位姿仍以第 0 帧为参考系（`model.py` "in the frame of the reference view 0"，DECISIONS W1）。大偏航的生成帧上，它预测的焦距可能偏离给定内参 27–76%（DECISIONS N）。
- **状态**：已用。

#### MoGe-2
*MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details*，arXiv [2507.02546](https://arxiv.org/abs/2507.02546)（2025-07），`2507.02546_MoGe-2.pdf`
- **做了什么**：单张图像恢复度量尺度的 3D 点图，细节锐利。
- **已借鉴**：E10–E13 生成帧的深度（给定视场角）。逐帧独立推理，不同帧之间不一致，是 E13 验收几乎全被拒的主因之一（D-B1）；E14 改用 MapAnything。
- **状态**：已用（已被替换）。

#### OmniRe
*OmniRe: Omni Urban Scene Reconstruction*，arXiv [2408.16760](https://arxiv.org/abs/2408.16760)（2024-08），`2408.16760_OmniRe.pdf`
- **做了什么**：基于 3DGS 场景图的城市场景重建，车辆、行人、骑车人等动态物体各自在规范空间建模。drivestudio 是它的官方代码库。
- **已借鉴**：整个项目的框架。我们关闭动态建模，只训练 Background + Sky（C1）。
- **状态**：已用。

#### Street Gaussians
*Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting*，arXiv [2401.01339](https://arxiv.org/abs/2401.01339)（2024-01），`2401.01339_StreetGaussians.pdf`
- **做了什么**：背景和前景车辆各用一组带语义的点云 + 3D 高斯，车辆位姿可优化，外观用 4D 球谐。
- **已借鉴**：U1 / U3 oracle 上界（drivestudio 的 StreetGS 实现，真值位姿 + LiDAR，DECISIONS Q）。
- **状态**：已用（只做上界）。

#### GLOMAP
*Global Structure-from-Motion Revisited*，arXiv [2407.20219](https://arxiv.org/abs/2407.20219)（2024-07），`2407.20219_GLOMAP.pdf`
- **做了什么**：全局式 SfM，精度与 COLMAP 相当，快几个数量级。
- **已借鉴**：E4c / E5c 的相机自标定（共享 RADIAL 内参 + 位姿）。5 个场景的留出帧均值 +3.8 dB（DECISIONS L）。
- **状态**：已用。

#### NKSR
*Neural Kernel Surface Reconstruction*，arXiv [2305.19590](https://arxiv.org/abs/2305.19590)（2023-05），`2305.19590_NKSR.pdf`
- **做了什么**：从大规模、稀疏、有噪声的点云重建隐式曲面，可扩展到大场景。
- **已借鉴**：
  - Phase 7 的网格，也是 E5 / E5c 的初始化和正则；
  - D-A2 用"离网格距离"找出侧视垃圾高斯；
  - D-A3 / E14 的屏蔽只作用于离网格 > 0.3 的高斯。
- **注意**：网格只覆盖融合点云的范围，远处的树冠和房子不在网格上（D-A2）。
- **状态**：已用。

### 借鉴中

#### MTGS
*MTGS: Multi-Traversal Gaussian Splatting*，arXiv [2503.12552](https://arxiv.org/abs/2503.12552)（2025-03），`2503.12552_MTGS.pdf`
- **做了什么**：用同一路段的多次经过重建。
  - 场景图：共享的静态节点（位置、形状、不透明度和 SH 零阶系数共享），每次经过各自的外观节点（SH 高阶系数残差），加上各自的动态节点。
  - 每个相机一个仿射颜色变换。
  - 正则：LiDAR 逆深度 0.5、单目深度 patch NCC 0.1、伪深度法向 0.1、flatten 1.0。
  - 评测：留出一整次、与其他经过几乎不重叠的经过。
  - 结果：比单次经过 LPIPS 0.313 → 0.265，AbsRel 0.145 → 0.089。
  - 作者承认的局限：所有经过都没看到的区域（如停放车辆下方）仍有漂浮物；反向经过时的卷帘快门会造成错位。
- **已借鉴**：
  - E5f：flatten 与大尺寸惩罚。FRONT 留出更清晰（清晰度比 0.68 → 0.82），侧视大糊团变细碎点。
  - E14：每轮生成帧当作一次"虚拟经过"，带单独的 3×4 仿射外观（MTGS 外观节点的简化版，没有做 SH 残差）。
- **仍可借鉴**：
  - 不依赖尺度的单目深度 NCC 损失，可以覆盖网格之外的远处；
  - 多次经过融合（§14 Mapillary 计划，唯一能让侧面内容"真实"的办法）；
  - "留出一整次经过"的评测协议。
- **状态**：借鉴中。

#### VoD-3DGS
*VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting*，arXiv [2501.17978](https://arxiv.org/abs/2501.17978)（2025-01），`2501.17978_VoD-3DGS.pdf`
- **做了什么**：给每个高斯加一个对称矩阵，让不透明度随视角变化（用 SGGX 投影面积），使某些高斯在特定视角下被抑制；动机是镜面高光和反射。渲染仍 > 60 FPS。
- **和我们的关系**：D-A3 / E14 的观测锥屏蔽是手工规则版的"视角相关不透明度"：高斯在其训练观测方向之外淡出。VoD-3DGS 说明这类表示可以做成可学习的、渲染开销很小。
- **可借鉴**：如果 E14 的屏蔽有效，可以把它换成可学习的视角相关不透明度，用真实 FRONT 和通过验收的生成视角一起训练，并作为模型表示的一部分导出（现在的屏蔽参数是单独的 `view_gate.pt`）。
- **状态**：借鉴中（思路）。

#### 2DGS
*2D Gaussian Splatting for Geometrically Accurate Radiance Fields*，arXiv [2403.17888](https://arxiv.org/abs/2403.17888)（2024-03），`2403.17888_2DGS.pdf`
- **做了什么**：把 3D 体积压成有朝向的 2D 高斯圆盘，配合透视正确的光线-圆盘求交、深度失真和法向一致性项，得到视角一致的几何。
- **可借鉴**：E5f 的 flatten 是它的弱化版。D-A2 的垃圾是"厚而半透明、靠叠加混色"的高斯，面片表示从根本上限制这种作弊。以后做道路和碰撞几何（可测试资产）时也需要。
- **状态**：借鉴中（E5f）；完整替换为候选。

#### PGSR
*PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction*，arXiv [2406.06521](https://arxiv.org/abs/2406.06521)（2024-06），`2406.06521_PGSR.pdf`
- **做了什么**：平面化高斯 + 无偏深度渲染（直接渲染相机到高斯平面的距离和法向），多视角几何一致性。
- **可借鉴**：同 2DGS，偏重表面精度；可作为导出道路 / 碰撞网格的备选。
- **状态**：候选。

### 候选

#### ReconDreamer
*ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration*，arXiv [2411.19548](https://arxiv.org/abs/2411.19548)（2024-11），`2411.19548_ReconDreamer.pdf`
- **做了什么**：逐步把世界模型的知识并入驾驶场景重建。DriveRestorer 在线修复新轨迹渲染里的伪影，配合渐进的数据更新策略，能处理多车道横移这类大机动。报告中 NTA-IoU、NTL-IoU、FID 相对 Street Gaussians 分别提升 24.87%、6.72%、29.97%。
- **和我们的关系**：Phase 9 的"渲染 → 修复 / 补全 → 蒸馏 → 扩大偏离范围"循环与它同构（E8、E10–E14）。
- **可借鉴**：渐进式扩大偏离角度（我们按 ±45° → ±90° 排）；用 NTA-IoU / NTL-IoU（车辆和车道线在新视角下的一致性）这类驾驶相关指标补充评测（OPEN_QUESTIONS / Phase 1 延后项）。
- **注意**：修复模型需要在驾驶数据上训练；我们目前只用现成权重。
- **状态**：候选（思路已在用）。

#### ReconDreamer++
*ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation*，arXiv [2503.18438](https://arxiv.org/abs/2503.18438)（2025-03），`2503.18438_ReconDreamer++.pdf`
- **做了什么**：
  - NTDNet：可学习的空间形变，弥合合成新视角与真实传感器观测之间的域差；
  - 对地面这类结构化元素，保留高斯里的几何先验，只优化外观属性。
- **可借鉴**：
  - E14 之后：蒸馏时冻结离网格近的高斯（路面）的几何，只让生成视角改外观，避免生成内容拉歪路面；
  - 用可学习的形变吸收生成帧与真实帧的错位。这比 E14 每轮只做一个颜色仿射更强。
- **状态**：候选。

#### DriveX
*Driving View Synthesis on Free-form Trajectories with Generative Prior*，arXiv [2412.01717](https://arxiv.org/abs/2412.01717)（2024-12），`2412.01717_DriveX.pdf`
- **做了什么**：在 3DGS 优化过程中渐进地蒸馏视频扩散先验：
  - 用视频扩散模型修复当前模型的新轨迹渲染，修复后的视频作为额外监督；
  - 把修复写成补全（inpainting）任务，把"哪里退化"和"生成能力"分开；
  - 伪真值随渲染变好而不断更新，两者相互促进。
- **和我们的关系**：最接近 Phase 9 的已发表方法。我们每一轮整段重新生成一次；它在优化过程中持续更新伪真值。
- **可借鉴**：伪真值随模型更新（减少 E8 / E10 那种"各轮目标互相冲突，被平均掉"）；补全式的任务定义与我们的空洞掩码一致。
- **状态**：候选。

#### FaithFusion
*FaithFusion: Harmonizing Reconstruction and Generation via Pixel-wise Information Gain*，arXiv [2511.21113](https://arxiv.org/abs/2511.21113)（2025-11），`2511.21113_FaithFusion.pdf`
- **做了什么**：用像素级期望信息增益（EIG）统一两件事：引导扩散模型只修高不确定性的区域；按像素权重把修改蒸馏回 3DGS。即插即用，不需要额外条件，也不改结构。在 Waymo 上 6 m 横移时 FID 为 107.47。
- **和我们的关系**：直接对应我们的"验收 + 置信度加权"（E11–E14）。我们现在是手工规则：通过的像素权重 1，无证据 0.3，冲突 0。
- **可借鉴**：用基于不确定性的像素权重替换手工权重，并用同一张权重图决定生成器该改哪里（现在由空洞掩码决定）。
- **状态**：候选，优先级高。

#### RGE-GS
*RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors*，arXiv [2506.22800](https://arxiv.org/abs/2506.22800)（2025-06），`2506.22800_RGE-GS.pdf`
- **做了什么**：针对单趟行驶扫描不全的场景扩展重建。用一个奖励网络在重建前识别并优先保留一致的生成内容；重建时按场景收敛情况差异化地调整高斯优化进度。
- **和我们的关系**：奖励网络对应我们的几何验收（E11–E14），但它是学出来的。
- **可借鉴**：按收敛情况调整优化（E14 各轮步数固定）。学习型奖励需要训练，要满足 DECISIONS S 的准入条件。
- **状态**：候选。

#### ViewCrafter
*ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis*，arXiv [2409.02048](https://arxiv.org/abs/2409.02048)（2024-09），`2409.02048_ViewCrafter.pdf`
- **做了什么**：以点云渲染为粗 3D 线索，用视频扩散模型生成精确相机控制的新视角；迭代合成并规划相机轨迹，逐步扩大点云和新视角的覆盖范围。
- **和我们的关系**：逐轮外扩与轨迹规划（E12 行驶 vs 扫视、E13 环视、E14 的轮次顺序）。
- **可借鉴**：按"下一步能看到多少新区域"来规划相机轨迹，而不是固定角度序列。
- **状态**：候选（思路已在用）。

#### CogNVS
*Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-view Synthesis from Monocular Videos*，arXiv [2507.12646](https://arxiv.org/abs/2507.12646)（2025-07），`2507.12646_CogNVS.pdf`
- **做了什么**：新视角里两类像素都可见的由重建渲染；只在新视角可见的由视频扩散模型补全。补全模型可以在 2D 视频上自监督训练，所以能在新视频上做测试时微调后零样本使用。
- **和我们的关系**：与我们的分解完全一致（真实观测负责可见部分，生成负责空洞）。DECISIONS S2 核查过：有可直接用的补全权重，但官方说明不做测试时微调时质量较低；完整微调配置需要至少 5 张 48 GB A6000。
- **可借鉴**：Phase 9 修正任务 4"完整现成模型基线"的候选。测试时微调在 DECISIONS S 下允许，但本机资源不够跑官方配置，只能做受限版本并如实标注。
- **状态**：候选。

#### DriveDreamer4D
*DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation*，arXiv [2410.13571](https://arxiv.org/abs/2410.13571)（2024-10），`2410.13571_DriveDreamer4D.pdf`
- **做了什么**：把驾驶世界模型当作"数据机器"，生成新轨迹（变道、加减速）的视频来补充 4D 重建的训练数据。
- **可借鉴**：早期的"生成视频监督重建"方案，是 ReconDreamer 的前身。
- **状态**：候选（参考）。

#### FreeVS
*FreeVS: Generative View Synthesis on Free Driving Trajectory*，arXiv [2410.18079](https://arxiv.org/abs/2410.18079)（2024-10），`2410.18079_FreeVS.pdf`
- **做了什么**：完全生成式的新轨迹视角合成。用视角先验的"伪图像"表示控制生成，训练时在伪图像上模拟各方向的相机运动；训练好后无需重建即可用于新序列。
- **可借鉴**：与 GEN3C 的缓存条件思路相同；它在驾驶数据上训练，对驾驶相机运动的适应性可能比通用模型好。需要训练，另外论文的评测侧重生成视频，不直接产出 3D。
- **状态**：候选（参考）。

#### SGD
*SGD: Street View Synthesis with Gaussian Splatting and Diffusion Prior*，arXiv [2403.20079](https://arxiv.org/abs/2403.20079)（2024-03），`2403.20079_SGD.pdf`
- **做了什么**：微调扩散模型，以相邻帧图像为条件，并用 LiDAR 深度提供空间信息，作为 3DGS 在偏离视角上的先验。
- **可借鉴**："相邻帧作为条件"与我们用 t−6 / t−15 真实帧 warp 作 GEN3C 缓存的思路相同。
- **状态**：不适用（需要 LiDAR 和微调），仅作参考。

#### StreetCrafter
*StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models*，arXiv [2412.13188](https://arxiv.org/abs/2412.13188)（2024-12），`2412.13188_StreetCrafter.pdf`
- **做了什么**：以 LiDAR 点云渲染为像素级条件的可控视频扩散模型，用于偏离轨迹的街景合成，并能融入动态场景表示以实时渲染。
- **可借鉴**：如果把 LiDAR 条件换成我们自己的融合点云或网格渲染，概念上与 GEN3C 缓存相同，但需要训练。
- **状态**：不适用（需要 LiDAR 条件和训练）。

#### Cosmos-Drive-Dreams
*Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models*，arXiv [2506.09042](https://arxiv.org/abs/2506.09042)（2025-06），`2506.09042_Cosmos-Drive-Dreams.pdf`
- **做了什么**：基于 NVIDIA Cosmos 的驾驶专用模型套件（Cosmos-Drive），做可控、多视角、时空一致的驾驶视频生成，用于合成数据。
- **可借鉴**：多视角生成正是我们缺的侧面和后方。但按 `docs/SCENE_COMPLETION_ANALYSIS.md` 的核查：控制输入需要 HD 地图或 LiDAR，Waymo 示例要求自己的后训练 checkpoint，完整多视角不适合本机单卡。
- **状态**：不适用（当前），留作以后有地图或更多算力时的方案。

#### AutoSplat
*AutoSplat: Constrained Gaussian Splatting for Autonomous Driving Scene Reconstruction*，arXiv [2407.02598](https://arxiv.org/abs/2407.02598)（2024-07），`2407.02598_AutoSplat.pdf`
- **做了什么**：对表示路面和天空的高斯施加几何约束，使变道等场景的多视角渲染一致；用 3D 模板和反射一致性约束监督前景物体看不到的一面。
- **可借鉴**：路面和天空的几何约束（我们有路面掩码和 NKSR 网格，可以把路面高斯约束在网格上）。前景部分不适用，因为我们掩掉动态物体。
- **状态**：候选。

#### WildGaussians
*WildGaussians: 3D Gaussian Splatting in the Wild*，arXiv [2407.08447](https://arxiv.org/abs/2407.08447)（2024-07），`2407.08447_WildGaussians.pdf`
- **做了什么**：用 DINO 特征处理遮挡，在 3DGS 里加外观建模模块处理光照变化。MTGS 把它作为逐图像外观嵌入的对照：LPIPS 0.300，MTGS 0.271。
- **可借鉴**：生成帧的外观漂移（E9 偏色、GEN3C 阴天）可以用更强的外观嵌入吸收，E14 目前只有每轮一个仿射变换。Mapillary 多次经过也会遇到同样的问题。
- **状态**：候选。

#### 3DGUT
*3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting*，arXiv [2412.12507](https://arxiv.org/abs/2412.12507)（2024-12），`2412.12507_3DGUT.pdf`
- **做了什么**：用无迹变换代替 EWA 投影，任意非线性投影都能精确投影，从而支持畸变相机和卷帘快门这类随时间变化的效果，同时保持光栅化的效率。
- **可借鉴**：
  - Mapillary 行车记录仪的广角、鱼眼和卷帘快门（§14 第 2 条）；
  - U3 在原轨迹上变差的一个候选原因是多相机的曝光时刻和卷帘快门没有建模（DECISIONS Q）。要得到"多相机一致"的上界，需要这类建模。
  - 目前 E5c 是在去畸变后的图像上重建（D13）。
- **状态**：候选（Mapillary 阶段）。

### 候选（E14 之后的计划，2026-09-28 加入，见 [PLAN_AFTER_E14.md](../PLAN_AFTER_E14.md)）

#### Lyra
*Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation*，arXiv [2509.19296](https://arxiv.org/abs/2509.19296)（2025-09，ICLR 2026），`2509.19296_Lyra.pdf`
- **做了什么**：给视频扩散模型的 RGB 解码器旁边加一个 3DGS 解码器，用 RGB 解码的结果监督它；训练数据完全由视频模型自己生成，不需要多视角数据。推理时从文本或单张图像生成 3D 场景，也支持从单目视频生成动态 3D。底层是 GEN3C。
- **可行性**（官方仓库）：代码 Apache 2.0，权重是 NVIDIA Open Model License；只在 H100 / A100 上测过，完全卸载时显存峰值约 43 GB；视频输入需要用 ViPE 预先提取深度、内参和位姿。
- **对应**：计划 E16，作为"单目视频 → 可渲染 3D、允许想象"的现成基线（Phase 9 修正任务 4）。
- **状态**：候选。

#### Lyra 2.0
*Lyra 2.0: Explorable Generative 3D Worlds*，arXiv [2604.13036](https://arxiv.org/abs/2604.13036)（2026-04），`2604.13036_Lyra-2.0.pdf`
- **做了什么**：针对长轨迹生成的两种退化：一是空间遗忘（回到看过的地方时要重新编），二是时间漂移（自回归误差累积）。
  - 对策一：维护每帧的 3D 几何，但只用于检索相关的历史帧、建立与目标视角的稠密对应；外观完全交给生成先验。
  - 对策二：训练时加入模型自己生成的历史，让模型适应它自己的误差。
  - 基于 Wan2.1-14B；用 Depth Anything v3 前馈预测每像素的 3DGS 属性。
  - 在 GB200 上每 80 帧约 194 秒，蒸馏版约 15 秒。
- **可借鉴**：按可见性从所有真实帧和生成帧中检索缓存来源（我们的 GEN3C 只有 2 个缓存）；用前馈模型预测每像素 3DGS 属性，代替各向同性的新高斯。
- **对应**：计划 E17 的历史帧检索；模型本身需要更大的 GPU。
- **状态**：候选（思路）。

#### Voyager
*Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation*，arXiv [2506.04225](https://arxiv.org/abs/2506.04225)（2025-06，ACM TOG），`2506.04225_Voyager.pdf`
- **做了什么**：从单张图像和相机路径联合生成对齐的 RGB 与深度视频，用可扩展的世界缓存（点云，带剔除）保持一致；自回归地扩展场景，不需要 SfM / MVS。
- **可行性**（官方仓库）：540p 至少需要 60 GB 显存，推荐 80 GB；输入是单张图像加预设的相机动作，不能以已有的点云或重建作为条件。
- **可借鉴**：直接生成 RGB-D，省掉单独估深度这一步（D-B1 说明深度一致性是关键）。本机跑不了。
- **状态**：不适用（本机），思路参考。

#### InfiniCube
*InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models*，arXiv [2412.03934](https://arxiv.org/abs/2412.03934)（2024-12，ICCV 2025），`2412.03934_InfiniCube.pdf`
- **做了什么**：三步。先用以 HD 地图为条件的稀疏体素生成模型生成无界的体素世界；再用一组贴像素的引导缓冲，把视频模型"钉"在体素世界上生成一致的外观；最后用体素和像素两个分支，前馈地把视频提升为动态 3D 高斯。
- **可借鉴**："先几何、后外观"。几何在 3D 里统一，视频模型只负责外观，一致性由几何保证。它需要 HD 地图，我们可以用想象网格代替。
- **对应**：计划 E17。
- **状态**：候选（思路）。

#### InfiniVerse
*InfiniVerse: Occupancy Guided Unbounded Scene Generation for Autonomous Driving*，arXiv [2606.31109](https://arxiv.org/abs/2606.31109)（2026-06），`2606.31109_InfiniVerse.pdf`
- **做了什么**：从单帧（多视角）重建 3D 占据栅格，沿任意轨迹自回归地扩展；视频扩散模型把粗占据栅格转成真实的视频；生成的视频再投影回去修正占据栅格（"草图—细化"），在视觉和空间之间相互增强。在 Waymo 和 nuScenes 上评测。
- **可借鉴**：生成结果回写并修正几何、再生成的闭环，对应 E17 第 4 步。
- **状态**：候选（思路）。

#### SEM-ROVER
*SEM-ROVER: Semantic Voxel-Guided Diffusion for Large-Scale Driving Scene Generation*，arXiv [2604.06113](https://arxiv.org/abs/2604.06113)（2026-04），`2604.06113_SEM-ROVER.pdf`
- **做了什么**：用 Σ-Voxfield（每个占据体素存固定数量的带颜色表面采样）表示场景，以语义为条件的扩散模型在局部体素邻域上生成，通过重叠区域的渐进外扩扩展到大场景，再用延迟渲染得到照片级图像，不需要逐场景优化。
- **可借鉴**：直接在 3D 里生成，从根本上避免多视角不一致；按语义条件生成，与我们的路面和天空分割可以衔接。需要训练。
- **状态**：候选（参考）。

#### World from Motion
*World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video*，arXiv [2607.01202](https://arxiv.org/abs/2607.01202)（2026-07，NVIDIA），`2607.01202_WorldFromMotion.pdf`
- **做了什么**：以沿输入和目标相机轨迹、逐像素对齐的渲染（外观、几何、3D 运动）为条件，让视频模型修正初始重建的伪影、补全缺失区域；测试时把生成内容（包括新看到的区域和运动）蒸馏回一个一致的动态 3DGS。训练数据是带模拟单目重建伪影的多视角视频对。
- **和我们的关系**：与 Phase 9 的做法最接近的已发表方法（重建 → 条件生成 → 蒸馏回单个 3DGS），区别是它为这个任务训练了专门的模型。项目页暂未见代码或权重。
- **可借鉴**：以"渲染的几何 + 外观"为条件，而不只是 RGB warp；用模拟的重建伪影构造训练数据（如果将来做 E18 这类适配）。
- **状态**：候选（参考）。

#### FreeFix
*FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models*，arXiv [2601.20857](https://arxiv.org/abs/2601.20857)（2026-01），`2601.20857_FreeFix.pdf`
- **做了什么**：不微调，用预训练图像扩散模型增强外推视角的渲染；2D / 3D 交替细化；逐像素置信度掩码只改不确定的区域。报告的一致性和效果与需要微调的方法相当或更好。
- **可借鉴**：计划 D-D1，免训练的逐像素置信度细化，可与 E14 的验收权重结合。
- **状态**：候选。

#### Instant NuRec
*Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation*，arXiv [2607.14203](https://arxiv.org/abs/2607.14203)（2026-07，NVIDIA），`2607.14203_InstantNuRec.pdf`
- **做了什么**：前馈模型，把标定过的多相机短驾驶片段一次前向变成可仿真的 3DGS 世界：静态和动态两层、天空立方体贴图、每个相机的 ISP 校正，原生支持 3DGUT 的非针孔相机。10–20 秒的场景约 1.5 秒重建完，在 Waymo 上比最强基线高 2.01 dB；与 NuRec 和 AlpaSim 集成。
- **可借鉴**：输出的分层结构（静态 / 动态 / 天空 / ISP）适合做可测试资产，将来和 AlpaSim 集成时参考。需要标定过的多相机，单个 FRONT 用不了。
- **状态**：不适用（当前）。

#### Cosmos-Predict2.5
*World Simulation with Video Foundation Models for Physical AI*，arXiv [2511.00062](https://arxiv.org/abs/2511.00062)（2025-11，NVIDIA），`2511.00062_Cosmos-Predict2.5.pdf`
- **做了什么**：
  - Cosmos-Predict2.5：基于 flow，把 Text2World、Image2World、Video2World 统一在一个模型里，用 Cosmos-Reason1 做文本对齐，在 2 亿条视频上训练，有 2B 和 14B 两个版本。
  - Cosmos-Transfer2.5：ControlNet 式的 Sim2Real / Real2Real 转换，比 Transfer1 小 3.5 倍、保真度更高。
  - 代码和权重以 NVIDIA Open Model License 发布（2025-10）。
- **可行性**（官方文档）：
  - Predict2.5 支持 Ampere 及更新的架构，但基础模型没有 3D 缓存或相机控制；驾驶多视角版 `auto/multiview` 需要 7 个相机的输入。
  - Transfer2.5-2B 以深度、分割、边缘、模糊为控制，单卡需要 65.4 GB 显存，且要求 Hopper 或更新的架构，本机 A6000 不满足。
- **可借鉴**：用我们渲染的深度和语义作为 Transfer2.5 的控制，生成符合几何的真实外观。需要云 GPU，要用户决定。
- **状态**：不适用（本机）。

#### MEt3R
*MEt3R: Measuring Multi-View Consistency in Generated Images*，arXiv [2501.06336](https://arxiv.org/abs/2501.06336)（2025-01），`2501.06336_MEt3R.pdf`
- **做了什么**：不需要真值的多视角一致性指标。用 DUSt3R 对一对图像做稠密重建，把一张 warp 到另一张，再比较特征，对视角相关的效果不敏感。
- **对应**：计划 D-E1。对"允许想象"的侧视补全，衡量同一地点从不同视角看是否一致，比逐像素和真实侧相机比较更贴近目标。
- **状态**：候选。

#### OneSceneEval
*Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate*，arXiv [2605.18754](https://arxiv.org/abs/2605.18754)（2026-05），`2605.18754_OneSceneEval.pdf`
- **做了什么**：指出 VGGT、MASt3R、DUSt3R、Fast3R 会对无关场景、重复图像甚至噪声"幻觉"出稠密几何和跨视角支持，所以 MEt3R 这类神经指标可能给坏结果打高分。提出更稳健的神经指标变体，以及基于 COLMAP（匹配、注册、稠密支持、重建失败）的一致性指标，与人工评判的相关性最高提升 4 倍。
- **对应**：计划 D-E1。一致性指标用它的 COLMAP 版本作主指标，MEt3R 作辅助。这与我们在几何验收里"不能把缺乏证据记为通过"的原则一致。
- **状态**：候选。

#### GLADOS
*Mind the Gap: Geometrically Accurate Generative Reconstruction from Disjoint Views*，arXiv [2605.07550](https://arxiv.org/abs/2605.07550)（2026-05），`2605.07550_GLADOS.pdf`
- **做了什么**：针对视角之间完全不重叠的情形。先用基础模型生成中间视角把不相交的输入连起来，再用全局对齐建立粗几何支架、吸收生成过程的局部矛盾，然后迭代扩展上下文、补全缺失区域并优化一致性。
- **可借鉴**："粗几何支架吸收生成矛盾"的思路对应 E17；将来 Mapillary 多段视频之间几乎不重叠时也相关。
- **状态**：候选（参考）。

#### FocusGS
*Targeted Structure Completion for Sparse-View 3D Reconstruction in Autonomous Driving*，arXiv [2607.04661](https://arxiv.org/abs/2607.04661)（2026-07），`2607.04661_FocusGS.pdf`
- **做了什么**：不做全局的体素化补全，而是先求出"几何歧义流形"，定位容易被遮挡、几何不确定的局部区域，只在这些区域里实例化并优化高斯。
- **可借鉴**：只在几何歧义区域补全、确定区域不动，与我们"已观测区域不被生成改写"一致。它的歧义区域定位可以替代我们手工的空洞判断。
- **状态**：候选（参考）。
