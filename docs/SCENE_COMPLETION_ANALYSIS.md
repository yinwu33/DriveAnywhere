# 单目街景补全：方案分析与验证协议

日期：2026-09-27。用户要求记录分析并按规划验证；对应 DECISIONS T。

## 目标与判断

目标是从 FRONT 单目视频得到可自由渲染的静态场景。隐藏区域允许与真实世界不同，但必须在平移、转向和返回时保持同一场景的结构、遮挡和外观。生成视频的真实感不等于可蒸馏的 3D 一致性。

采用“基础重建 → 目标视角 → 生成候选 → 几何验收 → 局部 3D 更新 → 再生成”的循环。真实 FRONT 与已经通过验证的生成 RGB-D 共同构成空间记忆；真实内容优先。3DGS 提供几何和可见性，真实图像提供清晰外观，生成记忆约束已经确定的隐藏内容。不能仅把模糊渲染再次交给生成器。

隐藏区域的不同候选应作为不同场景假设，不能把互相矛盾的建筑、树木等一起平均拟合。固定随机种子用于复现，本身不保证跨视角一致。未知区域先保留为候选，邻近视角具有足够视差与支持后才能晋升为稳定记忆。相互支持只证明自洽，不证明隐藏内容真实。

需要联合约束深度、对应点、遮挡和颜色；逐帧深度估计后仅对齐尺度不够。纯旋转缺少确定深度的视差，验证必须增加平移。生成出的动态内容不能直接当成静态场景监督。

未来“可测试资产”宜包括 3DGS 外观、道路/碰撞几何、道路语义和真实/生成来源。闭环仿真和物理距离评测还需要尺度验收，当前不启动这些后续任务。

## 现有证据及其边界

- E10 使用真实 FRONT warp 后，生成外观比旧的渲染缓存更自然，但大角度 3D 渲染仍有碎片、拖影和模糊。目标冲突、深度误差和拟合能力是待分离的原因，不能仅凭截图确定主因。
- `scripts/run_phase9.sh` 的旧路径给 GEN3C 的缓存仍是 `real0 real1`；历史生成用于空洞判断和最终拟合，没有充分进入生成缓存。新 `consistent` 路径已经实现。
- 磁盘上的 E11 8 步缓存接线实验已完成生成、深度与几何检查（`results/E11/val056/run_status.json`），修正 AGENTS/DECISIONS S 中“实验待运行”的旧状态；3D 蒸馏与画质验收尚未完成。
- 8 步结果：空洞内自洽验收比例 real-only 12.6768%，memory 22.0166%；无证据比例 27.2236% → 13.7532%。memory 平均仅占整幅图的 0.3599%。不把这些比例称为正确率，不把小覆盖率下的一次随机对照称为完整记忆机制成功。
- E10 及复用它的 E11 产物可能间接含旧 FRONT 留出帧信息，仅用于实现/蒸馏诊断。正式质量基准必须从 E5c 按排除留出源帧的新协议重生成记忆。
- U3 的 FRONT_LEFT/RIGHT 是训练相机；与单目法比较外推只能用 SIDE_LEFT/RIGHT。已有四相机均值不代表公平的外推差距。

## NVIDIA 模型选型（官方资料核查于 2026-09-27）

| 方法 | 作用与本项目匹配 | 当前限制 |
|---|---|---|
| Cosmos-Drive-Dreams / Transfer1 Single2MultiView | 前视视频与多视角结构控制生成环视视频，接近虚拟相机目标 | 需要 HD map 或 LiDAR 控制；自建 GS 深度到训练控制格式的适配须验证。单 GPU 示例不证明 A6000 48GB 可跑。Waymo 示例要求用户自己的后训练 checkpoint，不代表发布了可直接下载的 Waymo 权重 |
| Cosmos-Transfer2.5 Auto Multiview | 统一场景控制多视角视频 | 官方实现要求 GPU 数不少于启用视角数；完整环视不适合当前单卡 |
| Cosmos-Dreams（原 OmniDreams） | 初始 RGB、文本、逐帧地图和轨迹驱动的自回归视频 | 输出视频，不直接输出持久 3D。FlashDreams 文档约 48GB 最低显存、默认单视角；原生加速路径要求 Blackwell，不能把演示速度套用到 A6000 |
| Cosmos 3 | 通用图像、视频、动作模型；Edge 有单卡起步路线 | 新版本本身不解决静态 3D 资产与空间记忆；Super 标准推理资源超出本机 |
| GEN3C | 显式 RGB-D/点云缓存投影引导；直接对应空间记忆 | 已安装，先完成集成与蒸馏诊断；不把资源受限配置称为完整论文复现 |
| Alpamayo / AlpaSim | 驾驶策略与闭环测试框架 | 不是补全后端；场景稳定后再考虑集成 |
| NuRec / 3DGRUT | 重建与渲染；官方有 COLMAP + 3DGUT 单目流程 | 可作为相机/渲染基线，不自动补齐未观测区域；当前不切换主框架 |

官方来源：

- [Single2MultiView 模型卡](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView)
- [Single2MultiView 推理与 Waymo 后训练示例](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av_single2multiview.md)
- [Transfer2.5 多视角推理](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference_auto_multiview.md)
- [Cosmos-Dreams](https://github.com/nv-tlabs/omni-dreams)、[FlashDreams 硬件与推理](https://flashdreams.org/main/models/omnidreams.html)
- [Cosmos 3 推理](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/inference.md)
- [Alpamayo 官方平台](https://github.com/NVlabs/alpamayo-recipes)
- [NuRec 单目重建](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)

## 与迭代重建论文的关系

- [ReconDreamer](https://recondreamer.github.io/)：新轨迹渲染 → DriveRestorer 修复 → 与真实图像共同更新重建，逐步扩大轨迹范围。
- [GEN3C](https://arxiv.org/abs/2503.03751)：真实/生成 RGB-D 反投影为 3D 缓存，后续相机通过缓存投影获得条件。
- [ViewCrafter](https://drexubery.github.io/ViewCrafter/)：迭代新视角生成与点云更新，最终优化 3DGS。

这些方法支持循环思路，不能据此声称本项目无标定、无 LiDAR、单 FRONT、全侧后视约束已解决。

## 本次验证：先定位模糊产生的环节

此次执行 Phase 9 修正任务 3 的局部诊断，复用已实现的任务 1/2 产物；任务 1/2 的完整画质验收仍保持未完成。一次只推进此任务，不下载/训练新生成模型，不扩大到新场景。

1. 先在 E10 的标准 35 步生成结果上，对比直接生成与 E10 最终 checkpoint 在**完全相同相机、分辨率**的渲染；检查 +30°/+60°/+90°，防止用低步数失败代表生成器上限。
2. E11 的 real-only-s8 与 memory-s8 都从同一 E5c checkpoint 出发，采用同一局部生成视角集合、3000 步、seed=0、相同 spawn/loss 设置；只在经验证的空洞处监督，真实 FRONT 继续锚定。两臂分别写新目录，保留全部旧结果。
3. 报告原始空洞、各自通过验收区以及两臂共同通过验收区的拟合误差/清晰度；掩码边缘须排除。清晰度只作辅助，噪声和漂浮物不能算改善。共同区用于避免不同验收覆盖造成选择偏差。
4. 同时保存直接生成、原模型、拟合模型、未监督偏航与横移视角。新视角没有真值生成图时只作定性检查；原路返回的固定 3D 渲染只验证可重复性，不能算独立的几何验证。
5. 记录缓存覆盖、几何支持/未知、深度、依赖版本、输入路径/来源、commit、耗时和显存。所有结果在 `results/E11/val056/` 的新子目录。

决策：直接生成已模糊则再考虑标准步数缓存/新生成器；直接生成清晰、同视角拟合丢细节则排查深度/目标冲突/拟合；仅监督视角清晰则继续几何一致性工作。只在局部通过后扩大角度、道路和场景。

实验预算：先跑现成诊断和两次 3000 步局部高斯拟合；预计总 GPU 时间约一小时内，以实际日志为准。生成网络权重保持冻结，无云资源、无额外传感器、无闭环集成。
