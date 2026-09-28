# 实验登记表

长期追踪所有实验和诊断（用户要求，2026-09-28）。DECISIONS.md 记录为什么这样做；本表是查任何实验编号的唯一入口。

## 约定

- **启动前**：在"条目"里写方案，包括假设、相对基线改了什么、命令或配置、代码 commit、输出目录、预期、判定标准，然后提交。
- **结束后**：补结果，包括关键数字、图的路径、结论、下一步，更新总表状态，然后提交。失败或中止的实验也要记。
- 编号：正式实验用 `E<n>`（`E5f` 这类后缀表示同一实验的变体）；只读诊断用 `D-<字母><n>`；oracle 上界用 `U<n>`。
- 默认场景：val056（只在一个场景上迭代，扩到 5 个场景要用户同意）。
- 指标口径：
  - "FRONT 留出"是 `scripts/eval_front_heldout.py` 的 19 个留出帧，整图 PSNR / LPIPS。
  - "侧视未观测"是 `scripts/eval_cross_camera.py --cams 1 2 3 4 --region_ref results/E5c/val056`，只在 E5c 定义的同一组未观测像素上算 PSNR / LPIPS。只有 SIDE_LEFT / SIDE_RIGHT 可以和 U3 比。
  - "验收通过率"是空洞像素中通过 `validate_generated_views.py` 的比例。它只说明生成的 RGB-D 自洽，不说明隐藏内容正确。

## 总表

| 编号 | 日期 | 场景 | 内容 | 性质 | 状态 | 关键结果 | 详情 |
|---|---|---|---|---|---|---|---|
| E3 / E4 / E5 | 09-24–25 | 5 场景 | MapAnything 位姿 + 估计点图 / 融合清理 / NKSR 网格 | 非 oracle | 完成 | E5 比 E4 留出帧均值高 0.26 dB | DECISIONS J |
| E6 / E5pp | 09-25 | 5 场景 | SDXL + ControlNet GGDS 蒸馏 / 逐帧后处理 | 含生成 | 完成 | 留出帧 E6 −0.19 dB，E5pp −1.1 dB | DECISIONS K |
| E4c / E5c | 09-25 | 5 场景 | GLOMAP 自标定相机（D13） | 非 oracle | 完成，**当前基线** | 5 场景留出帧均值 +3.8 dB；val056 FRONT 留出 31.39 / 0.101，侧视未观测 16.56 / 0.356 | DECISIONS L |
| E7 / E5cfx | 09-25 | 5 场景 | NVIDIA Fixer 渐进蒸馏 / 后处理 | 含生成 | 完成 | E7 ≈ E5c；E5cfx 留出 LPIPS 0.210 → 0.183 | DECISIONS M |
| E8 | 09-25–26 | val056 | Wan VACE 转向补全，7 轮，3D 记忆 | 含生成 | 完成 | 侧视未观测 17.12 / 0.349；FRONT 留出 32.35 / 0.092 | DECISIONS N |
| E9 | 09-26 | val056 | LSD-3D 式 GGDS（SDXL + 深度 ControlNet） | 含生成 | 暂停 | 不锚定真实帧时发散；锚定后偏色 | DECISIONS O |
| U3 / U3long / U1 | 09-26 | val056 | StreetGS，3 相机 / FRONT，真值位姿 + LiDAR | **oracle** | 完成 | U3 SIDE 未观测 19.60 / 0.390、20.40 / 0.321；U3 FRONT 留出 26.57；U1 FRONT 留出 30.46 | DECISIONS Q |
| E10 | 09-27 | val056 | GEN3C v2（真实帧 warp 缓存），±30 / 60 / 90°，6 轮 | 含生成 | 完成 | 侧视未观测 16.81 / 0.340；FRONT 留出 31.70 / 0.103；缓存可能含留出帧（S 节） | DECISIONS R |
| E11 | 09-27–28 | val056 | GEN3C 跨轮记忆缓存 + 几何验收，局部 +60° 对照 | 含生成 | 诊断完成，质量失败 | 记忆覆盖 0.46%，有无记忆无明显差别 | DECISIONS S、T、E11_PROBE_RESULTS.md |
| E12 | 09-28 | val056 | 行驶转向 vs 三位置原地扫视，等预算 | 含生成 | 诊断完成，质量失败 | 验收通过率 8.73% / 3.11%，扫视无优势 | DECISIONS U、E12_SWEEP_RESULTS.md |
| D-U2 | 09-28 | val056 | E11 / E12 固定相机左右 90° 复查 | 只读 | 完成 | 左右 90° 全部失败 | SIDEVIEW_FAILURE_ANALYSIS.md |
| E13 | 09-28 | val056 | 3 位置 × 8 方向虚拟环视，两轮 ±180° | 含生成 | 完成，质量失败 | 验收通过率 0.11% / 0.54%，3D 几乎未变 | DECISIONS V、E13_SURROUND.md |
| D-A1 / D-A2 | 09-28 | val056 | 侧视垃圾来源：support、尺寸、离网格距离 | 只读 | 完成 | 垃圾 = FRONT 依赖的大尺寸、离网格高斯；剪掉最大轴 > 1 的高斯使 FRONT −1.9 dB | DECISIONS W1，下文 |
| D-B1 | 09-28 | val056 | 生成帧深度：MoGe 逐帧 vs MapAnything 联合 | 只读 | 完成 | 验收通过率 E13 0.11% → 3.6%，E12 8.7% → 20.2% | DECISIONS W1，下文 |
| D-A3 | 09-28 | val056 | 按观测视角锥屏蔽高斯 | 只读 | 计划中 | — | 下文 |
| E5f | 09-28 | val056 | E5c + flatten + 大尺寸惩罚（参考 MTGS） | 非 oracle | 计划中 | — | 下文 |
| E14 | 09-28 | val056 | 视角屏蔽 + MapAnything 联合深度 + 带平移轨迹 + 每轮外观残差 | 含生成 | 待 D-A3 后细化 | — | 下文 |

## 条目

### D-A1 / D-A2：侧视垃圾的来源

- **方案**（DECISIONS W）：
  - 逐个高斯统计：FRONT 训练视角中受约束像素上的混合权重（support）、尺寸、离 NKSR 网格的距离、needle 比例、DC 饱和度。
  - 在内存里改参数，同一个 checkpoint 渲染固定相机（frame 96 / 104 / 112 × 偏航 −90 / −60 / 0 / 60 / 90 / 180）和 FRONT 留出帧。
  - 脚本：`scripts/diagnose_floaters.py`，commit `efc29cf`（A1）/ `b864c7b`（A2）。
  - 输出：`results/_diagnostics/floaters/val056/E5c_a`、`E5c_b`。
- **结果**：见 DECISIONS W1。
  - 去掉 support < 1 的 18.8 万个高斯只去掉 1.9% 的侧视权重；只用 DC 颜色也去不掉彩色块。
  - 侧视权重集中在大尺寸、离网格远的高斯上（1.2% 的高斯贡献 80%），FRONT 离不开它们：最大轴 > 1 剪掉后 FRONT 留出 29.45，离网格 > 1 剪掉后 20.55。剪掉后侧视露出来的几乎全是空的。
- **结论**：侧面基本没被重建出来。FRONT 用一层只在前向成立的大高斯拟合远处和边缘；现有空洞判断不把这些像素算作空洞。

### D-B1：生成帧的联合深度

- **方案**：同一组生成帧、同一套验收参数，只换深度来源（MoGe / MapAnything × 全局 / 逐空洞尺度）。脚本 `scripts/run_depth_probe.py`，commit `efc29cf`、`7bcd38a`、`042554d`；输出 `results/_diagnostics/depth_probe/val056/{E13_r0,E13_r0_ma,E12_drive}`。
- **结果**：MapAnything 联合深度（全局尺度）让 E13 第 1 轮的验收通过率从 0.11% 升到 3.57%，E12 drive 从 8.73% 升到 20.23%。MapAnything 基本遵守给定位姿（相对第 0 帧旋转误差中位数 0.5–0.6°）。仍有 53–64% 的空洞像素没有证据；E13 原地转动时，67–112° 的侧视只有 0.2% 通过。
- **结论**：生成帧深度改用 MapAnything 联合推理；侧视轨迹要带平移。

### D-A3：按观测视角锥屏蔽高斯（计划）

- **假设**：D-A2 的垃圾高斯只在 FRONT 的观测方向上成立。如果渲染新视角时，把视线偏离其观测方向太多的高斯淡出，侧视会露出真正的空洞（可以交给生成），而 FRONT 不受影响。
- **方法**：
  - 对每个高斯，用两遍训练视角统计观测锥：第一遍求混合权重加权的平均观测方向 m；第二遍求在该视角权重 ≥ 0.5 像素的训练视角中，观测方向与 m 的最大夹角 θ。没有任何这种视角的高斯记为"未观测"。
  - 新视角下，高斯的偏离角 δ = max(0, ∠(视线, m) − θ)。不透明度乘以 g = clamp(1 − (δ − margin) / 15°, 0, 1)。未观测高斯 g = 0。
  - 变体：只屏蔽离网格 > 0.3 的高斯（网格附近的表面，如路面，从哪个方向看都应成立），或屏蔽全部；margin 取 15° / 30°。
- **判定**：
  - FRONT 留出帧下降不超过 0.1 dB；
  - 第 104 帧 ±90° 的彩色块和针状碎片消失，变成空洞；
  - 路面和网格附近的表面保留。
- **输出**：`results/_diagnostics/floaters/val056/E5c_gate`。

### E5f：E5c + MTGS 式几何正则（计划）

- **假设**：D-A2 的垃圾是"大尺寸、半透明、靠叠加混色"的高斯。MTGS 的 flatten 正则和惩罚大尺寸，能迫使 FRONT 用贴表面的小面片拟合，侧视更干净，FRONT 代价可接受。
- **改动**：只在 E5c 的配置上加两项上游已有的正则（`models/gaussians/vanilla.py` `compute_reg_loss`）：
  - `model.Background.reg.flatten.w=1.0`（最短轴 L1；MTGS λ_flatten = 1.0，定义不同，未调）；
  - `model.Background.reg.max_s_square_reg.w=0.05`（最大轴平方；上游 OmniRe 对 SMPL 节点用 0.05，未调）。
  - 其余与 E5c 相同，从头训练，种子相同。
- **评测**：FRONT 留出、侧视未观测、D-A2 同一套固定相机与侧视权重分布（大尺寸 / 离网格高斯的占比）。
- **判定**：FRONT 留出下降 ≤ 0.3 dB，且侧视彩色块明显减少（目测），侧视未观测 LPIPS 不变差。
- **输出**：`results/E5f/val056/`。

### E14：视角屏蔽 + 联合深度的转向补全（待 D-A3 后细化）

- 组成：D-A3 的视角屏蔽用于渲染和空洞判断；生成帧深度用 MapAnything 联合推理（全局尺度）；侧视轨迹带平移（drive 方式）；每一轮生成帧当作一次"虚拟经过"，带独立的外观残差（SH 高阶 + 仿射），几何和基色与真实 FRONT 共享（参考 MTGS）。
- 具体方案在 D-A3 结果出来后写入。
