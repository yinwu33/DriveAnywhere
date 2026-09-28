# E11 / E12 左右侧视失败诊断

日期：2026-09-28。Phase 9 修正任务 3，接用户反馈：E11、E12 drive / sweep 在左右 90° 均很差。自由视角质量验收失败，不能以此前局部改善或通过区 PSNR 代替用户所要求的场景质量。

## 已核对的覆盖范围

| 模型 | 本次局部生成/拟合 | 与左右 90° 的关系 |
|---|---|---|
| E11 real-only / memory | 从 E5c 开始，历史 +60° 轨迹的 25 个视角；原始 frame 80–128 | 没有拟合左右 90°；不是继承 E10 全部轮次的模型 |
| E12 drive / sweep | 从 E5c 开始，frame 96–112、yaw 0→+60°；121 个生成相机、25 个拟合相机 | 没有负偏航或左右 90° 的生成/拟合；只有一次局部更新 |

这些是诊断实验，不是完成侧面补全的版本。交互查看器允许整个序列和 ±180°，因此可以看到大量未更新区域。此前应把覆盖范围和完整质量失败讲得更明确；局部画面的认可不能被描述为用户认可完整 E11 场景。此前具体选择的模型/视角没有记录，不能推定。

## 可视化入口身份核对

用户随后提到 8090，并追问先前觉得好的版本是否为 E11。2026-09-28 只读核对进程、两次 8081 启动记录、当前 checkpoint、查看器代码：

| 端口 | 当前内容 | 与 E11 的关系 |
|---|---|---|
| 8090 | HUGSIM 发布示例 `hugsim_release/scenes/waymo/100613054308_0_200` | 另一场景和重建流程；不是本项目 val056/E11 |
| 8080 | E5c / E8（gen）/ E10（gen）/ U3 oracle comparison | 历史模型对照，未加载 E11 |
| 8081 | E5c / E11 real-only（gen）/ E11 memory（gen）/ E12 drive（gen）/ E12 sweep（gen） | 当时 E11 对照入口和现在相同 |

原 8081 启动记录 `e11_viewer_8081.json` 与当前 `e12_viewer_8081.json` 的 E11 checkpoint 路径完全相同；checkpoint 修改时间早于原查看器启动，当前 SHA256 与本次侧视复查保存值一致。real-only / memory 的 SHA256 分别为 `c3a5aabd6e94d919da55257cb639c2ecd601b68228b73f77e79f9b2fe5f64e4c` / `52562aa5034a76eb4a3d97de639be4d9058dbd9a02ae131d9c2208d634b735ef`。原 `e04d91c` 至当前的查看器代码只增加标签参数及显示名字，渲染和相机摆放没有变化；没有 E12 覆盖或续训这两个 E11 checkpoint。

两次 8081 的默认模型都是 E5c，默认 frame=0、yaw=0。重启后 GUI 回到默认设置，可能与之前手动选的模型、位置、角度不同。历史启动日志没有保存用户选择，不能断言先前认可的是哪一个 E11 分支，甚至不能排除当时在看 E5c。若用户指 8090，则当前实际入口为 HUGSIM；不能把它的外观当作 E11 的效果。

后方：查看器允许旋转/移动，只代表可以渲染该方向。E8/E10 的磁盘生成轨迹只到两侧 ±90°，E11/E12 的范围更窄；本项目这些实验均没有专门完成 180° 后视补全。回看时可能看到较早 FRONT 已观测的道路/区域，不能据此推定隐藏背面完成。HUGSIM 查看器也支持回头查看，但本次未验收其后方覆盖质量，未使用其模型、相机或数据进入本项目重建。

覆盖不足不能解释全部失败：已覆盖的 +60° 也有模糊、彩色漂浮片和破碎结构。E12 drive / sweep 只有原始空洞的 8.73% / 3.11% 通过生成 RGB-D 自洽检查，大量区域没有回写。通过检查不等于真实几何正确；旧的不透明失真高斯也不会因候选被拒收而自动消失。当前证据没有证明只需多采几个角度、只换生成器或只增加高斯就能解决。

## 本次固定相机检查

已在 frame 96 / 104 / 112、yaw −90 / −60 / 0 / +60 / +90°、right=0 渲染 E5c 与 E11/E12 四个模型。所有模型共享 E5c 估计位姿加 CamPose 修正的相机，保存 1280×704 RGB、深度、相机矩阵和五列对比图。只做失败范围检查，没有 GT、没有新生成或训练；不覆盖历史结果。

复查确认失败：三个位置的 −90° 几乎沿用 E5c 的大片条状/片状漂浮结构；+90° 在 E11/E12 的局部植被区域有变化，但仍被彩色模糊片、破碎地面和失真几何占据。+60° 也有相同问题。0° 保留较完整的道路和前方建筑，说明前向改善不能代表两侧场景建成。以上为目测，不是真值准确率，也没有定位各因素的责任比例。

五列依次为 E5c、E11 real-only（gen）、E11 memory（gen）、E12 drive（gen）、E12 sweep（gen）。

![frame 104 左侧 90°](../results/_diagnostics/E11_E12_sideview/val056/audit_20260928/005_compare.png)

![frame 104 右侧 90°](../results/_diagnostics/E11_E12_sideview/val056/audit_20260928/009_compare.png)

![frame 104 已拟合的右侧 60°](../results/_diagnostics/E11_E12_sideview/val056/audit_20260928/008_compare.png)

[全部 15 相机对照视频](../results/_diagnostics/E11_E12_sideview/val056/audit_20260928/comparison.mp4)。索引 000–004、005–009、010–014 分别对应 frame 96、104、112，每组依次为 −90/−60/0/+60/+90°。输出目录：`results/_diagnostics/E11_E12_sideview/val056/audit_20260928/`。

干净独立 worktree，代码 `39796c29c877aa0f2bb5323ead3b02d3f346c4f1`。成功运行 65.99 秒，Torch allocated 峰值 3.88 GiB，Torch 2.4.1+cu121、gsplat 1.3.0。75 张原分辨率 RGB、75 个有限值深度数组和 15 张五列图均核对，保存五个输入 checkpoint 的 SHA256。相机矩阵、全部模型配置快照、meta 和 artifact_audit.json 一并保留；此项无 GT 指标，不替代 Phase 1 评测。

首次启动时未把主 venv 的 bin 放入 PATH，nvdiffrast 无法找到 Ninja，未完成模型渲染。失败输出和日志独立保留为 `audit_20260928_environment_failed/` 与 `results/_logs/e11-e12-sideview-audit-environment-failed.log`；同一提交显式补齐 PATH 后重跑，没有回退到其他渲染器。已有 8080/8081 查看器及其他项目 GPU 进程保持运行。

```bash
PATH="$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH" CUDA_HOME=/usr/local/cuda-12.1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/main/bin/python scripts/render_sweep_comparison.py \
  --init_log_dir results/E5c/val056 \
  --log_dirs results/E11/val056/probe_20260928_seeded/real_only_s8_fit \
    results/E11/val056/probe_20260928_seeded/memory_s8_fit \
    results/E12/val056/probe_20260928/drive_fit \
    results/E12/val056/probe_20260928/sweep_fit \
  --labels E5c E11_real_only_gen E11_memory_gen E12_drive_gen E12_sweep_gen \
  --frames 96 104 112 --yaws -90 -60 0 60 90 --rights 0 \
  --out_dir results/_diagnostics/E11_E12_sideview/val056/audit_20260928
```

## 下一项应回答的问题

优先用固定、支持充分的单个/少量生成视角检查“目标图能否进入持久 3D”：相同 RGB-D 与预算下比较现有初始化和有限分裂/更密初始化，同时保留真实 FRONT 锚定。直接生成、同相机 3D、未参与拟合的平移视角分别验收；同时报告全图和拒收区，不能只看通过区 PSNR。单视角能拟合而多视角失败，再检验深度、遮挡和目标冲突；若单视角就失败，先修拟合。这是待执行的对照，本次未证明增加表示密度就是修复。

清理旧失真几何是待验证变量：必须先确认它不受真实观测支持，不能按“看起来差”删除真实区域。充分记忆与联合深度仍未完成。局部通过后，才逐步扩到两侧 90°，每一轮都检查同一空间在相邻位置的稳定性。完整自由视角、道路/碰撞几何和可测试资产继续保持未验收状态。
