# 迁移到 H100

2026-10-01，用户决定把仓库迁到一台 H100（硬盘 300 GB），以便按 DECISIONS S、Z 训练修复模型。

- 代码在 GitHub：`git@github.com:yinwu33/DriveAnywhere.git` 的 `main`。
- 不入 git 的数据和权重放在 OneDrive：`onedrive:/Projects/P05_DriveAnywhere`（用户指定）。
  - 上传脚本是 `results/_logs/upload_onedrive.sh`，日志是 `results/_logs/upload_onedrive.log`，都在旧机器上。
- Python 环境不迁，在 H100 上用 `envs/` 的 uv 脚本重建：CUDA 扩展要按 H100 的架构重新编译。

## OneDrive 上有什么

| 路径 | 内容 | 大小 | 说明 |
|---|---|---|---|
| `data/dashrecon/{val056,val039,val041,val087,val094}` | 5 个开发场景的自标定、位姿与深度、掩码、融合点云、网格 | 约 4 GB | GLOMAP 每次结果不完全相同（OPEN_QUESTIONS 33），要和旧结果比就用这份，不要重跑 |
| `data/dashrecon/{mt1,mt1a,mt1b,_mt1a_only}` | MT1 两次经过的联合标定、合并场景 | 约 1.3 GB | 合并场景的图像链接已展开成文件 |
| `data/dashrecon/_undistorted/{calib-glomap,calib-glomap-mt1}` | 去畸变的 FRONT 图像 | 约 0.7 GB | |
| `data/dashrecon/{diagnostics,scene_selection}` | 位姿诊断、选段普查（D-M1） | 小 | |
| `results/{E5c,E5f,E20,E23,E29,E30}` | 单段基线：E5f（几何基线）、E30（自由视角组合模型）等 | 约 15 GB | E23 / E29 / E30 的视角屏蔽来自 `results/E20/val056/view_gate.pt` |
| `results/{MT1,MT1A,MT1app}`，之后加 `{MT1e,MT1x,MT1m}` | 多次经过的实验 | 约 2 GB，之后约 4 GB | |
| `weights/huggingface_hub/models--*` | MapAnything、Fixer、SAM 2.1、Grounding DINO、SegFormer-B5、MoGe-2 | 约 14 GB | HuggingFace 缓存格式，符号链接存成 `.rclonelink`，下载时要加 `--links` |
| `weights/gen3c_checkpoints` | GEN3C-Cosmos-7B 及其 tokenizer、T5 | 约 71 GB | 只有 Phase 9 的生成补全要用 |
| `waymo_raw/validation/` | 5 个开发场景的原始 tfrecord（带 LiDAR，评测用） | 约 4.5 GB | |
| `waymo_raw/training/` | 本地全部 798 段 training（没有 LiDAR） | 137 GB | 训练修复模型的数据来源 |
| `docs_papers/` | 参考论文 PDF | 小 | |

## H100 上的步骤

需要先准备好（用户）：
- NVIDIA 驱动、CUDA 12.1 toolkit（或其他 12.x；`envs/setup_main.sh` 要求 12.1，用别的版本要先改脚本并记录）；
- `uv`、`rclone`（配置好名为 `onedrive` 的 remote）、GitHub SSH key。

1. 代码：
   ```bash
   git clone git@github.com:yinwu33/DriveAnywhere.git && cd DriveAnywhere
   ```
2. 环境：H100 是 sm_90。
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=9.0
   bash envs/setup_main.sh; bash envs/setup_mapanything.sh; bash envs/setup_masks.sh; bash envs/setup_sfm.sh
   bash envs/setup_nksr.sh; bash envs/setup_fixer.sh; bash envs/setup_waymo.sh   # setup_gen3c.sh 只在需要生成补全时
   PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q
   ```
   - nvdiffrast 在第一次使用时编译，运行时也要带上这两个环境变量（AGENTS §13 的命令都带了 `CUDA_HOME`）。
3. 权重：
   ```bash
   rclone copy --links onedrive:/Projects/P05_DriveAnywhere/weights/huggingface_hub ~/.cache/huggingface/hub
   # 需要生成补全时：rclone copy --links onedrive:/Projects/P05_DriveAnywhere/weights/gen3c_checkpoints .venvs/src/GEN3C/checkpoints
   ```
   之后用 `HF_HUB_OFFLINE=1` 运行。lpips、Inception 这类 torch hub 小权重会自动下载。
4. 数据：原始 tfrecord 放在仓库外，再把 `data/data` 链接过去（与旧机器相同，`data/data/{training,validation}`）。
   ```bash
   rclone copy onedrive:/Projects/P05_DriveAnywhere/waymo_raw/validation /path/to/waymo_raw/validation
   ln -s /path/to/waymo_raw data/data
   rclone copy onedrive:/Projects/P05_DriveAnywhere/data/dashrecon data/dashrecon
   rclone copy onedrive:/Projects/P05_DriveAnywhere/results results
   ```
   - training 不要一次全拉：每批 20 段（约 6.5 GB），处理完删掉原始文件。
   - 预处理命令同 AGENTS §13.2，换成 `--data_root data/data/training --split training --waymo_file_list data/waymo_train_list.txt`（scene_idx 是这个列表里的序号，例如 mt1a = 368）。
5. 验证迁移：在 H100 上重新评测 E30。
   ```bash
   PYTHONPATH=. .venvs/main/bin/python scripts/eval_front_heldout.py --log_dir results/E30/val056/model --example_frames 50 100 150
   ```
   - 要能复现 FRONT 留出 32.54 / 0.078（docs/EXPERIMENTS.md E30）。
   - 这一步需要 `data/waymo/processed/validation/056`：先预处理 val056（AGENTS §13.2，`--scene_ids 56`）。

## H100 上的硬盘预算（300 GB）

| 项目 | 大小 |
|---|---|
| Python 环境（不含 gen3c / wan / recam） | 约 35 GB |
| 权重（不含 GEN3C） | 约 14 GB |
| 开发场景：原始数据、预处理、dashrecon 产物、基线结果 | 约 30 GB |
| 修复模型的训练数据：100 段，每段只留训练对和 checkpoint，约 0.7 GB | 约 70–100 GB |
| training 原始数据的下载暂存 | 约 10 GB |
| 修复模型的 checkpoint | 约 20 GB |
| **合计** | **约 180–210 GB** |

- 加上 GEN3C（71 GB + 7 GB 环境）就接近上限，需要时再拉。
- 每段训练数据的中间产物（深度约 270 MB、中间 checkpoint 约 400 MB）训练完就删。
