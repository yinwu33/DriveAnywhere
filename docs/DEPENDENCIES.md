# DEPENDENCIES.md — 依赖确认（AGENTS.md Phase 0 任务 4）

> 环境一律用 **uv** 建立（DECISIONS D 节）。安装脚本见 `envs/`，venv 位于 `.venvs/<name>`（不入 git）。
> 硬件：1× RTX A6000 48GB，驱动 555.42（CUDA 12.5），系统 CUDA toolkit 12.1（`/usr/local/cuda-12.1`）。
> 下表中"已验证"指在本机实际运行过；"待确认"指该依赖所属的 Phase 尚未开始。

## 环境一览

| venv | 用途 | 建立方式 | 关键版本（已验证） |
|---|---|---|---|
| `main` | drivestudio 训练与渲染（Phase 6） | `CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6 bash envs/setup_main.sh` | Python 3.10.16，torch 2.4.1+cu121，gsplat 1.3.0，pytorch3d 0.7.8，nvdiffrast 0.3.3，numpy 1.23.1，open3d 0.16.0，torchmetrics 0.10.3，setuptools 80.10.2 |
| `waymo` | Waymo 预处理、选场景用的统计 | `bash envs/setup_waymo.sh` | Python 3.10.16，tensorflow 2.11.0，waymo-open-dataset-tf-2-11-0 1.6.0，numpy 1.21.5 |
| `mapanything` | Phase 3 位姿与点图 | `bash envs/setup_mapanything.sh` | Python 3.12.12，torch 2.5.1+cu121，mapanything 1.1.4（commit `3d10cf7`），uniception 0.1.7，numpy 2.5.2 |

## 各依赖

| 用途 | 依赖 | 仓库 / 许可 | 安装 | 显存与输入限制 | 状态 |
|---|---|---|---|---|---|
| 基础框架 | drivestudio（本仓库上游） | github.com/ziyc/drivestudio / MIT | `main` venv | — | 已验证：所有模块可 import |
| 光栅化 | gsplat 1.3.0 | nerfstudio-project/gsplat / Apache-2.0 | 从 git tag 源码编译，需要 `--no-build-isolation` | — | 已验证：`rasterization(..., render_mode="RGB+ED")` 输出 4 通道。代码依赖 `gsplat.cuda_legacy`，只能用 1.3.0 |
| 其他 CUDA 扩展 | pytorch3d V0.7.8、nvdiffrast v0.3.3 | Meta BSD / NVIDIA Source Code License | 源码编译 | — | 已验证。**nvdiffrast 首次使用时 JIT 编译**：运行时 PATH 里要有 venv 的 `bin/`（ninja）和 `$CUDA_HOME/bin` |
| 位姿与点图 | MapAnything | facebookresearch/map-anything / 代码 Apache-2.0；权重 `facebook/map-anything` 为 CC-BY-NC-4.0，`facebook/map-anything-apache` 为 Apache-2.0 | `mapanything` venv | 输入缩放到 518 宽的固定网格：1920×1280 先缩放到 518×345，再中心裁到 518×336。**197–199 帧一次推理：峰值 31.5 GB，约 43 s** | 已验证（Phase 3），选用哪个权重见 DECISIONS F 节 |
| 预处理 | waymo-open-dataset-tf-2-11-0 1.6.0 | waymo-research / Apache-2.0（数据另有条款） | `waymo` venv | — | 已验证 |
| 动态分割 | Grounded-SAM-2 | IDEA-Research/Grounded-SAM-2 | — | — | 待确认（Phase 4）。grounding-dino-base/tiny 已在本地 HF 缓存 |
| 天空/路面 | SegFormer-B5 Cityscapes（HF） | nvidia/segformer-b5-finetuned-cityscapes-1024-1024 | — | — | 待确认（Phase 4） |
| 表面重建 | NKSR | nv-tlabs/NKSR / NVIDIA Source Code License，模型 CC-BY-SA-4.0 | — | — | 待确认（Phase 7）。2025-09 起支持 torch 2.7 / CUDA 12.8 |
| 点云处理 | Open3D 0.16.0 | isl-org/Open3D / MIT | `main` venv | — | 已安装 |
| 位姿评测 | evo | MichaelGrupp/evo / GPL-3.0 | — | — | 待确认（Phase 1，延后） |
| 图像评测 | torchmetrics 0.10.3、lpips 0.1.4 | Apache-2.0 / BSD | `main` venv | — | 已安装 |

## 环境相关的坑（已处理）
- 上游 `requirements.txt` 固定了 `xformers==0.0.18`，它会把 torch 锁在 2.0。drivestudio 代码没有 import xformers 或 timm，所以主环境去掉了这两个包（见 `envs/main-requirements.txt`）。
- torchmetrics 0.10.3 会 import `pkg_resources`，而 setuptools ≥81 已经移除了它，因此固定 `setuptools<81`。
- uv 通过 git 拉取时偶尔会遇到代理报错 `Proxy CONNECT aborted`，重试即可。
