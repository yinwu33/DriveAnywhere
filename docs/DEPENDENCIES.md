# DEPENDENCIES.md — 依赖确认（AGENTS.md Phase 0 任务 4）

> 环境一律用 **uv** 建立（DECISIONS D 节）。安装脚本见 `envs/`，venv 位于 `.venvs/<name>`（不入 git）。
> 硬件：1× RTX A6000 48GB，驱动 555.42（CUDA 12.5），系统 CUDA toolkit 12.1（`/usr/local/cuda-12.1`）。
> 下表中"已验证"指在本机实际运行过；"待确认"指该依赖所属的 Phase 尚未开始。

## 环境一览

| venv | 用途 | 建立方式 | 关键版本（已验证） |
|---|---|---|---|
| `main` | drivestudio 训练与渲染（Phase 6） | `CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6 bash envs/setup_main.sh` | Python 3.10.16，torch 2.4.1+cu121，gsplat 1.3.0，pytorch3d 0.7.8，nvdiffrast 0.3.3，numpy 1.23.1，open3d 0.16.0，torchmetrics 0.10.3，setuptools 80.10.2；Phase 8 另装 diffusers 0.31.0，transformers 4.46.3，accelerate 1.1.1，huggingface_hub 0.26.5（已写入 `envs/main-requirements.txt`） |
| `waymo` | Waymo 预处理、选场景用的统计 | `bash envs/setup_waymo.sh` | Python 3.10.16，tensorflow 2.11.0，waymo-open-dataset-tf-2-11-0 1.6.0，numpy 1.21.5 |
| `mapanything` | Phase 3 位姿与点图 | `bash envs/setup_mapanything.sh` | Python 3.12.12，torch 2.5.1+cu121，mapanything 1.1.4（commit `3d10cf7`），uniception 0.1.7，numpy 2.5.2 |
| `masks` | Phase 4 掩码 | `bash envs/setup_masks.sh` | Python 3.12.12，torch 2.6.0+cu124，transformers 5.17.0，accelerate 1.15.0，scipy |
| `nksr` | Phase 7 网格 | `CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6 bash envs/setup_nksr.sh` | Python 3.10.16，torch 2.4.1+cu121，torch-scatter 2.1.2（pt24cu121），nksr 1.0.3（源码 commit `e403368`，编译约 15 min），python-pycg，Open3D 0.19.0 |

## 各依赖

| 用途 | 依赖 | 仓库 / 许可 | 安装 | 显存与输入限制 | 状态 |
|---|---|---|---|---|---|
| 基础框架 | drivestudio（本仓库上游） | github.com/ziyc/drivestudio / MIT | `main` venv | — | 已验证：所有模块可 import |
| 光栅化 | gsplat 1.3.0 | nerfstudio-project/gsplat / Apache-2.0 | 从 git tag 源码编译，需要 `--no-build-isolation` | — | 已验证：`rasterization(..., render_mode="RGB+ED")` 输出 4 通道。代码依赖 `gsplat.cuda_legacy`，只能用 1.3.0 |
| 其他 CUDA 扩展 | pytorch3d V0.7.8、nvdiffrast v0.3.3 | Meta BSD / NVIDIA Source Code License | 源码编译 | — | 已验证。**nvdiffrast 首次使用时 JIT 编译**：运行时 PATH 里要有 venv 的 `bin/`（ninja）和 `$CUDA_HOME/bin` |
| 位姿与点图 | MapAnything | facebookresearch/map-anything / 代码 Apache-2.0；权重 `facebook/map-anything` 为 CC-BY-NC-4.0，`facebook/map-anything-apache` 为 Apache-2.0 | `mapanything` venv | 输入缩放到 518 宽的固定网格：1920×1280 先缩放到 518×345，再中心裁到 518×336。**197–199 帧一次推理：峰值 31.5 GB，约 43 s** | 已验证（Phase 3），选用哪个权重见 DECISIONS F 节 |
| 预处理 | waymo-open-dataset-tf-2-11-0 1.6.0 | waymo-research / Apache-2.0（数据另有条款） | `waymo` venv | — | 已验证 |
| 动态分割 | Grounded-SAM-2：Grounding DINO base + SAM 2.1 large | 通过 HF transformers 使用；Grounding DINO 为 Apache-2.0，SAM 2.1 为 Apache-2.0 | `masks` venv | 检测输入 800×1200（全分辨率时检测失效）；实测每帧约 0.85 s（含 SegFormer），峰值约 4–7 GB | 已验证（Phase 4），见 DECISIONS G |
| 天空/路面 | SegFormer-B5 Cityscapes（HF） | nvidia/segformer-b5-finetuned-cityscapes-1024-1024（NVIDIA SegFormer license，仅限非商业用途） | `masks` venv | 输入 1024×1536，logits 上采样回原图 | 已验证（Phase 4）；类别 id 0 = road、10 = sky，加载时断言 |
| 表面重建 | NKSR | nv-tlabs/NKSR / NVIDIA Source Code License（仅限非商业用途），模型 CC-BY-SA-4.0 | `nksr` venv（上游改为源码编译，官方配方是 conda + CUDA 12.8；这里用 uv 针对系统 CUDA 12.1 编译） | kitchen-sink 权重（`ks.pth`，54.9 MB，从 HF `heiwang1997/nksr-checkpoints` 下载）的体素为 0.1（米）；val056 的 300 万点输入耗时 10 s，峰值 2.8 GB | 已验证（Phase 7），见 DECISIONS I |
| 生成式细化 | SDXL base 1.0 + SDXL depth ControlNet + fp16 VAE | `stabilityai/stable-diffusion-xl-base-1.0`（CreativeML Open RAIL++-M）、`diffusers/controlnet-depth-sdxl-1.0`（OpenRAIL++）、`madebyollin/sdxl-vae-fp16-fix`（MIT）；只用现成权重，不微调 | `main` venv；权重用 `envs/download_gen_weights.sh` 下载到 HF 缓存（9.6 GB，fp16），之后以 `HF_HUB_OFFLINE=1` 运行 | 生成分辨率 1248×832，DDIM 反演 5 步 + 去噪 5 步：与两个训练任务共享 GPU 时每张约 4 s；加上 drivestudio 训练器，峰值约 14 GB（E5+pp）/ 17 GB（E6） | 已验证（Phase 8），见 DECISIONS K |
| 点云处理 | Open3D 0.16.0 | isl-org/Open3D / MIT | `main` venv | — | 已安装 |
| 位姿评测 | evo | MichaelGrupp/evo / GPL-3.0 | — | — | 待确认（Phase 1，延后） |
| 图像评测 | torchmetrics 0.10.3、lpips 0.1.4 | Apache-2.0 / BSD | `main` venv | — | 已安装 |

## 环境相关的坑（已处理）
- 上游 `requirements.txt` 固定了 `xformers==0.0.18`，它会把 torch 锁在 2.0。drivestudio 代码没有 import xformers 或 timm，所以主环境去掉了这两个包（见 `envs/main-requirements.txt`）。
- torchmetrics 0.10.3 会 import `pkg_resources`，而 setuptools ≥81 已经移除了它，因此固定 `setuptools<81`。
- uv 通过 git 拉取时偶尔会遇到代理报错 `Proxy CONNECT aborted`，重试即可。
- transformers 5.x 在 torch < 2.6 时拒绝读取 `.bin` checkpoint（CVE-2025-32434），所以 `masks` venv 用 torch 2.6.0 cu124。驱动 555 支持 CUDA 12.5，可以运行。
- NKSR 在模块级 import `pycg.vis`，而它需要 Open3D，所以 `nksr` venv 额外装了 Open3D 0.19.0。
