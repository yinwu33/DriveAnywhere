"""Summarize E12 sampling, geometric evidence, fitting and common-camera images."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pool(rows: list[dict]) -> dict:
    """Pool pixel errors before PSNR; undefined detail remains unknown."""
    valid = [r for r in rows if r["pixels"] > 0]
    pixels = sum(r["pixels"] for r in valid)
    if not pixels:
        return {"pixels": 0, "psnr": None, "mae": None, "detail_variance_ratio": None, "laplacian_correlation": None}
    values = {key: sum(r[key] * r["pixels"] for r in valid) / pixels
              for key in ["mse", "mae", "target_laplacian_variance", "render_laplacian_variance"]}
    correlated = [r for r in valid if r["laplacian_correlation"] is not None]
    correlation = None
    if correlated:
        correlation = sum(r["laplacian_correlation"] * r["pixels"] for r in correlated) / sum(r["pixels"] for r in correlated)
    return {"pixels": pixels, "psnr": -10 * math.log10(max(values["mse"], 1e-12)), "mae": values["mae"],
            "detail_variance_ratio": values["render_laplacian_variance"] / values["target_laplacian_variance"] if values["target_laplacian_variance"] > 0 else None,
            "laplacian_correlation": correlation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.probe_dir
    state = json.loads((root / "run_status.json").read_text())
    if not state["completed"]:
        raise ValueError("probe stages have not completed")
    result = {"kind": "sampling and synthetic-target diagnostic; no hidden-view ground truth",
              "dashrecon_commit": state["dashrecon_commit"], "runtime_s": state["runtime_s"], "arms": {}}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, mode in zip(axes, ["drive", "sweep"]):
        plan = json.loads((root / f"{mode}_trajectory.json").read_text())["views"]
        ax.plot([v["pose_frame"] for v in plan], [v["yaw"] for v in plan], ".-", markersize=3)
        ax.set_title(f"{mode}: 121 camera samples")
        ax.set_xlabel("Position along FRONT trajectory (frame coordinate)")
        ax.set_ylabel("Relative yaw (degrees)")
        ax.grid(alpha=.25)
        validation = json.loads((root / f"{mode}_validation" / "validation.json").read_text())
        cams = json.loads((root / f"{mode}_views" / "cams.json").read_text())
        cache = json.loads((root / f"{mode}_views" / "cache.json").read_text())
        arm = {"geometric_accepted_hole_fraction": validation["accepted_fraction"],
               "geometric_unknown_hole_fraction": validation["unknown_fraction"],
               "input_hole_fraction": sum(cams["hole_frac"]) / len(cams["hole_frac"]),
               "cache_buffer_coverage": cache["coverage"],
               "generation": json.loads((root / f"{mode}_views" / "fill.json").read_text()),
               "front": json.loads((root / f"{mode}_fit" / "metrics.json").read_text())["test"],
               "fit_meta": json.loads((root / f"{mode}_fit" / "meta.json").read_text()), "fitting": {}}
        for phase in ["before", "after"]:
            metrics = json.loads((root / f"{mode}_{phase}" / "metrics.json").read_text())
            arm["fitting"][phase] = {region: pool([r["regions"][region] for r in metrics["rows"]])
                                      for region in ["hole", "accepted"]}
        result["arms"][mode] = arm
    figure.tight_layout()
    figure.savefig(root / "sampling.png", dpi=160)
    plt.close(figure)
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    (root / "metrics.json").write_text(json.dumps({"diagnostic": True, "uses_ground_truth": False,
        "arms": {mode: {key: arm[key] for key in ["geometric_accepted_hole_fraction", "geometric_unknown_hole_fraction", "input_hole_fraction", "front", "fitting"]}
                 for mode, arm in result["arms"].items()}}, indent=2))
    lines = ["# E12 行驶与关键位置扫视对照", "",
             "两组从 E5c 重生成输入；相同 GEN3C 权重、121 帧、35 步、seed=1；各用 25 个目标视角优化 3000 步，seed=0。",
             "真实 FRONT 和监督池索引的随机流相同，采样位置与角度是实验变量。未使用旧 E10/E11 生成记忆或其他相机输入。",
             "", "![采样位置与角度](sampling.png)", "",
             "扫视在 frame 96、104、112 做 0→60→0→60° 扫描，三个扫视段分别固定相机中心，中间用短平移连接。",
             "几何验收要求相机中心相隔至少 1 个场景单位、各参考中心也相隔至少 1 单位，每个像素至少有 0.5° 视差；同中心纯旋转不提供深度支持。",
             "这是一套新增诊断，不修改旧实验的评测协议；通过比例不能与旧 E11 的邻帧投票直接比较，也不代表隐藏内容正确率。",
             "", "| Arm | Input holes | Accepted within holes | Unknown within holes | FRONT PSNR | FRONT LPIPS |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, arm in result["arms"].items():
        front = arm["front"]
        lines.append(f"| {mode} | {arm['input_hole_fraction']:.3%} | {arm['geometric_accepted_hole_fraction']:.3%} | {arm['geometric_unknown_hole_fraction']:.3%} | {front['image_metrics/test/psnr']:.3f} | {front['image_metrics/test/lpips']:.4f} |")
    lines += ["", "## 对各自生成目标的拟合", "", "PSNR/细节指标对照各自生成图，目标和通过区不同，不能用两组绝对 PSNR 判断谁更接近真实隐藏场景。", "",
              "| Arm | Stage | Accepted pixels | PSNR | Detail variance ratio | Detail correlation |",
              "|---|---|---:|---:|---:|---:|"]
    for mode, arm in result["arms"].items():
        for phase in ["before", "after"]:
            row = arm["fitting"][phase]["accepted"]
            numbers = ["unknown" if row[key] is None else f"{row[key]:.4f}" for key in ["psnr", "detail_variance_ratio", "laplacian_correlation"]]
            lines.append(f"| {mode} | {phase} | {row['pixels']} | " + " | ".join(numbers) + " |")
    lines += ["", "## 相同相机的 3D 对照", "", "三列依次为 E5c、drive（含生成）、sweep（含生成）。相机来自同一个 E5c；含 +0.5 场景单位的横移，检验扫视之外的渲染。", "",
              "![frame 104 yaw 60](renders/common/018_compare.png)", "", "![frame 104 yaw 30](renders/common/014_compare.png)", "",
              "[全部固定视角对照视频](renders/common/comparison.mp4)"]
    for mode in ["drive", "sweep"]:
        lines += ["", f"## {mode} 直接生成与回写", "", f"![{mode}](./{mode}_after/100.png)", "",
                  f"[生成视频](./{mode}_views/filled_compare.mp4) · [同相机拟合及未监督偏移](./{mode}_after/comparison.mp4) · [几何验收](./{mode}_validation/validation.json)"]
    lines += ["", "生成内容仍是场景假设。固定场景往返可重复性不证明几何正确；本次不代表全场景、360° 或可碰撞仿真资产验收。", "",
              f"Commit: `{state['dashrecon_commit']}`；累计 {state['runtime_s']/60:.1f} 分钟。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[summary] {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
