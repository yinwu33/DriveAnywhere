"""Summarize a completed E11 local probe into JSON and Markdown (no GPU).

Pixel-weighted RGB MSE is pooled before converting to PSNR. Detail ratios use
pooled within-image Laplacian variances; correlations are pixel-weighted. These
measure synthetic target fitting, not hidden-view ground-truth correctness.
"""
import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_dir", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.probe_dir / "run_status.json").read_text())
    if not state["completed"]:
        raise ValueError("probe is not complete; inspect run_status.json")
    output = []
    names = [stage["name"] for stage in state["stages"] if not stage["name"].endswith("_train")]
    for name in names:
        metrics = json.loads((args.probe_dir / name / "metrics.json").read_text())
        regions = list(metrics["rows"][0]["regions"])
        for region in regions:
            rows = [row["regions"][region] for row in metrics["rows"] if row["regions"][region]["pixels"] > 0]
            pixels = sum(row["pixels"] for row in rows)
            if pixels == 0:
                output.append({"run": name, "region": region, "pixels": 0, "psnr": None,
                               "mae": None, "detail_variance_ratio": None, "laplacian_correlation": None})
                continue
            pooled = {key: sum(row[key] * row["pixels"] for row in rows) / pixels
                      for key in ("mse", "mae", "target_laplacian_variance", "render_laplacian_variance")}
            correlated = [row for row in rows if row["laplacian_correlation"] is not None]
            correlation = None
            if correlated:
                correlation = sum(row["laplacian_correlation"] * row["pixels"] for row in correlated) / sum(row["pixels"] for row in correlated)
            ratio = None
            if pooled["target_laplacian_variance"] > 0:
                ratio = pooled["render_laplacian_variance"] / pooled["target_laplacian_variance"]
            output.append({"run": name, "region": region, "pixels": pixels,
                           "psnr": -10 * math.log10(max(pooled["mse"], 1e-12)), "mae": pooled["mae"],
                           "detail_variance_ratio": ratio, "laplacian_correlation": correlation})
    result = {"kind": "historical-input implementation diagnostic, not ground truth",
              "dashrecon_commit": state["dashrecon_commit"], "rows": output,
              "total_stage_runtime_s": sum(stage["elapsed_s"] for stage in state["stages"])}
    (args.probe_dir / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["# E11 局部蒸馏诊断", "", "历史产物实现诊断；不代表隐藏区域正确率或无泄漏基准。E10 为 35 步生成，E11 为 8 步缓存接线生成。",
             "", "PSNR 对照合成目标。细节方差比可能被噪声提高，必须结合细节相关性和新视角目测；未监督视角没有像素真值。",
             "", "| Run | Region | Pixels | PSNR | MAE | Detail variance ratio | Detail correlation |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in output:
        numbers = ["unknown" if row[key] is None else f"{row[key]:.4f}" for key in ("psnr", "mae", "detail_variance_ratio", "laplacian_correlation")]
        lines.append(f"| {row['run']} | {row['region']} | {row['pixels']} | " + " | ".join(numbers) + " |")
    lines += ["", "## 图像与视频", "", "四列依次为直接生成、同相机 3D、未监督偏航、未监督横移。"]
    for name in names:
        lines += ["", f"### {name}", "", f"![{name}]({name}/052.png)", "",
                  f"[序列对比]({name}/comparison.mp4) · [局部往返]({name}/local_roundtrip.mp4)"]
    lines += ["", "往返是固定 3D 场景的重复渲染检查，不是独立几何证据。", "",
              f"Commit: `{state['dashrecon_commit']}`；累计阶段耗时 {result['total_stage_runtime_s']/60:.1f} 分钟。"]
    (args.probe_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
