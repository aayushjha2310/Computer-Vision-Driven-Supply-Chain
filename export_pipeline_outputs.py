"""
Run end-to-end pipeline and export all textual + visual outputs to project root.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_pipeline import SupplyChainPipeline
from src.utils.config import load_config


def _write_text_report(results: dict, out_path: Path) -> None:
    lines = [
        "=" * 72,
        "  VISION-DRIVEN GLOBAL SUPPLY CHAIN INTELLIGENCE — PIPELINE REPORT",
        "=" * 72,
        f"Status       : {results.get('status', 'unknown').upper()}",
        f"Started      : {results.get('started_at', 'N/A')}",
        f"Completed    : {results.get('completed_at', 'N/A')}",
        "",
    ]
    stages = results.get("stages", {})
    for stage_name, data in stages.items():
        lines.append("-" * 72)
        lines.append(f"STAGE: {stage_name.upper().replace('_', ' ')}")
        lines.append("-" * 72)
        lines.append(json.dumps(data, indent=2, default=str))
        lines.append("")

    if results.get("error"):
        lines.append("ERROR:")
        lines.append(results["error"])
        if results.get("traceback"):
            lines.append(results["traceback"])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_charts(results: dict, chart_dir: Path) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    stages = results.get("stages", {})

    # 1. ML model comparison (RMSE)
    ml = stages.get("ml_training", {})
    reg = ml.get("regression_results", {})
    if reg:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = list(reg.keys())
        rmses = [reg[m]["rmse"] for m in models]
        colors = ["#2ecc71" if m == ml.get("best_model") else "#3498db" for m in models]
        ax.bar(models, rmses, color=colors)
        ax.set_title("Price Forecast Model Comparison (RMSE)")
        ax.set_ylabel("RMSE (USD)")
        ax.set_xlabel("Model")
        for i, v in enumerate(rmses):
            ax.text(i, v + max(rmses) * 0.01, f"{v:,.0f}", ha="center", fontsize=9)
        fig.tight_layout()
        p = chart_dir / "01_model_rmse_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))

    # 2. Fraud detection F1 scores
    fraud = ml.get("fraud_detection", {})
    if fraud:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = list(fraud.keys())
        f1s = [fraud[m]["f1"] for m in models]
        ax.bar(models, f1s, color="#e74c3c")
        ax.set_title("Fraud Detection — F1 Score by Model")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        p = chart_dir / "02_fraud_detection_f1.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))

    # 3. Regional revenue
    regional = stages.get("data_ingestion", {}).get("regional_summary", [])
    if regional:
        fig, ax = plt.subplots(figsize=(8, 5))
        df = pd.DataFrame(regional)
        ax.bar(df["region"], df["total_revenue"] / 1e6, color="#9b59b6")
        ax.set_title("Total Revenue by Region (Millions USD)")
        ax.set_ylabel("Revenue ($M)")
        ax.set_xlabel("Region")
        fig.tight_layout()
        p = chart_dir / "03_regional_revenue.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))

    # 4. Fraud vs defect counts by region
    if regional:
        fig, ax = plt.subplots(figsize=(8, 5))
        df = pd.DataFrame(regional)
        x = range(len(df))
        w = 0.35
        ax.bar([i - w / 2 for i in x], df["fraud_count"], w, label="Fraud", color="#e67e22")
        ax.bar([i + w / 2 for i in x], df["defect_count"], w, label="Defect", color="#1abc9c")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["region"])
        ax.set_title("Fraud & Defect Counts by Region")
        ax.legend()
        fig.tight_layout()
        p = chart_dir / "04_fraud_defect_by_region.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))

    # 5. Damage inspection ratios
    damage = stages.get("computer_vision", {}).get("damage_inspections", [])
    if damage:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [Path(d["file"]).stem for d in damage]
        ratios = [d["damage_ratio"] * 100 for d in damage]
        ax.barh(labels, ratios, color="#c0392b")
        ax.set_title("SAM Damage Inspection — Damage Area (%)")
        ax.set_xlabel("Damage Ratio (%)")
        fig.tight_layout()
        p = chart_dir / "05_damage_inspection_ratios.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))

    return saved


def _copy_visual_artifacts(config: dict, visual_dir: Path) -> list[str]:
    visual_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    patterns = [
        (Path(config["paths"]["data_raw"]), ["*.jpg", "*.png", "*.mp4"]),
        (Path(config["paths"]["data_raw"]) / "annotated", ["*.jpg"]),
        (Path(config["paths"]["data_synthetic"]), ["*.png"]),
    ]
    for base, globs in patterns:
        if not base.exists():
            continue
        for pattern in globs:
            for src in base.glob(pattern):
                dest = visual_dir / src.name
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                copied.append(str(dest))
    return copied


def export_to_root(results: dict, config: dict) -> dict[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = ROOT / "PIPELINE_REPORT.json"
    txt_path = ROOT / "PIPELINE_REPORT.txt"
    summary_path = ROOT / "PIPELINE_SUMMARY.txt"
    visual_dir = ROOT / "pipeline_visual_outputs"
    chart_dir = ROOT / "pipeline_chart_outputs"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    _write_text_report(results, txt_path)

    charts = _generate_charts(results, chart_dir)
    visuals = _copy_visual_artifacts(config, visual_dir)

    summary_lines = [
        f"Pipeline Run Export — {ts} UTC",
        f"Status: {results.get('status', 'unknown')}",
        "",
        "TEXT OUTPUTS (project root):",
        f"  - {json_path.name}",
        f"  - {txt_path.name}",
        f"  - {summary_path.name}",
        "",
        f"GENERATED CHARTS ({len(charts)} files):",
    ]
    summary_lines.extend([f"  - {Path(c).name}" for c in charts])
    summary_lines.append("")
    summary_lines.append(f"VISUAL ARTIFACTS ({len(visuals)} files):")
    summary_lines.extend([f"  - {Path(v).name}" for v in visuals[:30]])
    if len(visuals) > 30:
        summary_lines.append(f"  ... and {len(visuals) - 30} more")

    stages = results.get("stages", {})
    if stages.get("ml_training"):
        ml = stages["ml_training"]
        summary_lines.extend([
            "",
            "KEY METRICS:",
            f"  Best price model : {ml.get('best_model')}",
            f"  Optuna best RMSE : {ml.get('optuna_best_rmse')}",
        ])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "json": str(json_path),
        "text": str(txt_path),
        "summary": str(summary_path),
        "charts_dir": str(chart_dir),
        "visuals_dir": str(visual_dir),
        "n_charts": str(len(charts)),
        "n_visuals": str(len(visuals)),
    }


def main():
    print("Running full pipeline and exporting outputs to project root...\n")
    pipeline = SupplyChainPipeline()
    results = pipeline.run_all()
    config = load_config()
    export_info = export_to_root(results, config)

    print("\n" + "=" * 60)
    print("OUTPUTS SAVED TO PROJECT ROOT")
    print("=" * 60)
    for k, v in export_info.items():
        print(f"  {k}: {v}")

    if results.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
