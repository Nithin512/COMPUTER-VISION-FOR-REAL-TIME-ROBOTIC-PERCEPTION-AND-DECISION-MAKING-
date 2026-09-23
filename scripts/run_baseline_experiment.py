from __future__ import annotations

import csv
import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
TABLES = PROJECT_ROOT / "outputs" / "tables"
LOGS = PROJECT_ROOT / "outputs" / "logs"

STYLE = {
    "bg": "#FFFFFF",
    "primary": "#183B56",
    "secondary": "#486581",
    "accent": "#2F80ED",
    "highlight": "#56B4A9",
    "light": "#EAF2F8",
    "neutral": "#F4F6F8",
    "text": "#1F2933",
    "muted": "#52606D",
    "border": "#BCCCDC",
}


def ensure_dirs() -> None:
    for path in [FIGURES, TABLES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{STYLE["secondary"]}"/>
    </marker>
    <style>
      .title {{ font: 700 24px Arial, sans-serif; fill: {STYLE["primary"]}; }}
      .subtitle {{ font: 400 13px Arial, sans-serif; fill: {STYLE["muted"]}; }}
      .box {{ fill: {STYLE["neutral"]}; stroke: {STYLE["border"]}; stroke-width: 1.4; rx: 8; ry: 8; }}
      .box2 {{ fill: {STYLE["light"]}; stroke: {STYLE["border"]}; stroke-width: 1.4; rx: 8; ry: 8; }}
      .label {{ font: 700 14px Arial, sans-serif; fill: {STYLE["text"]}; }}
      .small {{ font: 400 12px Arial, sans-serif; fill: {STYLE["muted"]}; }}
      .line {{ fill: none; stroke: {STYLE["secondary"]}; stroke-width: 1.5; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" fill="{STYLE["bg"]}"/>
  <text class="title" x="40" y="48">{title}</text>
'''


def write_pipeline_figure() -> None:
    boxes = [
        (40, 105, "Image Stream", "Public dataset frames"),
        (235, 105, "Pre-processing", "Resize, normalise, filter"),
        (430, 105, "Perception Model", "Objects, regions, motion cues"),
        (625, 105, "Temporal Reasoning", "Tracking and consistency"),
        (820, 105, "Decision Rule", "Confidence and risk state"),
        (1015, 105, "Action Policy", "Proceed, slow, stop, inspect"),
    ]
    parts = [svg_header(1240, 310, "Real-Time Computer Vision Evaluation Pipeline")]
    parts.append(f'  <text class="subtitle" x="40" y="72">Methodology figure for linking visual perception outputs to robotic decision-making metrics.</text>')
    for i, (x, y, heading, detail) in enumerate(boxes):
        cls = "box2" if i in {2, 4} else "box"
        parts.append(f'  <rect class="{cls}" x="{x}" y="{y}" width="165" height="88"/>')
        parts.append(f'  <text class="label" x="{x + 16}" y="{y + 34}">{heading}</text>')
        parts.append(f'  <text class="small" x="{x + 16}" y="{y + 58}">{detail}</text>')
        if i < len(boxes) - 1:
            parts.append(f'  <path class="line" d="M {x + 165} {y + 44} L {x + 195} {y + 44}"/>')
    parts.append(f'  <rect x="40" y="230" width="1160" height="42" fill="{STYLE["neutral"]}" stroke="{STYLE["border"]}" stroke-width="1.2" rx="8" ry="8"/>')
    parts.append(f'  <text class="small" x="58" y="256">Outputs: latency, frames per second, detection confidence, temporal stability, decision consistency, and failure-case evidence.</text>')
    parts.append("</svg>\n")
    write_text(FIGURES / "figure_3_1_realtime_cv_evaluation_pipeline.svg", "\n".join(parts))


def write_metric_framework_figure() -> None:
    cards = [
        (70, 115, "Perception Quality", "confidence, IoU, precision, recall"),
        (365, 115, "Real-Time Feasibility", "latency, FPS, processing variance"),
        (660, 115, "Decision Reliability", "risk state, action consistency"),
        (955, 115, "Robustness", "lighting, clutter, motion, occlusion"),
    ]
    parts = [svg_header(1240, 360, "Evaluation Framework for Robotic Perception")]
    parts.append(f'  <text class="subtitle" x="40" y="72">Quality dimensions used to judge whether visual perception can support timely robotic decisions.</text>')
    parts.append(f'  <rect x="455" y="250" width="330" height="58" fill="{STYLE["light"]}" stroke="{STYLE["border"]}" stroke-width="1.4" rx="8" ry="8"/>')
    parts.append(f'  <text class="label" x="505" y="284">Decision-making readiness judgement</text>')
    for x, y, heading, detail in cards:
        parts.append(f'  <rect class="box" x="{x}" y="{y}" width="215" height="86"/>')
        parts.append(f'  <text class="label" x="{x + 18}" y="{y + 34}">{heading}</text>')
        parts.append(f'  <text class="small" x="{x + 18}" y="{y + 58}">{detail}</text>')
        parts.append(f'  <path class="line" d="M {x + 108} {y + 86} C {x + 108} 230 620 230 620 250"/>')
    parts.append("</svg>\n")
    write_text(FIGURES / "figure_3_2_evaluation_framework.svg", "\n".join(parts))


def write_metric_plan() -> None:
    rows = [
        ["dimension", "metric", "source", "chapter_four_use"],
        ["perception_quality", "confidence distribution", "model detections or annotated dataset", "findings and analysis"],
        ["perception_quality", "precision/recall or IoU", "only if labels are available", "findings and analysis"],
        ["real_time_feasibility", "latency_ms", "local experiment run log", "analysis"],
        ["real_time_feasibility", "frames_per_second", "local experiment run log", "analysis"],
        ["decision_reliability", "action_state_consistency", "decision-rule output across frames", "discussion"],
        ["robustness", "failure_case_type", "manual review of low-confidence or unstable frames", "discussion"],
    ]
    with (TABLES / "evaluation_metric_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def image_files() -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in DATA_RAW.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def run_image_proxy(images: list[Path]) -> dict[str, object]:
    try:
        from PIL import Image, ImageFilter, ImageStat
    except Exception as exc:
        return {"status": "skipped", "reason": f"Pillow unavailable: {exc}", "rows": 0}

    rows = [["file", "width", "height", "preprocess_ms", "mean_intensity", "edge_salience_proxy", "decision_proxy"]]
    for path in images:
        start = time.perf_counter()
        with Image.open(path) as img:
            gray = img.convert("L")
            width, height = gray.size
            resized = gray.resize((320, max(1, int(320 * height / width))))
            edges = resized.filter(ImageFilter.FIND_EDGES)
            mean_intensity = ImageStat.Stat(resized).mean[0]
            edge_mean = ImageStat.Stat(edges).mean[0] / 255.0
        elapsed_ms = (time.perf_counter() - start) * 1000
        decision = "caution" if edge_mean >= 0.18 else "proceed"
        rows.append([
            str(path.relative_to(PROJECT_ROOT)),
            width,
            height,
            f"{elapsed_ms:.3f}",
            f"{mean_intensity:.3f}",
            f"{edge_mean:.5f}",
            decision,
        ])

    with (TABLES / "image_proxy_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return {"status": "completed", "rows": len(rows) - 1, "output": "outputs/tables/image_proxy_metrics.csv"}


def main() -> None:
    ensure_dirs()
    write_pipeline_figure()
    write_metric_framework_figure()
    write_metric_plan()

    images = image_files()
    image_result = run_image_proxy(images) if images else {"status": "waiting_for_dataset", "rows": 0}
    status = {
        "project": "Project_7_Nithin",
        "title": "Computer Vision for Real-Time Robotic Perception and Decision-Making",
        "generated_protocol_figures": [
            "outputs/figures/figure_3_1_realtime_cv_evaluation_pipeline.svg",
            "outputs/figures/figure_3_2_evaluation_framework.svg",
        ],
        "metric_plan": "outputs/tables/evaluation_metric_plan.csv",
        "raw_images_found": len(images),
        "image_proxy_evaluation": image_result,
        "final_findings_ready": bool(images and image_result.get("status") == "completed"),
    }
    write_text(LOGS / "experiment_status.json", json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
