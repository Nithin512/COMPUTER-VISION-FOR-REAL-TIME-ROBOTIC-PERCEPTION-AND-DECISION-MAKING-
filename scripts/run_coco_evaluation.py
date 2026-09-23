from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
COCO = RAW / "coco_val2017_subset"
ANN_ZIP = COCO / "annotations_trainval2017.zip"
ANN_JSON = COCO / "annotations" / "instances_val2017.json"
IMG_DIR = COCO / "images"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
LOG = ROOT / "outputs" / "logs"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"

STYLE = {"bg":"#FFFFFF","primary":"#183B56","secondary":"#486581","accent":"#2F80ED","highlight":"#56B4A9","light":"#EAF2F8","neutral":"#F4F6F8","text":"#1F2933","muted":"#52606D","border":"#BCCCDC"}
TARGET = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign","bench","chair","backpack","handbag"}
DECISION = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}


def dirs() -> None:
    for p in [COCO, IMG_DIR, PROCESSED, FIG, TAB, LOG]:
        p.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def annotations() -> dict[str, Any]:
    dirs()
    if not ANN_JSON.exists():
        download(ANN_URL, ANN_ZIP)
        with zipfile.ZipFile(ANN_ZIP) as z:
            z.extract("annotations/instances_val2017.json", COCO)
    with ANN_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def select(coco: dict[str, Any], n: int, target_names: set[str]):
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    target_ids = {i for i, name in cats.items() if name in target_names}
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd") or ann["category_id"] not in target_ids or ann.get("area", 0) < 2500:
            continue
        by_image[ann["image_id"]].append(ann)
    scored = []
    for image_id, anns in by_image.items():
        names = {cats[a["category_id"]] for a in anns}
        scored.append((sum(x in DECISION for x in names), len(names), len(anns), image_id))
    selected_ids = [image_id for *_, image_id in sorted(scored, reverse=True)[:n]]
    return [images[i] for i in selected_ids], by_image, cats


def ensure_images(selected: list[dict[str, Any]]) -> None:
    for im in selected:
        download(IMG_URL.format(file_name=im["file_name"]), IMG_DIR / im["file_name"])


def iou(pred_box: list[float], gt_box: list[float]) -> float:
    ax1, ay1, ax2, ay2 = pred_box
    bx1, by1, bw, bh = gt_box
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bw) * max(0.0, bh)
    return inter / (aa + ba - inter) if aa + ba - inter else 0.0


def match(preds: list[dict[str, Any]], gts: list[dict[str, Any]]):
    used: set[int] = set()
    vals: list[float] = []
    for pred in sorted(preds, key=lambda x: x["score"], reverse=True):
        best_i, best = None, 0.0
        for i, gt in enumerate(gts):
            if i in used or pred["category_id"] != gt["category_id"]:
                continue
            v = iou(pred["box"], gt["bbox"])
            if v > best:
                best_i, best = i, v
        if best_i is not None and best >= 0.5:
            used.add(best_i)
            vals.append(best)
    return len(vals), vals


def action_state(preds: list[dict[str, Any]], area: float) -> str:
    rel = [p for p in preds if p["category"] in DECISION and p["score"] >= 0.55]
    if not rel:
        return "inspect"
    biggest = 0.0
    for p in rel:
        x1, y1, x2, y2 = p["box"]
        biggest = max(biggest, ((x2 - x1) * (y2 - y1)) / area)
    if biggest >= 0.18 or any(p["category"] in {"person", "car", "truck", "bus"} and p["score"] >= 0.70 for p in rel):
        return "stop"
    return "slow"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot_style() -> None:
    plt.rcParams.update({"font.family":"Arial","figure.facecolor":STYLE["bg"],"axes.facecolor":STYLE["bg"],"axes.edgecolor":STYLE["border"],"axes.labelcolor":STYLE["text"],"xtick.color":STYLE["muted"],"ytick.color":STYLE["muted"],"text.color":STYLE["text"],"axes.titleweight":"bold","axes.titlecolor":STYLE["primary"]})


def bar(labels, values, title, ylabel, stem):
    plot_style(); fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colors = [STYLE["primary"], STYLE["accent"], STYLE["highlight"], STYLE["secondary"]]
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_title(title, loc="left", pad=14); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"{stem}.svg"); fig.savefig(FIG / f"{stem}.png", dpi=220); plt.close(fig)


def annotate(rows, preds_by_file):
    font = ImageFont.load_default(); panel_w, panel_h = 560, 430
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), "white"); draw_canvas = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows[:4]):
        im = Image.open(IMG_DIR / row["file_name"]).convert("RGB")
        original_w, original_h = im.size
        im.thumbnail((520, 360)); sx, sy = im.width / original_w, im.height / original_h
        draw = ImageDraw.Draw(im)
        for pred in preds_by_file.get(row["file_name"], [])[:5]:
            if pred["score"] < 0.45:
                continue
            x1, y1, x2, y2 = pred["box"]; box = [x1*sx, y1*sy, x2*sx, y2*sy]
            draw.rectangle(box, outline=STYLE["accent"], width=3)
            label = f"{pred['category']} {pred['score']:.2f}"; tb = draw.textbbox((box[0], max(0, box[1]-16)), label, font=font)
            draw.rectangle(tb, fill=STYLE["light"]); draw.text((box[0], max(0, box[1]-16)), label, fill=STYLE["text"], font=font)
        x, y = (idx % 2) * panel_w + 24, (idx // 2) * panel_h + 24
        canvas.paste(im, (x, y)); draw_canvas.text((x, y + im.height + 12), f"{row['file_name']} | decision: {row['decision_state']}", fill=STYLE["primary"], font=font)
    canvas.save(FIG / "figure_4_1_detection_examples.png", dpi=(220, 220))


def figures(rows, summary):
    bar(["Precision", "Recall", "F1", "Mean IoU"], [summary["micro_precision"], summary["micro_recall"], summary["micro_f1"], summary["mean_iou_matched"]], "Detection Quality on COCO Validation Subset", "Score", "figure_4_2_detection_quality")
    plot_style(); fig, ax = plt.subplots(figsize=(7.4, 4.2)); x = np.arange(len(rows)); y = [r["latency_ms"] for r in rows]
    ax.plot(x, y, color=STYLE["primary"], marker="o", linewidth=1.6); ax.set_title("Runtime Profile Across Evaluation Images", loc="left", pad=14); ax.set_xlabel("Evaluation image"); ax.set_ylabel("Latency per image (ms)"); ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "figure_4_3_runtime_profile.svg"); fig.savefig(FIG / "figure_4_3_runtime_profile.png", dpi=220); plt.close(fig)
    counts = Counter(r["decision_state"] for r in rows); labels = [x for x in ["stop","slow","inspect","proceed"] if x in counts]
    bar(labels, [counts[x] for x in labels], "Decision-State Distribution", "Image count", "figure_4_4_decision_states")
    plot_style(); fig, ax = plt.subplots(figsize=(7.4, 4.2)); ax.scatter([r["mean_confidence"] for r in rows], [r["recall"] for r in rows], s=58, color=STYLE["accent"], edgecolor=STYLE["primary"], linewidth=0.7)
    ax.set_title("Confidence and Recall Relationship", loc="left", pad=14); ax.set_xlabel("Mean detection confidence"); ax.set_ylabel("Recall"); ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.grid(color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "figure_4_5_confidence_recall.svg"); fig.savefig(FIG / "figure_4_5_confidence_recall.png", dpi=220); plt.close(fig)


def load_model(model_name: str):
    if model_name == "high_accuracy":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
        model_label = "torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1"
        model_source = "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html"
    elif model_name == "fast":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, progress=True)
        model_label = "torchvision fasterrcnn_mobilenet_v3_large_320_fpn COCO_V1"
        model_source = "https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    return model, weights, model_label, model_source


def evaluate(sample_count: int, threshold: float, model_name: str, class_scope: str):
    target_names = DECISION if class_scope == "decision_only" else TARGET
    coco = annotations(); selected, by_image, coco_cats = select(coco, sample_count, target_names); ensure_images(selected)
    model, weights, model_label, model_source = load_model(model_name)
    transform = weights.transforms(); labels = weights.meta["categories"]
    rows, preds_by_file, all_ious = [], {}, []; total_gt = total_pred = total_match = 0
    with torch.inference_mode():
        for im_meta in selected:
            image = Image.open(IMG_DIR / im_meta["file_name"]).convert("RGB"); tensor = transform(image)
            start = time.perf_counter(); out = model([tensor])[0]; latency = (time.perf_counter() - start) * 1000
            preds = []
            for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
                lid = int(label.item()); name = labels[lid] if lid < len(labels) else str(lid); s = float(score.item())
                if s >= threshold and name in target_names:
                    preds.append({"box":[float(x) for x in box.tolist()], "score":s, "category_id":lid, "category":name})
            gts = by_image[im_meta["id"]]; m, ious = match(preds, gts)
            gt, pr = len(gts), len(preds); precision = m / pr if pr else 0.0; recall = m / gt if gt else 0.0; f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
            mean_conf = statistics.mean([p["score"] for p in preds]) if preds else 0.0; state = action_state(preds, im_meta["width"] * im_meta["height"])
            row = {"image_id":im_meta["id"],"file_name":im_meta["file_name"],"width":im_meta["width"],"height":im_meta["height"],"gt_objects":gt,"detections":pr,"matched_detections":m,"precision":round(precision,4),"recall":round(recall,4),"f1":round(f1,4),"mean_iou_matched":round(statistics.mean(ious),4) if ious else 0.0,"mean_confidence":round(mean_conf,4),"latency_ms":round(latency,3),"fps":round(1000/latency,3) if latency else 0.0,"decision_state":state}
            rows.append(row); preds_by_file[im_meta["file_name"]] = preds; all_ious += ious; total_gt += gt; total_pred += pr; total_match += m
    lat = [r["latency_ms"] for r in rows]; micro_p = total_match / total_pred if total_pred else 0.0; micro_r = total_match / total_gt if total_gt else 0.0; micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0.0
    summary = {"dataset":"COCO val2017 robotically salient medium/large-object subset","dataset_source":"https://cocodataset.org/","annotation_source":ANN_URL,"model":model_label,"model_source":model_source,"class_scope":class_scope,"target_categories":sorted(target_names),"subset_rule":"non-crowd target-category objects with annotation area >= 2500 pixels","sample_images":len(rows),"score_threshold":threshold,"total_ground_truth_objects":total_gt,"total_detections":total_pred,"total_matched_detections":total_match,"micro_precision":round(micro_p,4),"micro_recall":round(micro_r,4),"micro_f1":round(micro_f1,4),"mean_iou_matched":round(statistics.mean(all_ious),4) if all_ious else 0.0,"mean_latency_ms":round(statistics.mean(lat),3) if lat else 0.0,"median_latency_ms":round(statistics.median(lat),3) if lat else 0.0,"mean_fps":round(1000/statistics.mean(lat),3) if lat and statistics.mean(lat) else 0.0,"decision_state_counts":dict(Counter(r["decision_state"] for r in rows)),"final_findings_ready":bool(rows)}
    write_csv(TAB / "detection_metrics_by_image.csv", rows); write_csv(TAB / "summary_metrics.csv", [{k: json.dumps(v) if isinstance(v, dict) else v for k, v in summary.items()}])
    (PROCESSED / "selected_coco_images.json").write_text(json.dumps({"images":selected,"target_categories":sorted(target_names),"class_scope":class_scope}, indent=2), encoding="utf-8")
    (PROCESSED / "predictions_by_image.json").write_text(json.dumps(preds_by_file, indent=2), encoding="utf-8")
    (LOG / "coco_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    annotate(rows, preds_by_file); figures(rows, summary)
    status = {"project":ROOT.name,"complete":True,"tables":["outputs/tables/detection_metrics_by_image.csv","outputs/tables/summary_metrics.csv"],"figures":["outputs/figures/figure_4_1_detection_examples.png","outputs/figures/figure_4_2_detection_quality.png","outputs/figures/figure_4_3_runtime_profile.png","outputs/figures/figure_4_4_decision_states.png","outputs/figures/figure_4_5_confidence_recall.png"],"summary":summary}
    (LOG / "experiment_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project 7 COCO-based computer vision evaluation."); parser.add_argument("--sample-count", type=int, default=16); parser.add_argument("--threshold", type=float, default=0.45); parser.add_argument("--model", choices=["high_accuracy", "fast"], default="high_accuracy"); parser.add_argument("--class-scope", choices=["all_target", "decision_only"], default="all_target"); args = parser.parse_args()
    print(json.dumps(evaluate(args.sample_count, args.threshold, args.model, args.class_scope), indent=2))


if __name__ == "__main__":
    main()





