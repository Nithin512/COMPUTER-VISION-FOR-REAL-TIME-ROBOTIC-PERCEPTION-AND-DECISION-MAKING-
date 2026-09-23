from __future__ import annotations

import argparse, csv, json, math, os, random, statistics, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests, torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
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
TARGET = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]
TARGET_SET = set(TARGET)
STATES = ["proceed", "inspect", "slow", "stop"]
RANK = {s: i for i, s in enumerate(STATES)}
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SIZES = ["small", "medium", "large"]
STYLE = {"primary":"#183B56", "secondary":"#486581", "accent":"#2F80ED", "highlight":"#56B4A9", "light":"#EAF2F8", "neutral":"#F4F6F8", "text":"#1F2933", "muted":"#52606D", "border":"#BCCCDC", "risk":"#9B1C1C"}


def dirs():
    for p in [COCO, IMG_DIR, PROCESSED, FIG, TAB, LOG]:
        p.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1024:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=90) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def load_coco() -> dict[str, Any]:
    dirs()
    if not ANN_JSON.exists():
        download(ANN_URL, ANN_ZIP)
        with zipfile.ZipFile(ANN_ZIP) as z:
            z.extract("annotations/instances_val2017.json", COCO)
    return json.loads(ANN_JSON.read_text(encoding="utf-8"))


def size_bin(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def gt_area(gt):
    return max(0.0, gt["bbox"][2]) * max(0.0, gt["bbox"][3])


def pred_area(p):
    x1, y1, x2, y2 = p["box"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def indexes(coco):
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_id = {v: k for k, v in cats.items()}
    images = {im["id"]: im for im in coco["images"]}
    target_ids = {name_to_id[n] for n in TARGET if n in name_to_id}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd") or ann["category_id"] not in target_ids:
            continue
        row = dict(ann)
        row["category_name"] = cats[row["category_id"]]
        row["size_bin"] = size_bin(float(row.get("area", gt_area(row))))
        by_image[row["image_id"]].append(row)
    return cats, name_to_id, images, by_image


def state_from(objs, width, height, pred=False):
    if not objs:
        return "proceed"
    area = max(width * height, 1)
    rels = []
    for obj in objs:
        name = obj.get("category") if pred else obj.get("category_name")
        if name not in TARGET_SET:
            continue
        rel = pred_area(obj) / area if pred else gt_area(obj) / area
        score = float(obj.get("score", 1.0))
        rels.append((name, rel, score))
    if not rels:
        return "proceed"
    if any(n in {"person", "car", "bus", "truck"} and r >= 0.08 and s >= 0.55 for n, r, s in rels):
        return "stop"
    if any(n == "stop sign" and r >= 0.01 and s >= 0.55 for n, r, s in rels):
        return "stop"
    if any(r >= 0.03 and s >= 0.50 for _, r, s in rels) or len(rels) >= 3:
        return "slow"
    return "inspect"


def sample(coco, n: int, positive_ratio: float, seed: int):
    _cats, _name_to_id, images, by_image = indexes(coco)
    rng = random.Random(seed)
    pos_n = min(len(by_image), max(1, round(n * positive_ratio)))
    neg_n = max(0, n - pos_n)
    class_to_images = {name: [] for name in TARGET}
    for image_id, anns in by_image.items():
        for name in {a["category_name"] for a in anns}:
            class_to_images[name].append(image_id)
    for ids in class_to_images.values():
        rng.shuffle(ids)
    selected, seen = [], set()
    cursors = {name: 0 for name in TARGET}
    while len(selected) < pos_n:
        moved = False
        for name in TARGET:
            ids = class_to_images[name]
            while cursors[name] < len(ids) and ids[cursors[name]] in seen:
                cursors[name] += 1
            if cursors[name] < len(ids):
                image_id = ids[cursors[name]]
                selected.append(image_id)
                seen.add(image_id)
                cursors[name] += 1
                moved = True
                if len(selected) >= pos_n:
                    break
        if not moved:
            break
    fill = [i for i in by_image if i not in seen]
    rng.shuffle(fill)
    for image_id in fill:
        if len(selected) >= pos_n:
            break
        selected.append(image_id)
        seen.add(image_id)
    negatives = [i for i in images if i not in by_image and i not in seen]
    rng.shuffle(negatives)
    for image_id in negatives[:neg_n]:
        selected.append(image_id)
        seen.add(image_id)
    rng.shuffle(selected)
    selected_images = [images[i] for i in selected]
    selected_by_image = {i: by_image.get(i, []) for i in selected}
    class_counts, size_counts, states = Counter(), Counter(), Counter()
    for im in selected_images:
        anns = selected_by_image[im["id"]]
        states[state_from(anns, im["width"], im["height"])] += 1
        for ann in anns:
            class_counts[ann["category_name"]] += 1
            size_counts[ann["size_bin"]] += 1
    profile = {
        "seed": seed,
        "sample_images": len(selected_images),
        "positive_images": sum(1 for im in selected_images if selected_by_image[im["id"]]),
        "background_images": sum(1 for im in selected_images if not selected_by_image[im["id"]]),
        "target_objects": sum(len(selected_by_image[im["id"]]) for im in selected_images),
        "class_counts": dict(class_counts),
        "size_counts": dict(size_counts),
        "ground_truth_decision_states": dict(states),
        "target_classes": TARGET,
        "small_object_policy": "small, medium and large objects retained; no 2500-pixel exclusion applied",
    }
    return selected_images, selected_by_image, profile


def ensure_images(selected, workers: int):
    def fetch(im):
        download(IMG_URL.format(file_name=im["file_name"]), IMG_DIR / im["file_name"])
        return im["file_name"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, im) for im in selected]
        for fut in as_completed(futures):
            fut.result()


def iou(pred_box, gt_bbox) -> float:
    ax1, ay1, ax2, ay2 = pred_box
    bx1, by1, bw, bh = gt_bbox
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bw) * max(0, bh)
    return inter / (aa + ba - inter) if aa + ba - inter else 0.0


def match(preds, gts):
    used, matches, fp = set(), [], []
    for pi, pred in sorted(enumerate(preds), key=lambda x: x[1]["score"], reverse=True):
        best_i, best = None, 0.0
        for gi, gt in enumerate(gts):
            if gi in used or pred["category_id"] != gt["category_id"]:
                continue
            val = iou(pred["box"], gt["bbox"])
            if val > best:
                best_i, best = gi, val
        if best_i is not None and best >= 0.50:
            used.add(best_i)
            matches.append({"pred_index": pi, "gt_index": best_i, "iou": best})
        else:
            fp.append(pi)
    fn = [i for i in range(len(gts)) if i not in used]
    return {"matches": matches, "fp_indices": fp, "fn_indices": fn}


def load_model(name: str):
    if name == "high_accuracy":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
        label = "Faster R-CNN ResNet50-FPN v2"
        family = "accuracy-oriented two-stage detector"
        source = "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html"
    else:
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights, progress=True)
        label = "SSDlite320 MobileNetV3-Large"
        family = "lightweight single-shot detector"
        source = "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html"
    model.eval()
    return model, weights, label, family, source


def perturb(image, condition: str, rng: random.Random):
    if condition == "original":
        return image
    if condition == "brightness_low":
        return ImageEnhance.Brightness(image).enhance(0.62)
    if condition == "brightness_high":
        return ImageEnhance.Brightness(image).enhance(1.35)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "noise":
        arr = np.asarray(image).astype(np.int16)
        noise = np.random.default_rng(9107).normal(0, 18, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if condition == "occlusion":
        im = image.copy()
        d = ImageDraw.Draw(im)
        w, h = im.size
        x0, y0 = round(w * 0.38), round(h * 0.36)
        d.rectangle([x0, y0, x0 + round(w * 0.24), y0 + round(h * 0.24)], fill=(244, 246, 248))
        return im
    raise ValueError(condition)


def infer(model_name: str, selected, name_to_id, min_score: float, threads: int, condition="original"):
    torch.set_num_threads(max(1, threads))
    model, weights, label, family, source = load_model(model_name)
    transform = weights.transforms()
    categories = weights.meta["categories"]
    preds_by_file, latencies = {}, []
    rng = random.Random(9107)
    with torch.inference_mode():
        for idx, im in enumerate(selected, 1):
            image = Image.open(IMG_DIR / im["file_name"]).convert("RGB")
            tensor = transform(perturb(image, condition, rng))
            start = time.perf_counter()
            out = model([tensor])[0]
            latency = (time.perf_counter() - start) * 1000
            preds = []
            for box, score, label_id in zip(out["boxes"], out["scores"], out["labels"]):
                lid = int(label_id.item())
                category = categories[lid] if lid < len(categories) else str(lid)
                conf = float(score.item())
                if category in TARGET_SET and category in name_to_id and conf >= min_score:
                    preds.append({"box": [float(x) for x in box.tolist()], "score": conf, "category": category, "category_id": name_to_id[category]})
            preds_by_file[im["file_name"]] = preds
            latencies.append({"model": model_name, "condition": condition, "image_id": im["id"], "file_name": im["file_name"], "latency_ms": round(latency, 3), "fps": round(1000 / latency, 4) if latency else 0.0, "prediction_count_min_score": len(preds)})
            if idx % 25 == 0:
                print(f"{model_name}/{condition}: processed {idx}/{len(selected)} images")
    return {"model": model_name, "label": label, "family": family, "source": source, "condition": condition, "predictions": preds_by_file, "latencies": latencies}


def pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return values[int(k)] if f == c else values[f] * (c - k) + values[c] * (k - f)


def aggregate(predictions, selected, by_image, threshold: float, latencies):
    total_gt = total_pred = total_match = 0
    ious = []
    class_counts = {n: {"gt": 0, "pred": 0, "match": 0, "ious": []} for n in TARGET}
    size_counts = {n: {"gt": 0, "pred": 0, "match": 0, "ious": []} for n in SIZES}
    confusion = {(a, b): 0 for a in STATES for b in STATES}
    false_safe = unnecessary_stop = 0
    errors = []
    for im in selected:
        gts = by_image.get(im["id"], [])
        preds = [p for p in predictions.get(im["file_name"], []) if p["score"] >= threshold]
        res = match(preds, gts)
        total_gt += len(gts)
        total_pred += len(preds)
        total_match += len(res["matches"])
        for gt in gts:
            class_counts[gt["category_name"]]["gt"] += 1
            size_counts[gt["size_bin"]]["gt"] += 1
        for pr in preds:
            class_counts[pr["category"]]["pred"] += 1
            size_counts[size_bin(pred_area(pr))]["pred"] += 1
        for m in res["matches"]:
            gt, pr = gts[m["gt_index"]], preds[m["pred_index"]]
            ious.append(m["iou"])
            class_counts[gt["category_name"]]["match"] += 1
            class_counts[gt["category_name"]]["ious"].append(m["iou"])
            size_counts[gt["size_bin"]]["match"] += 1
            size_counts[gt["size_bin"]]["ious"].append(m["iou"])
        for idx in res["fp_indices"][:4]:
            pr = preds[idx]
            errors.append({"file_name": im["file_name"], "error_type": "false_positive", "category": pr["category"], "score": round(pr["score"], 4), "size_bin": size_bin(pred_area(pr)), "iou": ""})
        for idx in res["fn_indices"][:4]:
            gt = gts[idx]
            errors.append({"file_name": im["file_name"], "error_type": "false_negative", "category": gt["category_name"], "score": "", "size_bin": gt["size_bin"], "iou": ""})
        for m in res["matches"]:
            if m["iou"] < 0.65:
                gt, pr = gts[m["gt_index"]], preds[m["pred_index"]]
                errors.append({"file_name": im["file_name"], "error_type": "poor_localisation", "category": gt["category_name"], "score": round(pr["score"], 4), "size_bin": gt["size_bin"], "iou": round(m["iou"], 4)})
        actual = state_from(gts, im["width"], im["height"])
        predicted = state_from(preds, im["width"], im["height"], pred=True)
        confusion[(actual, predicted)] += 1
        if RANK[predicted] < RANK[actual]:
            false_safe += 1
        if predicted == "stop" and actual != "stop":
            unnecessary_stop += 1
    precision = total_match / total_pred if total_pred else 0.0
    recall = total_match / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lat = [float(x["latency_ms"]) for x in latencies]
    decision_correct = sum(v for (a, b), v in confusion.items() if a == b)
    return {
        "threshold": threshold,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "matched": total_match,
        "false_positives": total_pred - total_match,
        "false_negatives": total_gt - total_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": statistics.mean(ious) if ious else 0.0,
        "median_iou": statistics.median(ious) if ious else 0.0,
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "median_latency_ms": statistics.median(lat) if lat else 0.0,
        "std_latency_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "p95_latency_ms": pct(lat, 0.95),
        "fps": 1000 / statistics.mean(lat) if lat and statistics.mean(lat) else 0.0,
        "decision_accuracy": decision_correct / len(selected) if selected else 0.0,
        "false_safe_decisions": false_safe,
        "unnecessary_stop_decisions": unnecessary_stop,
        "class_counts": class_counts,
        "size_counts": size_counts,
        "confusion": confusion,
        "errors": errors,
    }


def ap50(predictions, selected, by_image):
    out = {}
    for cls in TARGET:
        gt_total = 0
        pred_items = []
        used = defaultdict(set)
        for im in selected:
            gt_total += sum(1 for g in by_image.get(im["id"], []) if g["category_name"] == cls)
            pred_items += [(im, p) for p in predictions.get(im["file_name"], []) if p["category"] == cls]
        pred_items.sort(key=lambda x: x[1]["score"], reverse=True)
        if gt_total == 0 or not pred_items:
            out[cls] = 0.0
            continue
        tp, fp = [], []
        for im, pr in pred_items:
            gts = [g for g in by_image.get(im["id"], []) if g["category_name"] == cls]
            best_i, best = None, 0.0
            for gi, gt in enumerate(gts):
                if gi in used[im["file_name"]]:
                    continue
                val = iou(pr["box"], gt["bbox"])
                if val > best:
                    best_i, best = gi, val
            if best_i is not None and best >= 0.50:
                used[im["file_name"]].add(best_i)
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recalls = tp / gt_total
        precisions = tp / np.maximum(tp + fp, 1)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        out[cls] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return out


def ci(predictions, selected, by_image, threshold, metric, seed, rounds=300):
    rng = random.Random(seed)
    vals = []
    fake_lat = [{"latency_ms": 1.0} for _ in selected]
    for _ in range(rounds):
        sampled = [rng.choice(selected) for _ in selected]
        vals.append(float(aggregate(predictions, sampled, by_image, threshold, fake_lat)[metric]))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot_style():
    plt.rcParams.update({"font.family":"Arial", "figure.facecolor":"#FFFFFF", "axes.facecolor":"#FFFFFF", "axes.edgecolor":STYLE["border"], "axes.labelcolor":STYLE["text"], "xtick.color":STYLE["muted"], "ytick.color":STYLE["muted"], "text.color":STYLE["text"], "axes.titleweight":"bold", "axes.titlecolor":STYLE["primary"]})


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=220)
    plt.savefig(FIG / f"{name}.svg")
    plt.close()


def make_figures(th_rows, class_rows, size_rows, robust_rows, confusion_rows):
    plot_style()
    best = [r for r in th_rows if r["is_best"] == "yes"]
    labels = [r["model_label"] for r in best]
    x = np.arange(len(labels)); width = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x-width, [float(r["precision"]) for r in best], width, label="Precision", color=STYLE["primary"])
    ax.bar(x, [float(r["recall"]) for r in best], width, label="Recall", color=STYLE["secondary"])
    ax.bar(x+width, [float(r["f1"]) for r in best], width, label="F1", color=STYLE["highlight"])
    ax.set_title("Overall Detection Performance at Selected Operating Points", loc="left", pad=14)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
    savefig("figure_4_1_overall_model_performance")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model, color in [("high_accuracy", STYLE["primary"]), ("lightweight", STYLE["highlight"] )]:
        rows = [r for r in th_rows if r["model"] == model]
        if rows:
            ax.plot([float(r["threshold"]) for r in rows], [float(r["f1"]) for r in rows], marker="o", linewidth=1.8, label=rows[0]["model_label"], color=color)
    ax.set_title("Confidence-Threshold Sensitivity", loc="left", pad=14)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1 score"); ax.set_ylim(0, 1)
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
    savefig("figure_4_2_threshold_sensitivity")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rows = [r for r in class_rows if r["is_best"] == "yes"]
    x = np.arange(len(TARGET)); width = 0.36
    for i, model in enumerate(["high_accuracy", "lightweight"]):
        vals = []
        for cls in TARGET:
            f = [r for r in rows if r["model"] == model and r["class"] == cls]
            vals.append(float(f[0]["f1"]) if f else 0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=model.replace("_", " "), color=STYLE["primary"] if i == 0 else STYLE["highlight"])
    ax.set_title("Per-Class F1 at Selected Operating Points", loc="left", pad=14)
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1); ax.set_xticks(x); ax.set_xticklabels(TARGET, rotation=35, ha="right")
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
    savefig("figure_4_3_per_class_f1")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(labels, [float(r["mean_latency_ms"]) for r in best], color=[STYLE["primary"], STYLE["highlight"]][:len(best)])
    ax.set_title("Accuracy-Latency Trade-Off", loc="left", pad=14)
    ax.set_ylabel("Mean latency per image (ms)"); ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False)
    savefig("figure_4_4_latency_tradeoff")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rows = [r for r in size_rows if r["is_best"] == "yes"]
    x = np.arange(len(SIZES)); width = 0.36
    for i, model in enumerate(["high_accuracy", "lightweight"]):
        vals = []
        for size in SIZES:
            f = [r for r in rows if r["model"] == model and r["size_bin"] == size]
            vals.append(float(f[0]["recall"]) if f else 0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=model.replace("_", " "), color=STYLE["primary"] if i == 0 else STYLE["highlight"])
    ax.set_title("Object-Size Recall", loc="left", pad=14)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1); ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
    savefig("figure_4_5_object_size_recall")

    if robust_rows:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for model, color in [("high_accuracy", STYLE["primary"]), ("lightweight", STYLE["highlight"] )]:
            rows = [r for r in robust_rows if r["model"] == model]
            if rows:
                ax.plot([r["condition"] for r in rows], [float(r["f1"]) for r in rows], marker="o", linewidth=1.8, label=model.replace("_", " "), color=color)
        ax.set_title("Robustness Subset F1 Under Controlled Perturbations", loc="left", pad=14)
        ax.set_ylabel("F1 score"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color=STYLE["border"], linewidth=0.6, alpha=0.6); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
        savefig("figure_4_6_robustness_f1")

    for model in ["high_accuracy", "lightweight"]:
        rows = [r for r in confusion_rows if r["model"] == model]
        if not rows:
            continue
        matrix = np.zeros((len(STATES), len(STATES)))
        for r in rows:
            matrix[STATES.index(r["actual_state"]), STATES.index(r["predicted_state"])] = int(r["count"])
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Decision-State Confusion Matrix: {model.replace('_', ' ')}", loc="left", pad=14)
        ax.set_xticks(np.arange(len(STATES))); ax.set_yticks(np.arange(len(STATES)))
        ax.set_xticklabels(STATES, rotation=30, ha="right"); ax.set_yticklabels(STATES)
        ax.set_xlabel("Predicted state"); ax.set_ylabel("Ground-truth proxy state")
        for i in range(len(STATES)):
            for j in range(len(STATES)):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=STYLE["text"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        savefig(f"figure_4_7_decision_confusion_{model}")


def error_figure(model_name, threshold, predictions, selected, by_image):
    candidates = []
    for im in selected:
        gts = by_image.get(im["id"], [])
        preds = [p for p in predictions.get(im["file_name"], []) if p["score"] >= threshold]
        res = match(preds, gts)
        difficulty = len(res["fp_indices"]) + len(res["fn_indices"]) + sum(1 for m in res["matches"] if m["iou"] < 0.65)
        if difficulty:
            candidates.append((difficulty, im, preds, res))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[:4]
    if not chosen:
        return
    canvas = Image.new("RGB", (1300, 1000), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 18); small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = small = ImageFont.load_default()
    for idx, (_d, im_meta, preds, res) in enumerate(chosen):
        img = Image.open(IMG_DIR / im_meta["file_name"]).convert("RGB")
        ow, oh = img.size; img.thumbnail((600, 405)); sx, sy = img.width / ow, img.height / oh
        d = ImageDraw.Draw(img)
        gts = by_image.get(im_meta["id"], [])
        matched_gt = {m["gt_index"] for m in res["matches"]}; matched_pred = {m["pred_index"] for m in res["matches"]}
        for gi, gt in enumerate(gts):
            x, y, w, h = gt["bbox"]
            color = STYLE["highlight"] if gi in matched_gt else STYLE["risk"]
            d.rectangle([x*sx, y*sy, (x+w)*sx, (y+h)*sy], outline=color, width=3)
            d.text((x*sx, max(0, y*sy-18)), f"GT {gt['category_name']}", fill=color, font=small)
        for pi, pr in enumerate(preds):
            if pi in matched_pred:
                continue
            x1, y1, x2, y2 = pr["box"]
            d.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline=STYLE["accent"], width=3)
            d.text((x1*sx, max(0, y1*sy-18)), f"FP {pr['category']} {pr['score']:.2f}", fill=STYLE["accent"], font=small)
        x0, y0 = (idx % 2) * 650 + 22, (idx // 2) * 500 + 22
        canvas.paste(img, (x0, y0))
        ImageDraw.Draw(canvas).text((x0, y0 + img.height + 10), f"{im_meta['file_name']} | {model_name}, threshold {threshold:.2f}", fill=STYLE["primary"], font=font)
    canvas.save(FIG / "figure_4_8_error_examples.png", dpi=(220, 220))


def main():
    parser = argparse.ArgumentParser(description="Expanded Project 7 COCO technical evaluation")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--positive-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    parser.add_argument("--models", nargs="+", choices=["high_accuracy", "lightweight"], default=["high_accuracy", "lightweight"])
    parser.add_argument("--robustness-count", type=int, default=20)
    parser.add_argument("--robustness-models", nargs="+", choices=["high_accuracy", "lightweight"], default=["high_accuracy", "lightweight"])
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    dirs()
    coco = load_coco()
    _cats, name_to_id, _images, _all_by_image = indexes(coco)
    selected, by_image, profile = sample(coco, args.sample_count, args.positive_ratio, args.seed)
    ensure_images(selected, args.download_workers)
    (PROCESSED / "extended_selected_coco_images.json").write_text(json.dumps({"profile": profile, "images": selected}, indent=2), encoding="utf-8")
    write_csv(TAB / "sample_profile.csv", [{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in profile.items()])

    outputs = {}
    for model_name in args.models:
        pred_path = PROCESSED / f"extended_predictions_{model_name}.json"
        lat_path = TAB / f"latency_{model_name}.csv"
        meta_path = LOG / f"model_meta_{model_name}.json"
        if args.reuse_predictions and pred_path.exists() and lat_path.exists() and meta_path.exists():
            outputs[model_name] = {
                **json.loads(meta_path.read_text(encoding="utf-8")),
                "predictions": json.loads(pred_path.read_text(encoding="utf-8")),
                "latencies": list(csv.DictReader(lat_path.open(newline="", encoding="utf-8"))),
            }
        else:
            res = infer(model_name, selected, name_to_id, args.min_score, args.threads)
            outputs[model_name] = res
            pred_path.write_text(json.dumps(res["predictions"], indent=2), encoding="utf-8")
            write_csv(lat_path, res["latencies"])
            meta_path.write_text(json.dumps({k: v for k, v in res.items() if k not in {"predictions", "latencies"}}, indent=2), encoding="utf-8")

    th_rows, class_rows, size_rows, confusion_rows, error_rows, ap_rows, map_rows = [], [], [], [], [], [], []
    best_by_model = {}
    for model_name, out in outputs.items():
        stats_list = [aggregate(out["predictions"], selected, by_image, t, out["latencies"]) for t in args.thresholds]
        best = max(stats_list, key=lambda s: (s["f1"], s["precision"], -s["false_safe_decisions"]))
        best_by_model[model_name] = best
        ap = ap50(out["predictions"], selected, by_image)
        map50 = statistics.mean([v for v in ap.values() if v > 0]) if any(v > 0 for v in ap.values()) else 0.0
        lo, hi = ci(out["predictions"], selected, by_image, best["threshold"], "f1", args.seed + len(model_name), 300)
        for cls, value in ap.items():
            ap_rows.append({"model": model_name, "model_label": out["label"], "class": cls, "ap50": round(value, 4)})
        map_rows.append({"model": model_name, "model_label": out["label"], "map50_subset": round(map50, 4), "best_threshold": best["threshold"], "f1_ci95_low": round(lo, 4), "f1_ci95_high": round(hi, 4)})
        for st in stats_list:
            is_best = st is best
            th_rows.append({"model": model_name, "model_label": out["label"], "model_family": out["family"], "threshold": st["threshold"], "is_best": "yes" if is_best else "no", "total_gt": st["total_gt"], "total_pred": st["total_pred"], "matched": st["matched"], "false_positives": st["false_positives"], "false_negatives": st["false_negatives"], "precision": round(st["precision"], 4), "recall": round(st["recall"], 4), "f1": round(st["f1"], 4), "mean_iou": round(st["mean_iou"], 4), "median_iou": round(st["median_iou"], 4), "mean_latency_ms": round(st["mean_latency_ms"], 3), "median_latency_ms": round(st["median_latency_ms"], 3), "std_latency_ms": round(st["std_latency_ms"], 3), "p95_latency_ms": round(st["p95_latency_ms"], 3), "fps": round(st["fps"], 4), "decision_accuracy": round(st["decision_accuracy"], 4), "false_safe_decisions": st["false_safe_decisions"], "unnecessary_stop_decisions": st["unnecessary_stop_decisions"], "map50_subset": round(map50, 4) if is_best else "", "f1_ci95_low": round(lo, 4) if is_best else "", "f1_ci95_high": round(hi, 4) if is_best else ""})
            for cls in TARGET:
                c = st["class_counts"][cls]
                p = c["match"] / c["pred"] if c["pred"] else 0.0
                r = c["match"] / c["gt"] if c["gt"] else 0.0
                f = 2 * p * r / (p + r) if p + r else 0.0
                class_rows.append({"model": model_name, "model_label": out["label"], "threshold": st["threshold"], "is_best": "yes" if is_best else "no", "class": cls, "gt": c["gt"], "pred": c["pred"], "matched": c["match"], "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "mean_iou": round(statistics.mean(c["ious"]), 4) if c["ious"] else 0.0})
            for size in SIZES:
                c = st["size_counts"][size]
                p = c["match"] / c["pred"] if c["pred"] else 0.0
                r = c["match"] / c["gt"] if c["gt"] else 0.0
                f = 2 * p * r / (p + r) if p + r else 0.0
                size_rows.append({"model": model_name, "model_label": out["label"], "threshold": st["threshold"], "is_best": "yes" if is_best else "no", "size_bin": size, "gt": c["gt"], "pred": c["pred"], "matched": c["match"], "precision_by_box_size": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "mean_iou": round(statistics.mean(c["ious"]), 4) if c["ious"] else 0.0})
            if is_best:
                for (actual, predicted), count in st["confusion"].items():
                    confusion_rows.append({"model": model_name, "model_label": out["label"], "threshold": st["threshold"], "actual_state": actual, "predicted_state": predicted, "count": count})
                for er in st["errors"]:
                    row = dict(er); row["model"] = model_name; row["threshold"] = st["threshold"]; error_rows.append(row)

    robust_rows = []
    if args.robustness_count > 0:
        robust_selected = selected[: min(args.robustness_count, len(selected))]
        for model_name in args.robustness_models:
            if model_name not in outputs:
                continue
            threshold = float(best_by_model[model_name]["threshold"])
            for condition in ["brightness_low", "brightness_high", "blur", "noise", "occlusion"]:
                res = infer(model_name, robust_selected, name_to_id, args.min_score, args.threads, condition)
                st = aggregate(res["predictions"], robust_selected, by_image, threshold, res["latencies"])
                robust_rows.append({"model": model_name, "model_label": res["label"], "condition": condition, "threshold": threshold, "sample_images": len(robust_selected), "precision": round(st["precision"], 4), "recall": round(st["recall"], 4), "f1": round(st["f1"], 4), "mean_iou": round(st["mean_iou"], 4), "mean_latency_ms": round(st["mean_latency_ms"], 3), "false_positives": st["false_positives"], "false_negatives": st["false_negatives"]})

    write_csv(TAB / "extended_threshold_metrics.csv", th_rows)
    write_csv(TAB / "extended_per_class_metrics.csv", class_rows)
    write_csv(TAB / "extended_size_metrics.csv", size_rows)
    write_csv(TAB / "extended_decision_confusion_matrix.csv", confusion_rows)
    write_csv(TAB / "extended_error_cases.csv", error_rows)
    write_csv(TAB / "extended_ap50_by_class.csv", ap_rows)
    write_csv(TAB / "extended_map50_summary.csv", map_rows)
    write_csv(TAB / "extended_robustness_metrics.csv", robust_rows)
    make_figures(th_rows, class_rows, size_rows, robust_rows, confusion_rows)
    if "high_accuracy" in outputs:
        error_figure("high_accuracy", float(best_by_model["high_accuracy"]["threshold"]), outputs["high_accuracy"]["predictions"], selected, by_image)
    summary = {"sample_profile": profile, "models": {m: {k: v for k, v in o.items() if k not in {"predictions", "latencies"}} for m, o in outputs.items()}, "thresholds": args.thresholds, "best_by_model": {m: {"threshold": s["threshold"], "precision": round(s["precision"], 4), "recall": round(s["recall"], 4), "f1": round(s["f1"], 4), "mean_iou": round(s["mean_iou"], 4), "mean_latency_ms": round(s["mean_latency_ms"], 3), "fps": round(s["fps"], 4), "decision_accuracy": round(s["decision_accuracy"], 4)} for m, s in best_by_model.items()}, "tables": ["extended_threshold_metrics.csv", "extended_per_class_metrics.csv", "extended_size_metrics.csv", "extended_decision_confusion_matrix.csv", "extended_error_cases.csv", "extended_ap50_by_class.csv", "extended_map50_summary.csv", "extended_robustness_metrics.csv"], "figures": [p.name for p in sorted(FIG.glob("figure_4_*"))], "final_findings_ready": True}
    (LOG / "extended_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

