# Project 7 Code

This folder contains reproducible technical-evaluation code for the computer-vision dissertation.

Current expanded entrypoint:

```powershell
python code/run_extended_coco_evaluation.py --sample-count 200 --seed 20260909 --models high_accuracy lightweight --robustness-count 20
```

The expanded evaluator:

- samples COCO validation images with a fixed seed;
- keeps the decision-relevant classes: person, bicycle, car, motorcycle, bus, truck, traffic light and stop sign;
- includes background images so decision-state evaluation can produce proceed, inspect, slow and stop outcomes;
- keeps small, medium and large target objects instead of excluding objects below 2,500 pixels;
- evaluates Faster R-CNN ResNet50-FPN v2 and SSDlite320 MobileNetV3-Large;
- computes threshold metrics, per-class metrics, AP50/mAP50 on the dissertation subset, object-size metrics, robustness metrics, latency statistics, decision-state confusion matrices and error cases;
- exports results to `outputs/tables`, figures to `outputs/figures` and logs to `outputs/logs`.

The script evaluates all confidence thresholds from one inference pass per model. This avoids rerunning a detector for every threshold.
