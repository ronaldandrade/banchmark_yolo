"""Carrega results/*.json em um DataFrame único (YOLO, RT-DETR, RF-DETR)."""
import json
import re
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"


def parse(stem):
    """stem -> (família, porte, rótulo curto)."""
    if m := re.fullmatch(r"yolo(v?)(\d+)([ntsm])", stem):
        fam = f"YOLOv{m[2]}" if m[1] else f"YOLO{m[2]}"
        return fam, m[3], f"{'v' if m[1] else ''}{m[2]}{m[3]}"
    if m := re.fullmatch(r"rtdetr-([lx])", stem):
        return "RT-DETR", m[1], f"rt-{m[1]}"
    if m := re.fullmatch(r"rfdetr-(\w+)", stem):
        return "RF-DETR", m[1], f"rf-{m[1][0]}"
    classic = {  # stem -> (família, porte, rótulo)
        "fasterrcnn_resnet50_fpn": ("Faster R-CNN", "r50", "frcnn"),
        "fasterrcnn_resnet50_fpn_v2": ("Faster R-CNN", "r50-v2", "frcnn-v2"),
        "retinanet_resnet50_fpn": ("RetinaNet", "r50", "retina"),
        "retinanet_resnet50_fpn_v2": ("RetinaNet", "r50-v2", "retina-v2"),
        "fcos_resnet50_fpn": ("FCOS", "r50", "fcos"),
        "ssd300_vgg16": ("SSD", "vgg16", "ssd300"),
        "ssdlite320_mobilenet_v3_large": ("SSD", "mnv3", "ssdlite"),
        "detr_resnet50": ("DETR", "r50", "detr"),
    }
    return classic[stem]


def load():
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        fam, size, short = parse(Path(d["model"]).stem)
        lat = d["latency_ms"]
        rows.append({
            "Família": fam, "Modelo": Path(d["model"]).stem, "Rótulo": short,
            "Precisão": "FP16" if d["half"] else "FP32", "Entrada": d["imgsz"],
            "Params (M)": d["params_M"], "mAP50-95": d["mAP50-95"], "mAP50": d["mAP50"],
            "AP_S": d["AP_small"], "AP_M": d["AP_medium"], "AP_L": d["AP_large"],
            "Inferência (ms)": lat.get("inference", {}).get("median"),   # ausente no RF-DETR
            "Total (ms)": lat["total"]["median"], "P95 total (ms)": lat["total"]["p95"],
            "FPS": d["fps_total"], "VRAM pico (MB)": d["peak_gpu_mem_mb"],
        })
    return pd.DataFrame(rows)
