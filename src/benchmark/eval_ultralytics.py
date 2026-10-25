"""Avalia um modelo Ultralytics no COCO val2017 com pycocotools e mede latência.

Uso: python -m src.benchmark.eval_ultralytics --weights yolo11n.pt [--limit 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
COCO_DIR = ROOT / "data" / "coco"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = todas as imagens")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    gt = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    # Ultralytics devolve índices 0..79; COCO usa ids esparsos 1..90
    cat_ids = sorted(gt.getCatIds())

    model = YOLO(args.weights)
    kw = dict(imgsz=args.imgsz, conf=0.001, iou=0.65, half=args.half,
              device=0, verbose=False, max_det=300)

    first = str(COCO_DIR / "val2017" / gt.loadImgs(img_ids[0])[0]["file_name"])
    for _ in range(args.warmup):
        model.predict(first, **kw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    dets, pre, inf, post, total = [], [], [], [], []
    for iid in img_ids:
        path = str(COCO_DIR / "val2017" / gt.loadImgs(iid)[0]["file_name"])
        t0 = time.perf_counter()
        r = model.predict(path, **kw)[0]
        torch.cuda.synchronize()
        total.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        b = r.boxes
        if len(b):
            xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                dets.append({"image_id": iid, "category_id": cat_ids[c],
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})

    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    n_params = sum(p.numel() for p in model.model.parameters())
    res = {
        "model": args.weights, "imgsz": args.imgsz, "half": args.half, "n_images": len(img_ids),
        "mAP50-95": ev.stats[0], "mAP50": ev.stats[1], "mAP75": ev.stats[2],
        "AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5], "AR100": ev.stats[8],
        "latency_ms": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                       for k, v in dict(pre=pre, inference=inf, post=post, total=total).items()},
        "fps_total": 1000 / float(np.median(total)),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "params_M": n_params / 1e6,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    out = Path(args.out); out.mkdir(exist_ok=True)
    f = out / f"{Path(args.weights).stem}_{args.imgsz}{'_fp16' if args.half else ''}.json"
    f.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print("salvo em", f)


if __name__ == "__main__":
    main()
