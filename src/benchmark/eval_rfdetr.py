"""Avalia um RF-DETR no COCO val2017 com pycocotools e mede latência.

O RF-DETR roda na resolução nativa de cada variante (Nano 384, Small 512, Medium 576, Base 560),
não em 640. Não há NMS: a latência 'total' inclui pré/pós-processamento do próprio predict().

Uso: python -m src.benchmark.eval_rfdetr --variant small [--half] [--limit 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import rfdetr

ROOT = Path(__file__).resolve().parents[2]
COCO_DIR = ROOT / "data" / "coco"
VARIANTS = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
            "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    gt = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    valid_cats = set(gt.getCatIds())

    model = VARIANTS[args.variant]()
    res = model.model.resolution
    # compile=False: PyTorch eager, comparável ao restante do benchmark (sem TorchScript/TensorRT)
    model.inference(compile=False, dtype=torch.float16 if args.half else torch.float32)
    kw = dict(threshold=0.001, include_source_image=False)

    def path(iid):
        return str(COCO_DIR / "val2017" / gt.loadImgs(iid)[0]["file_name"])

    for _ in range(args.warmup):
        model.predict(path(img_ids[0]), **kw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    dets, total = [], []
    for iid in img_ids:
        p = path(iid)
        t0 = time.perf_counter()
        d = model.predict(p, **kw)
        torch.cuda.synchronize()
        total.append((time.perf_counter() - t0) * 1000)
        for (x1, y1, x2, y2), s, c in zip(d.xyxy, d.confidence, d.class_id):
            if int(c) not in valid_cats:   # checkpoints COCO devolvem o id de categoria (1-90) direto
                continue
            dets.append({"image_id": iid, "category_id": int(c),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})

    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    net = model.model.model
    n_params = sum(p.numel() for p in net.parameters())
    name = f"rfdetr-{args.variant}"
    out = {
        "model": name, "imgsz": res, "half": args.half, "n_images": len(img_ids),
        "mAP50-95": ev.stats[0], "mAP50": ev.stats[1], "mAP75": ev.stats[2],
        "AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5], "AR100": ev.stats[8],
        "latency_ms": {"total": {"median": float(np.median(total)), "p95": float(np.percentile(total, 95))}},
        "fps_total": 1000 / float(np.median(total)),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "params_M": n_params / 1e6,
        "n_dets_per_img": len(dets) / len(img_ids),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    o = Path(args.out); o.mkdir(exist_ok=True)
    f = o / f"{name}_{res}{'_fp16' if args.half else ''}.json"
    f.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("salvo em", f)


if __name__ == "__main__":
    main()
