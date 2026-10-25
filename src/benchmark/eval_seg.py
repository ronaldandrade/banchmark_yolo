"""Segmentação de instância no COCO val2017 (mask AP + box AP) e latência.

Backends: Ultralytics (-seg), torchvision Mask R-CNN, RF-DETR-Seg.
A codificação RLE das máscaras e a avaliação ficam FORA da medição de latência.
No máximo 100 detecções por imagem (maxDets do COCO), limiar 0,001.

Uso: python -m src.benchmark.eval_seg --model yolo11n-seg [--half] [--limit 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
COCO_DIR = ROOT / "data" / "coco"
THR, MAXDET = 0.001, 100
RFDETR_SEG = {"rfdetr-seg-nano": "RFDETRSegNano", "rfdetr-seg-small": "RFDETRSegSmall",
              "rfdetr-seg-medium": "RFDETRSegMedium", "rfdetr-seg-large": "RFDETRSegLarge"}
TV_SEG = {"maskrcnn_resnet50_fpn": "maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2": "maskrcnn_resnet50_fpn_v2"}


def build(name, half, cat_ids):
    """-> (modelo/nn.Module p/ contar params, predict(path) -> (xyxy, scores, cat_ids, masks bool N×H×W), res, usa_autocast)."""
    if name in TV_SEG:
        from torchvision.models import detection as det
        from torchvision.transforms.functional import to_tensor
        m = getattr(det, TV_SEG[name])(weights="DEFAULT", box_score_thresh=THR, box_detections_per_img=MAXDET).cuda().eval()

        def predict(path):
            o = m([to_tensor(Image.open(path).convert("RGB")).cuda()])[0]
            return (o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(),
                    o["labels"].cpu().numpy(), (o["masks"][:, 0] > 0.5).cpu().numpy())
        return m, predict, 800, half

    if name in RFDETR_SEG:
        import rfdetr
        w = getattr(rfdetr, RFDETR_SEG[name])()
        w.inference(compile=False, dtype=torch.float16 if half else torch.float32)
        valid = set(cat_ids)

        def predict(path):
            d = w.predict(path, threshold=THR, include_source_image=False)
            k = np.array([int(c) in valid for c in d.class_id], dtype=bool)
            return d.xyxy[k], d.confidence[k], d.class_id[k], d.mask[k]
        return w.model.model, predict, w.model.resolution, False

    from ultralytics import YOLO
    y = YOLO(f"{name}.pt")
    cats = np.array(cat_ids)

    def predict(path):
        r = y.predict(path, imgsz=640, conf=THR, iou=0.65, half=half, device=0, verbose=False,
                      max_det=MAXDET, retina_masks=True)[0]
        if r.masks is None or not len(r.boxes):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 1, 1), bool)
        return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                cats[r.boxes.cls.cpu().numpy().astype(int)], r.masks.data.cpu().numpy() > 0.5)
    return y.model, predict, 640, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "seg"))
    args = ap.parse_args()

    gt = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    cat_ids = sorted(gt.getCatIds())
    path = lambda iid: str(COCO_DIR / "val2017" / gt.loadImgs(iid)[0]["file_name"])

    model, predict, res, autocast = build(args.model, args.half, cat_ids)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        for _ in range(args.warmup):
            predict(path(img_ids[0]))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        raw, total = [], []
        for iid in img_ids:
            t0 = time.perf_counter()
            out = predict(path(iid))
            torch.cuda.synchronize()
            total.append((time.perf_counter() - t0) * 1000)
            raw.append((iid, out))

    dets = []
    for iid, (boxes, scores, labels, masks) in raw:   # RLE fora da medição
        for (x1, y1, x2, y2), s, c, m in zip(boxes, scores, labels, masks):
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            dets.append({"image_id": iid, "category_id": int(c), "score": float(s), "segmentation": rle,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})

    stats = {}
    for typ in ("segm", "bbox"):
        ev = COCOeval(gt, gt.loadRes(dets), typ)
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats[typ] = ev.stats

    s, b = stats["segm"], stats["bbox"]
    result = {
        "model": args.model, "imgsz": res, "half": args.half, "n_images": len(img_ids),
        "mask_mAP50-95": s[0], "mask_mAP50": s[1], "mask_mAP75": s[2],
        "mask_AP_small": s[3], "mask_AP_medium": s[4], "mask_AP_large": s[5],
        "box_mAP50-95": b[0], "box_mAP50": b[1],
        "latency_ms": {"total": {"median": float(np.median(total)), "p95": float(np.percentile(total, 95))}},
        "fps_total": 1000 / float(np.median(total)),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    f = o / f"{args.model}_{res}{'_fp16' if args.half else ''}.json"
    f.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print("salvo em", f)


if __name__ == "__main__":
    main()
