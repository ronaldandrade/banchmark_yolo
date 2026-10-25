"""Avalia detectores clássicos (torchvision + DETR/HuggingFace) no COCO val2017.

Resolução nativa de cada modelo (Faster R-CNN/RetinaNet/FCOS/DETR: lado menor 800; SSD300: 300;
SSDlite: 320). FP16 = torch.autocast (precisão mista), pois NMS/RoIAlign em FP16 puro não é suportado
de forma confiável pelo torchvision. Latência total = leitura + pré + inferência + pós, batch 1.

Uso: python -m src.benchmark.eval_classic --model fasterrcnn_resnet50_fpn [--half] [--limit 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms.functional import to_tensor
from torchvision.models import detection as det

ROOT = Path(__file__).resolve().parents[2]
COCO_DIR = ROOT / "data" / "coco"
THR = 0.001   # limiar baixo para o cálculo de mAP, como nos demais modelos

TV = {  # nome -> (construtor, kwargs de limiar, resolução nativa)
    "fasterrcnn_resnet50_fpn": (det.fasterrcnn_resnet50_fpn, dict(box_score_thresh=THR), 800),
    "fasterrcnn_resnet50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2, dict(box_score_thresh=THR), 800),
    "retinanet_resnet50_fpn": (det.retinanet_resnet50_fpn, dict(score_thresh=THR), 800),
    "retinanet_resnet50_fpn_v2": (det.retinanet_resnet50_fpn_v2, dict(score_thresh=THR), 800),
    "fcos_resnet50_fpn": (det.fcos_resnet50_fpn, dict(score_thresh=THR), 800),
    "ssd300_vgg16": (det.ssd300_vgg16, dict(score_thresh=THR), 300),
    "ssdlite320_mobilenet_v3_large": (det.ssdlite320_mobilenet_v3_large, dict(score_thresh=THR), 320),
}
HF = {"detr_resnet50": ("facebook/detr-resnet-50", 800)}


def build(name):
    """Retorna (modelo, função predict(path, dev) -> (boxes xyxy, scores, cat_ids COCO), resolução)."""
    if name in TV:
        ctor, kw, res = TV[name]
        m = ctor(weights="DEFAULT", **kw).cuda().eval()   # DEFAULT = pesos COCO oficiais
        if name.startswith("fcos"):
            # O FCOS decodifica as caixas no dtype da regressão (FP16 sob autocast) e o batched_nms soma
            # classe*(coord_max+1) às caixas: em FP16 isso satura (>65504 -> inf) e perde precisão (passo de 32 px
            # acima de 32768), corrompendo o NMS das classes de índice alto (mAP 0,392 -> 0,310). NMS em FP32 corrige.
            import types
            from torchvision.models.detection import fcos as fcos_mod
            _nms, _clip = fcos_mod.box_ops.batched_nms, fcos_mod.box_ops.clip_boxes_to_image
            fcos_mod.box_ops = types.SimpleNamespace(
                batched_nms=lambda b, sc, l, t: _nms(b.float(), sc.float(), l, t), clip_boxes_to_image=_clip)

        def predict(path):
            img = to_tensor(Image.open(path).convert("RGB")).cuda(non_blocking=True)
            o = m([img])[0]
            return o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy()
        return m, predict, res
    from transformers import AutoImageProcessor, DetrForObjectDetection
    repo, res = HF[name]
    proc = AutoImageProcessor.from_pretrained(repo)
    m = DetrForObjectDetection.from_pretrained(repo).cuda().eval()

    def predict(path):
        img = Image.open(path).convert("RGB")
        inp = proc(images=img, return_tensors="pt").to("cuda")
        out = m(pixel_values=inp["pixel_values"])
        r = proc.post_process_object_detection(out, threshold=THR, target_sizes=[img.size[::-1]])[0]
        return r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy()
    return m, predict, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[*TV, *HF], required=True)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    gt = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    valid = set(gt.getCatIds())
    path = lambda iid: str(COCO_DIR / "val2017" / gt.loadImgs(iid)[0]["file_name"])

    model, predict, res = build(args.model)
    ctx = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=args.half)

    with torch.inference_mode(), ctx():
        for _ in range(args.warmup):
            predict(path(img_ids[0]))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        dets, total = [], []
        for iid in img_ids:
            p = path(iid)
            t0 = time.perf_counter()
            boxes, scores, labels = predict(p)   # .cpu() já sincroniza
            torch.cuda.synchronize()
            total.append((time.perf_counter() - t0) * 1000)
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
                if int(c) in valid:              # modelos COCO usam o id de categoria (1-90) direto
                    dets.append({"image_id": iid, "category_id": int(c),
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})

    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    out = {
        "model": args.model, "imgsz": res, "half": args.half, "n_images": len(img_ids),
        "mAP50-95": ev.stats[0], "mAP50": ev.stats[1], "mAP75": ev.stats[2],
        "AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5], "AR100": ev.stats[8],
        "latency_ms": {"total": {"median": float(np.median(total)), "p95": float(np.percentile(total, 95))}},
        "fps_total": 1000 / float(np.median(total)),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "precision_mode": "autocast" if args.half else "fp32",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    o = Path(args.out); o.mkdir(exist_ok=True)
    f = o / f"{args.model}_{res}{'_fp16' if args.half else ''}.json"
    f.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("salvo em", f)


if __name__ == "__main__":
    main()
