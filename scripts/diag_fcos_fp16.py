"""Diagnóstico: por que o FCOS perde mAP em FP16 (autocast)? Compara variantes em N imagens do val2017."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models import detection as det
from torchvision.transforms.functional import to_tensor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
ROOT = Path(__file__).resolve().parents[1]
gt = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
ids = sorted(gt.getImgIds())[:N]
valid = set(gt.getCatIds())
imgs = {i: to_tensor(Image.open(ROOT / "data/coco/val2017" / gt.loadImgs(i)[0]["file_name"]).convert("RGB")) for i in ids}


def fresh():
    return det.fcos_resnet50_fpn(weights="DEFAULT", score_thresh=0.001).cuda().eval()


def wrap_head(m, fn):
    orig = m.head.forward
    m.head.forward = lambda x: fn(orig, x)


def variant(name):
    m = fresh()
    if name == "fp32":
        return m, False
    if name == "autocast_all":
        return m, True
    if name == "head_fp32":      # backbone+FPN em fp16, cabeça inteira em fp32
        def f(orig, x):
            with torch.autocast("cuda", enabled=False):
                return orig([t.float() for t in x])
        wrap_head(m, f); return m, True
    if name == "head_out_fp32":  # cabeça em fp16, mas saídas convertidas p/ fp32 antes do decode/score
        def f(orig, x):
            return {k: v.float() for k, v in orig(x).items()}
        wrap_head(m, f); return m, True
    if name == "out_bbox_fp32":  # só a regressão das caixas em fp32; scores em fp16
        def f(orig, x):
            o = orig(x); o["bbox_regression"] = o["bbox_regression"].float(); return o
        wrap_head(m, f); return m, True
    if name == "out_scores_fp32":  # só cls_logits e centerness em fp32; caixas em fp16
        def f(orig, x):
            o = orig(x); o["cls_logits"] = o["cls_logits"].float(); o["bbox_ctrness"] = o["bbox_ctrness"].float(); return o
        wrap_head(m, f); return m, True
    if name == "nms_fp32":       # tudo em fp16, exceto o NMS (caixas convertidas para fp32 só dentro dele)
        from torchvision.models.detection import fcos as F
        real = F.box_ops.batched_nms
        class Ops:                # proxy: troca só batched_nms, mantém o resto de box_ops
            def __getattr__(self, k): return getattr(F.box_ops, k)
            batched_nms = staticmethod(lambda b, sc, l, t: real(b.float(), sc.float(), l, t))
        F.box_ops = type("P", (), {"batched_nms": staticmethod(lambda b, sc, l, t: real(b.float(), sc.float(), l, t)),
                                   "clip_boxes_to_image": F.box_ops.clip_boxes_to_image})
        return m, True
    if name == "backbone_fp32":  # backbone+FPN em fp32, cabeça em fp16
        orig_bb = m.backbone.forward
        def bb(x):
            with torch.autocast("cuda", enabled=False):
                return orig_bb(x.float())
        m.backbone.forward = bb; return m, True


for name in sys.argv[2].split(",") if len(sys.argv) > 2 else ["fp32", "autocast_all", "head_fp32", "head_out_fp32", "backbone_fp32"]:
    m, ac = variant(name)
    dets, n_dets = [], 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=ac):
        for i in ids:
            o = m([imgs[i].cuda()])[0]
            b, s, l = o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy()
            n_dets += len(s)
            for (x1, y1, x2, y2), sc, c in zip(b, s, l):
                if int(c) in valid:
                    dets.append({"image_id": i, "category_id": int(c), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(sc)})
    ev = COCOeval(gt, gt.loadRes(dets), "bbox"); ev.params.imgIds = ids
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"{name:15s} mAP={ev.stats[0]:.4f} mAP50={ev.stats[1]:.4f} AP_S={ev.stats[3]:.4f} dets/img={n_dets/len(ids):.1f}", flush=True)
    del m; torch.cuda.empty_cache()
