"""Mede o erro da decodificação de caixas do FCOS em FP16 vs FP32 (mesmos regs e âncoras)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models import detection as det
from torchvision.models.detection import _utils as U
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
gt = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
ids = sorted(gt.getImgIds())[:100]
m = det.fcos_resnet50_fpn(weights="DEFAULT", score_thresh=0.2).cuda().eval()   # só detecções relevantes

captured = []
orig = U.BoxLinearCoder.decode
def spy(self, rel_codes, boxes):
    captured.append((rel_codes.detach().float().clone(), boxes.detach().float().clone()))
    return orig(self, rel_codes, boxes)
U.BoxLinearCoder.decode = spy

with torch.inference_mode():
    for i in ids:
        m([to_tensor(Image.open(ROOT / "data/coco/val2017" / gt.loadImgs(i)[0]["file_name"]).convert("RGB")).cuda()])
U.BoxLinearCoder.decode = orig

rel = torch.cat([r for r, _ in captured]); anc = torch.cat([b for _, b in captured])
ref = orig(m.box_coder, rel, anc)                         # FP32
h = orig(m.box_coder, rel.half(), anc)                    # FP16 (âncoras viram half dentro do decode)
err = (h.float() - ref).abs()
def pair_iou(a, b):
    lt = torch.max(a[:, :2], b[:, :2]); rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(1)
    return inter / (area(a) + area(b) - inter).clamp(min=1e-6)
iou = pair_iou(ref, h.float())
print(f"caixas: {len(ref)}  (score>0.2, 100 imagens)")
print("erro abs por coordenada (px): mediana %.3f  P95 %.3f  P99 %.3f  máx %.2f" % tuple(
    np.percentile(err.flatten().cpu().numpy(), [50, 95, 99, 100])))
print("âncora: coord máx %.0f px;  espaçamento do FP16 em [512,1024) = 0.5 px, em [1024,2048) = 1 px" % anc.max())
print("IoU fp16 vs fp32 : mediana %.3f  P5 %.3f  fração <0.5: %.3f  <0.9: %.3f" % (
    iou.median(), iou.quantile(0.05), (iou < 0.5).float().mean(), (iou < 0.9).float().mean()))
big = ((ref[:, 2] - ref[:, 0]) * (ref[:, 3] - ref[:, 1])).sqrt() > 96
e = (h.float() - ref).abs().max(1).values
print("erro máx por caixa: objetos grandes (>96px) mediana %.2f | pequenos mediana %.2f" % (e[big].median(), e[~big].median()))
