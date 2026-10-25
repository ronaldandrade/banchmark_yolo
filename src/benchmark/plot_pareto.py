"""Pareto mAP50-95 x latência total (FP32 e FP16), todos os modelos de results/*.json."""
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D

from src.benchmark.common import load

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# cor = grupo (3 slots, validado --pairs all); formato = família
GROUP_COLOR = {"YOLO (Ultralytics)": "#2a78d6", "Transformers (DETR, RT-DETR, RF-DETR)": "#eb6834",
               "Clássicos CNN (torchvision)": "#1baf7a"}
GROUP = {"RT-DETR": "Transformers (DETR, RT-DETR, RF-DETR)", "RF-DETR": "Transformers (DETR, RT-DETR, RF-DETR)",
         "DETR": "Transformers (DETR, RT-DETR, RF-DETR)", "Faster R-CNN": "Clássicos CNN (torchvision)",
         "RetinaNet": "Clássicos CNN (torchvision)", "FCOS": "Clássicos CNN (torchvision)",
         "SSD": "Clássicos CNN (torchvision)"}
MARKER = {"YOLOv8": "o", "YOLOv9": "s", "YOLOv10": "^", "YOLO11": "D", "YOLO12": "P",
          "RT-DETR": "h", "RF-DETR": "*", "DETR": "p",
          "Faster R-CNN": "v", "RetinaNet": "<", "FCOS": ">", "SSD": "X"}
SIZE = {"RF-DETR": 200}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def group(fam):
    return GROUP.get(fam, "YOLO (Ultralytics)")


def frontier(d):
    keep, best = [], -1
    for i, r in d.sort_values("Total (ms)").iterrows():
        if r["mAP50-95"] > best:
            keep.append(i); best = r["mAP50-95"]
    return keep


df = load()
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, facecolor=SURFACE)
for ax, prec in zip(axes, ("FP32", "FP16")):
    ax.set_facecolor(SURFACE)
    d = df[df["Precisão"] == prec]
    fr = frontier(d)
    f = d.loc[fr].sort_values("Total (ms)")
    ax.plot(f["Total (ms)"], f["mAP50-95"] * 100, color=INK, lw=2, alpha=0.5, zorder=1)
    texts = []
    for i, r in d.iterrows():
        ax.scatter(r["Total (ms)"], r["mAP50-95"] * 100, s=SIZE.get(r["Família"], 80), marker=MARKER[r["Família"]],
                   color=GROUP_COLOR[group(r["Família"])], edgecolor=SURFACE, linewidth=1.3, zorder=3)
        texts.append(ax.text(r["Total (ms)"], r["mAP50-95"] * 100, r["Rótulo"], fontsize=8,
                             color=INK if i in fr else INK2, fontweight="bold" if i in fr else "normal"))
    adjust_text(texts, ax=ax, expand=(1.3, 1.5), arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    ax.set_title(prec, loc="left", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("Latência total por imagem (ms, mediana, escala log)")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("mAP@[.5:.95] (%) — COCO val2017")

hs = [Line2D([], [], marker="o", ls="", color=c, markersize=9, label=g) for g, c in GROUP_COLOR.items()]
hs += [Line2D([], [], marker=m, ls="", color=INK2, markersize=8, label=fam) for fam, m in MARKER.items()]
hs += [Line2D([], [], color=INK, lw=2, alpha=0.5, label="Fronteira de Pareto")]
fig.legend(handles=hs, loc="lower center", ncol=7, frameon=False, fontsize=9, labelcolor=INK2,
           handletextpad=0.3, columnspacing=1.2)
fig.suptitle("Detecção no COCO val2017 — acurácia × latência (RTX 3050 Laptop, batch 1)",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.11, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"pareto_benchmark.{ext}", dpi=200, facecolor=SURFACE)
for prec in ("FP32", "FP16"):
    d = df[df["Precisão"] == prec]
    print(prec, "fronteira:", " → ".join(d.loc[frontier(d)].sort_values("Total (ms)")["Rótulo"]))
