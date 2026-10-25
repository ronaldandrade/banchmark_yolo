"""Consolida results/*.json em results/tabela_benchmark.{csv,md}."""
from src.benchmark.common import RESULTS, load

df = load().sort_values(["Precisão", "Família", "Params (M)"]).reset_index(drop=True)
df.to_csv(RESULTS / "tabela_benchmark.csv", index=False)
fmt = df.drop(columns="Rótulo").round({"Params (M)": 2, "mAP50-95": 3, "mAP50": 3, "AP_S": 3, "AP_M": 3, "AP_L": 3,
        "Inferência (ms)": 2, "Total (ms)": 2, "P95 total (ms)": 2, "FPS": 0, "VRAM pico (MB)": 0})
fmt = fmt.astype(object).where(fmt.notna(), "—")
out = ["Latência 'Total' = leitura + pré-processamento + inferência + pós-processamento, mediana, batch 1. "
       "'Inferência' só existe para modelos Ultralytics (—nos demais). Modelos que não são YOLO/RT-DETR rodam na "
       "resolução nativa (coluna Entrada; lado menor para os de 800). FP16: `half` real nos Ultralytics e RF-DETR; "
       "`torch.autocast` (precisão mista) em torchvision e DETR.\n"]
for p in ("FP32", "FP16"):
    out.append(f"### {p}\n\n" + fmt[fmt["Precisão"] == p].drop(columns="Precisão")
               .to_markdown(index=False))
(RESULTS / "tabela_benchmark.md").write_text("\n\n".join(out) + "\n")
print(f"{len(df)} linhas -> results/tabela_benchmark.md")
