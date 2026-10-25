# Benchmark de Modelos de Visão Computacional: Detecção de Objetos e Segmentação

> Documento-base do projeto. Serve como especificação metodológica e, futuramente, como esqueleto do artigo científico.
> Status: **v0.2 — detecção de objetos concluída (28 modelos); segmentação e ablações pendentes** (ver §14 e §15).
> Decisões marcadas com ⚠️ ainda precisam ser confirmadas.

## 1. Motivação

A família YOLO e arquiteturas baseadas em Transformers (DETR, RT-DETR, RF-DETR) evoluem rapidamente, e os resultados publicados raramente são comparáveis entre si: cada artigo usa hardware, resolução de entrada, pós-processamento (NMS) e protocolo de medição de latência diferentes. Este projeto propõe um benchmark **reprodutível, sob hardware e protocolo únicos**, que mede simultaneamente **acurácia** e **custo computacional** de modelos de detecção e segmentação.

## 2. Objetivos

**Geral:** comparar, sob condições controladas, os principais modelos de detecção de objetos e segmentação (de instância e semântica), avaliando o compromisso entre acurácia e eficiência.

**Específicos:**
1. Avaliar acurácia (mAP e variantes) de cada modelo em conjuntos de dados padrão.
2. Medir latência, throughput, uso de memória de GPU, número de parâmetros e FLOPs.
3. Analisar o desempenho por tamanho de objeto (pequeno/médio/grande) e por classe.
4. Avaliar o impacto de resolução de entrada, precisão numérica (FP32/FP16) e exportação (PyTorch vs ONNX/TensorRT).
5. Disponibilizar código e resultados de forma que terceiros reproduzam o estudo.

## 3. Perguntas de pesquisa

- **RQ1.** Qual a fronteira de Pareto acurácia × latência entre as famílias avaliadas?
- **RQ2.** Detectores *end-to-end* (sem NMS), como YOLOv10 e RT-DETR, compensam em latência real frente aos baseados em NMS?
- **RQ3.** Como o desempenho degrada em objetos pequenos e com resolução reduzida?
- **RQ4.** Qual o ganho de FP16 e de exportação (ONNX/TensorRT) sobre a inferência nativa em PyTorch, e a que custo de acurácia?
- **RQ5.** Em segmentação, quanto custa adicionar a cabeça de máscara em relação à detecção pura?

## 4. Escopo

### 4.1 Detecção de objetos

| Família | Modelos candidatos | Paradigma |
|---|---|---|
| YOLO (Ultralytics) | YOLOv8, YOLO11, YOLO12 (n/s/m) | CNN, *anchor-free*, NMS |
| YOLOv9 / YOLOv10 | YOLOv9 (t/s/m), YOLOv10 (n/s/m) | CNN; v10 *NMS-free* |
| RT-DETR | RT-DETR-L (Ultralytics) ¹ | Transformer híbrido, *end-to-end* |
| RF-DETR | RF-DETR Nano/Small/Base | Transformer (backbone DINOv2) |
| Faster R-CNN | ResNet50-FPN (torchvision) | Dois estágios (referência clássica) |
| RetinaNet / SSD | torchvision | Um estágio (referência clássica) |
| DETR | ResNet-50 (HuggingFace) | Transformer (referência histórica) |
| FCOS | ResNet50-FPN (torchvision) | Um estágio, *anchor-free* (adicionado) |

¹ A Ultralytics só distribui RT-DETR-L e -X; o X não cabe em 4 GB. As variantes R18/R50 (HuggingFace) **não foram avaliadas**.

**Executado:** todos os acima, exceto RT-DETR-R18/R50. Variantes efetivamente avaliadas: YOLOv8/9/10/11/12 nos portes n/t, s, m; RT-DETR-L; RF-DETR Nano/Small/Medium/Base; Faster R-CNN, RetinaNet (v1 e v2), FCOS, SSD300, SSDlite320, DETR-R50. Os portes l/x dos YOLO ficaram de fora (4 GB de VRAM).

### 4.2 Segmentação

| Tipo | Modelos candidatos |
|---|---|
| Instância | YOLOv8-seg, YOLO11-seg, Mask R-CNN (torchvision), RF-DETR-Seg (se disponível) |
| Semântica | SegFormer (B0–B2), DeepLabV3+ (MobileNet/ResNet50) |
| Promptable (qualitativo) | SAM / SAM2 (apenas comparação de custo e qualidade visual) |

**Status: não executada.** Os adaptadores (`src/benchmark/eval_seg.py`, `scripts/run_seg.sh`) foram escritos para segmentação de instância (mask AP e box AP), mas **nunca foram testados nem executados**. A segmentação semântica exige outro dataset (VOC/ADE20K) e não foi iniciada.

> ⚠️ A lista final será fechada após verificar disponibilidade de pesos e licenças (ver §10), priorizando variantes que caibam em 4 GB de VRAM.

## 5. Conjuntos de dados

| Dataset | Tarefa | Uso |
|---|---|---|
| **COCO 2017 val** (5.000 imagens, 80 classes) | Detecção + segmentação de instância | Avaliação principal (modelos usam pesos pré-treinados em COCO train) |
| **Pascal VOC 2012 val** | Detecção / segmentação semântica | Avaliação complementar |
| ⚠️ Dataset de domínio específico (a definir) | Detecção | Teste de generalização e *fine-tuning* curto |

Observação: como a maioria dos modelos é pré-treinada em COCO train2017, a avaliação em val2017 é justa; a avaliação em outro domínio mede generalização.

## 6. Métricas

**Acurácia** (via `pycocotools`, protocolo COCO oficial):
- mAP@[.5:.95], mAP@.5, mAP@.75; AP por tamanho (S/M/L); AR@100.
- Segmentação de instância: mask AP. Semântica: mIoU.

**Eficiência:**
- Latência (ms/imagem), dividida em **pré-processamento, inferência e pós-processamento (NMS)**.
- Throughput (FPS) com *batch* 1 e *batch* maior.
- Pico de memória de GPU (MB), nº de parâmetros (M), FLOPs (G).

**Síntese:** gráficos de Pareto (mAP × latência) e tabelas por família.

## 7. Protocolo experimental

Para garantir comparabilidade:
1. **Hardware único** (ver §8); GPU em modo de desempenho fixo, sem outros processos pesados.
2. **Resolução padrão 640×640** (letterbox) para todos; estudo de sensibilidade com 320/480/800/1024.
3. **Limiares fixos:** `conf=0.001` e `iou(NMS)=0.65` para cálculo de mAP; `conf=0.25` para medições de latência típicas.
4. **Medição de latência:** 50 iterações de *warm-up* descartadas, ≥ 300 iterações medidas, `torch.cuda.synchronize()` antes de cada leitura de tempo, relatando mediana e P95 (e desvio-padrão).
5. **Precisão:** FP32 e FP16; exportação ONNX e TensorRT como estudo à parte.
6. **Sementes e versões** (Python, PyTorch, CUDA, bibliotecas) registradas em cada execução; resultados salvos em JSON/CSV junto ao hash de configuração.
7. **Pesos oficiais pré-treinados**, sem ajustes de hiperparâmetros por modelo, para evitar viés do avaliador.

### 7.1 Desvios do protocolo (como foi executado)

- **Resolução:** YOLO e RT-DETR em 640×640. Os demais rodam na **resolução nativa** de cada modelo: RF-DETR Nano 384 / Small 512 / Medium 576 / Base 560; Faster R-CNN, RetinaNet, FCOS e DETR com lado menor 800; SSD300 em 300; SSDlite em 320. O estudo de sensibilidade à resolução (320–1024) **não foi feito**.
- **Limiar:** `conf = 0,001` para todos (cálculo de mAP). `iou(NMS) = 0,65` nos Ultralytics. As medições com `conf = 0,25` **não foram feitas**.
- **Latência:** mediana e P95 do **tempo total por imagem** (leitura do arquivo + pré-processamento + inferência + pós-processamento), batch 1, com `cuda.synchronize()`. Warm-up de 50 iterações (Ultralytics/RF-DETR) ou 30 (torchvision/DETR); 5.000 imagens medidas (não ≥ 300 iterações repetidas). A quebra pré/inferência/pós só existe para os modelos Ultralytics.
- **FP16:** `half` real em Ultralytics e RF-DETR; **`torch.autocast` (precisão mista)** em torchvision e DETR, pois NMS/RoIAlign em FP16 puro não são suportados de forma confiável. A comparação FP16 entre famílias não é idêntica.
- **Medição única:** cada configuração foi medida **uma vez**. Não há barras de erro; diferenças pequenas (ordem de 0,3 ms) estão dentro do ruído.
- **Avaliação:** `pycocotools` oficial sobre as predições exportadas; um único avaliador para todas as famílias. Predições com `category_id` fora das 80 classes COCO são descartadas.
- **Não feito:** ONNX/TensorRT, FLOPs, estudo por classe, *fine-tuning*, outros datasets.

## 8. Ambiente experimental

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4 GB VRAM** |
| Driver | 535.230.02 (CUDA ≤ 12.2) |
| CPU / RAM | 16 threads / 7,4 GB |
| SO | Linux 5.15 (Ubuntu) |
| Python | 3.11 (venv isolado em `.venv/`) |
| PyTorch | 2.6.0 + CUDA 12.4 (wheels `cu124`; funcionam no driver 535 por compatibilidade de versão menor) |
| Bibliotecas | ultralytics 8.4.183, rfdetr 1.11.2, transformers 5.20.2, torchvision 0.21.0 (versões exatas em `requirements.lock.txt`) |

**Limitações conhecidas:** os 4 GB de VRAM e 7,4 GB de RAM restringem modelos grandes (L/X, SAM-H, DETR com *batch* alto) e treinamento. Isso é tratado como parte do estudo: o benchmark representa um cenário de **hardware de borda/consumidor**, e modelos que não couberem serão reportados como tal (OOM), não omitidos. Resultados não devem ser extrapolados para GPUs de data center.

## 9. Estrutura do repositório

```
banchmark_yolo/
├── PROJETO.md                 # este documento
├── requirements.txt / requirements.lock.txt
├── setup_env.sh               # cria o ambiente isolado (.venv)
├── data/coco/                 # COCO val2017 (fora do versionamento)
├── scripts/                   # download_coco.sh, run_yolo_all.sh, run_rtdetr.sh, run_rfdetr.sh,
│                              # run_classic.sh, run_seg.sh (não executado)
├── src/benchmark/
│   ├── eval_ultralytics.py    # YOLO e RT-DETR
│   ├── eval_rfdetr.py         # RF-DETR
│   ├── eval_classic.py        # torchvision + DETR (HuggingFace)
│   ├── eval_seg.py            # segmentação (não testado)
│   ├── common.py, make_table.py, plot_pareto.py
├── results/                   # um JSON por modelo/precisão + tabela_benchmark.{md,csv} + logs/
└── paper/figures/             # pareto_benchmark.{png,pdf}
```

## 10. Riscos e pontos em aberto

- **Licenças:** Ultralytics (YOLOv8/11/12) é AGPL-3.0; RF-DETR e RT-DETR possuem licenças distintas (verificar). Relevante para citação e eventual uso comercial.
- **Equivalência de pós-processamento:** NMS implementado por cada biblioteca difere; usar o avaliador COCO único sobre as predições brutas exportadas.
- **Vazamento:** confirmar que nenhum peso foi treinado em COCO val2017.
- **Memória:** modelos podem exigir *batch* 1 ou resolução menor; documentar cada exceção.
- **Reprodutibilidade de latência em notebook:** *thermal throttling*; registrar temperatura/clock e executar múltiplas rodadas.

## 11. Cronograma sugerido

| Fase | Entregável |
|---|---|
| 1. Setup | Ambiente, download de dados e pesos, verificação de GPU |
| 2. Pipeline | Harness de inferência + avaliador COCO unificado + timer |
| 3. Detecção | Execução e análise de todos os detectores |
| 4. Segmentação | Execução e análise de instância/semântica |
| 5. Ablações | Resolução, precisão, exportação |
| 6. Escrita | Figuras, tabelas e artigo |

## 12. Estrutura prevista do artigo

1. Introdução · 2. Trabalhos relacionados · 3. Metodologia (§4–§8 deste documento) · 4. Resultados (acurácia, eficiência, Pareto, por tamanho) · 5. Discussão e ameaças à validade · 6. Conclusão · Apêndice: configurações e comandos para reprodução.

## 13. Referências iniciais

- Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
- Redmon et al., *You Only Look Once*, CVPR 2016.
- Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020.
- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)*, CVPR 2024.
- Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024.
- Wang et al., *YOLOv9: Learning What You Want to Learn Using PGI*, 2024.
- Ren et al., *Faster R-CNN*, NeurIPS 2015; He et al., *Mask R-CNN*, ICCV 2017.
- Xie et al., *SegFormer*, NeurIPS 2021.
- Kirillov et al., *Segment Anything*, ICCV 2023.
- Robicheaux et al., *RF-DETR* (Roboflow), 2025.

## 14. Resultados (detecção, COCO val2017, execução de 21–23/10/2026)

Hardware: RTX 3050 Laptop (4 GB), batch 1. Detalhes e desvios de protocolo em §7.1. Latências = tempo total por imagem. Tabela completa (inclui P95, VRAM FP16, AP médio/grande) em `results/tabela_benchmark.md`; figura em `paper/figures/pareto_benchmark.pdf`.

| Modelo                        | Família      |   Entrada |   Params (M) |   mAP |   mAP50 |   AP_S |   ms FP32 |   ms FP16 |   mAP FP16 |   VRAM FP32 (MB) |
|:------------------------------|:-------------|----------:|-------------:|------:|--------:|-------:|----------:|----------:|-----------:|-----------------:|
| rfdetr-medium                 | RF-DETR      |       576 |         33.7 | 0.547 |   0.735 |  0.358 |      38.2 |      23.3 |      0.546 |              322 |
| rfdetr-base                   | RF-DETR      |       560 |         32.2 | 0.532 |   0.721 |  0.346 |      42.3 |      23.6 |      0.532 |              321 |
| rfdetr-small                  | RF-DETR      |       512 |         32.1 | 0.528 |   0.72  |  0.317 |      29.4 |      19.4 |      0.528 |              299 |
| rtdetr-l                      | RT-DETR      |       640 |         33   | 0.528 |   0.712 |  0.344 |      46.2 |      32.3 |      0.527 |              242 |
| yolo12m                       | YOLO12       |       640 |         20.2 | 0.514 |   0.684 |  0.345 |      24.4 |      16.6 |      0.514 |              191 |
| yolo11m                       | YOLO11       |       640 |         20.1 | 0.505 |   0.675 |  0.334 |      20.5 |      13.3 |      0.505 |              190 |
| yolov9m                       | YOLOv9       |       640 |         20.2 | 0.504 |   0.673 |  0.324 |      23.9 |      16.1 |      0.504 |              153 |
| yolov10m                      | YOLOv10      |       640 |         16.6 | 0.502 |   0.673 |  0.323 |      19.7 |      11.8 |      0.501 |              152 |
| yolov8m                       | YOLOv8       |       640 |         25.9 | 0.493 |   0.661 |  0.31  |      22   |      13.1 |      0.493 |              161 |
| rfdetr-nano                   | RF-DETR      |       384 |         30.5 | 0.48  |   0.671 |  0.25  |      17   |      15.3 |      0.48  |              269 |
| fasterrcnn_resnet50_fpn_v2    | Faster R-CNN |       800 |         43.7 | 0.469 |   0.679 |  0.31  |     158.3 |      93.6 |      0.469 |              671 |
| yolo12s                       | YOLO12       |       640 |          9.3 | 0.468 |   0.636 |  0.276 |      13.7 |       9   |      0.468 |              110 |
| yolov10s                      | YOLOv10      |       640 |          8.1 | 0.458 |   0.628 |  0.262 |      10.4 |       7.5 |      0.457 |               96 |
| yolo11s                       | YOLO11       |       640 |          9.5 | 0.456 |   0.626 |  0.278 |      10.6 |       7.6 |      0.456 |              111 |
| yolov9s                       | YOLOv9       |       640 |          7.3 | 0.455 |   0.617 |  0.257 |      12.7 |       9.7 |      0.455 |               82 |
| yolov8s                       | YOLOv8       |       640 |         11.2 | 0.442 |   0.608 |  0.254 |      11.5 |       7.8 |      0.441 |               79 |
| detr_resnet50                 | DETR         |       800 |         41.5 | 0.421 |   0.623 |  0.213 |      91.7 |      71   |      0.42  |              477 |
| retinanet_resnet50_fpn_v2     | RetinaNet    |       800 |         38.2 | 0.415 |   0.618 |  0.27  |     117.8 |      75.5 |      0.415 |              378 |
| yolo12n                       | YOLO12       |       640 |          2.6 | 0.398 |   0.556 |  0.194 |       8.6 |       8.7 |      0.398 |               66 |
| fcos_resnet50_fpn             | FCOS         |       800 |         32.3 | 0.392 |   0.583 |  0.222 |     128   |      99   |      0.391 |              423 |
| yolo11n                       | YOLO11       |       640 |          2.6 | 0.387 |   0.545 |  0.194 |       6.6 |       6.2 |      0.387 |               66 |
| yolov10n                      | YOLOv10      |       640 |          2.8 | 0.385 |   0.541 |  0.193 |       6.6 |       6.5 |      0.385 |               62 |
| yolov9t                       | YOLOv9       |       640 |          2.1 | 0.373 |   0.521 |  0.173 |       9.1 |       9.2 |      0.373 |               43 |
| fasterrcnn_resnet50_fpn       | Faster R-CNN |       800 |         41.8 | 0.372 |   0.59  |  0.213 |     113.5 |      96.2 |      0.371 |              664 |
| yolov8n                       | YOLOv8       |       640 |          3.2 | 0.367 |   0.521 |  0.178 |       6.8 |       5.7 |      0.366 |               33 |
| retinanet_resnet50_fpn        | RetinaNet    |       800 |         34   | 0.364 |   0.557 |  0.191 |     139.2 |     106.4 |      0.364 |              428 |
| ssd300_vgg16                  | SSD          |       300 |         35.6 | 0.251 |   0.415 |  0.055 |      46.8 |      37.2 |      0.251 |              212 |
| ssdlite320_mobilenet_v3_large | SSD          |       320 |          3.4 | 0.213 |   0.343 |  0.011 |      27.6 |      28   |      0.213 |               33 |

*mAP = mAP@[.5:.95]. ms = latência total mediana. VRAM = pico de memória alocada pelo PyTorch (não inclui contexto CUDA).*

### 14.1 Principais observações

- **Fronteira de Pareto (mAP × latência total).** FP32: YOLO11n → YOLO12n → YOLOv10s → YOLO12s → RF-DETR-N → YOLOv10m → YOLO11m → YOLO12m → RF-DETR-S → RF-DETR-M. FP16: YOLOv8n → YOLO11n → YOLOv10s → YOLO12s → YOLOv10m → YOLO11m → YOLO12m → RF-DETR-S → RF-DETR-M. **Nenhum detector clássico, nem o RT-DETR-L, está na fronteira.**
- **Maior acurácia:** RF-DETR-Medium (0,547; 38,2 ms em FP32, 23,3 ms em FP16). Melhor YOLO: YOLO12m (0,514; 24,4 ms). RT-DETR-L tem o mesmo mAP do RF-DETR-Small (0,528) com 46,2 ms contra 29,4 ms.
- **Faixa rápida (< ~12 ms):** dominada por YOLO. YOLO11n e YOLOv10n têm ≈ 0,386 de mAP em ≈ 6,6 ms (FP32).
- **YOLOv8 e YOLOv9** ficam abaixo de YOLO11/12/v10 em todos os portes. YOLOv9 é lento para o que entrega (v9m: 21,2 ms só de inferência contra 17,8 ms do YOLO11m com mAP equivalente).
- **YOLO12 vs. YOLO11:** +1 ponto de mAP, mas o YOLO12n gasta 5,8 ms de inferência contra 3,8 ms do YOLO11n; em FP16 o YOLO12n sai da fronteira.
- **Clássicos:** o melhor, Faster R-CNN v2 (0,469), leva 158 ms em FP32, contra 20,5 ms do YOLO11m (0,505). SSD/SSDlite são rápidos mas com mAP de 0,25/0,21 e AP_S ≈ 0,01–0,06 (quase não detectam objetos pequenos). DETR-R50 (0,421; 92 ms) supera RetinaNet v1/FCOS/Faster R-CNN v1 em acurácia.
- **Objetos pequenos (AP_S):** RF-DETR-M 0,358; YOLO12m 0,345; RT-DETR-L 0,344; YOLO11m 0,334; Faster R-CNN v2 0,310; SSD300 0,055; SSDlite 0,011.

### 14.2 Respostas (parciais) às perguntas de pesquisa

- **RQ1 (Pareto):** respondida para este hardware — ver fronteira acima.
- **RQ2 (end-to-end vs. NMS):** **parcial.** YOLOv10 (sem NMS) aparece na fronteira nos portes s e m, e RF-DETR (sem NMS) ocupa o topo; mas o RT-DETR-L é dominado. Não há separação do custo do NMS na latência (só existe para Ultralytics), então não dá para atribuir a diferença ao pós-processamento.
- **RQ3 (objetos pequenos / resolução):** **parcial.** AP_S por modelo disponível; a degradação com a resolução **não foi estudada**.
- **RQ4 (FP16, exportação):** **parcial.** Em Ultralytics, FP16 não altera o mAP (variação ≤ 0,001), reduz em ~30–40 % o tempo dos modelos médios e quase metade a VRAM; ganho pequeno nos nano. RF-DETR-Base: 42,3 → 23,6 ms. ONNX/TensorRT **não foram avaliados**.
- **RQ5 (custo da máscara):** **não respondida** (segmentação não executada).

### 14.3 Ressalvas e anomalias

- **FCOS em FP16 (anomalia investigada e corrigida).** A primeira execução deu 0,310 mAP em FP16 (autocast) contra 0,392 em FP32. Diagnóstico (`scripts/diag_fcos_fp16.py`, `scripts/diag_fcos_decode.py`; 500 imagens): a rede e a cabeça em FP16 estão corretas (converter só as saídas da cabeça para FP32 recupera 0,4291 = FP32); o erro de decodificação das caixas em FP16 é pequeno (≤ 0,74 px, IoU mediano 0,997) e **não** é a causa; a causa é o **NMS recebendo caixas em FP16**. O `batched_nms` do torchvision separa as classes somando `classe × (coord_máx + 1)` às caixas; em FP16 isso passa de 32.768 (passo numérico de 32 px) e vira `inf` para classes de índice alto (acima de ~70), corrompendo a supressão de duplicatas. Rodando só o NMS em FP32 o mAP é 0,4297 (500 imgs). **Correção aplicada** em `eval_classic.py` (NMS em FP32 no FCOS); resultado final no COCO val2017 completo: **0,391 (FP16) contra 0,392 (FP32)**, 99 ms contra 128 ms. O resultado original foi guardado em `results/diag/`. Os demais modelos torchvision/DETR têm mAP FP16 ≈ FP32 (diferença ≤ 0,002), logo não são afetados (o FCOS é o único que decodifica no dtype da regressão). **Lição para o artigo:** FP16 "ingênuo" em detectores com NMS por deslocamento de classe pode degradar silenciosamente a acurácia; vale reportar mAP em FP32 e FP16 lado a lado, como aqui.
- **RF-DETR-Base** é a variante antiga do pacote e é dominada pelo Medium (mais lento e menos preciso); incluída por completude.
- **Resoluções diferentes** entre famílias (§7.1): a comparação não é "mesma entrada". O RF-DETR obtém o melhor mAP mesmo com entrada menor (384–576 px).
- **Latência total** inclui leitura do arquivo em disco e, nos Ultralytics, o NMS com `conf = 0,001`; o RF-DETR não tem NMS e roda 300 consultas fixas.
- **Térmica:** a GPU atingiu ~75 °C durante as execuções longas; não foi registrado o clock. Possível *throttling* não descartado.
- **Medição única** por configuração; nenhuma barra de erro.
- **Conferência de plausibilidade:** o YOLO11n obteve mAP 0,387 contra 39,5 divulgado pela Ultralytics, o que indica que a conversão e a avaliação COCO estão corretas. Os valores dos demais modelos **não foram comparados sistematicamente** com os divulgados pelos autores.

## 15. Pendências

1. Segmentação de instância (adaptador escrito, não testado) e semântica (dataset a definir).
2. Repetir cada medição de latência (3–5 rodadas) e registrar temperatura/clock.
3. ~~Investigar a queda do FCOS em FP16~~ — concluído (ver §14.3).
4. Estudos de resolução, ONNX/TensorRT e latência com `conf = 0,25`.
5. RT-DETR-R18/R50 (HuggingFace) e portes l/x onde couberem.
6. Conferir as métricas contra os valores publicados e verificar as referências (§13).
7. Ajustar manualmente os rótulos do gráfico de Pareto para a versão do artigo.
