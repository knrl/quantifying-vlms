# Edge Reliability Gap in Vision-Language Models

**Quantifying Failure Modes of Compressed VLMs Under Visual Corruption**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)


## Repository Structure

```
workspace_2/
│
├── scripts/                           # Experiment pipeline (phases 0–6)
│   ├── edge_reliability_gap.py        # Phase 0: sanity check (25+25 rows)
│   ├── batch_inference.py             # Phase 1: full inference (2×2,000 rows)
│   ├── robustness_blur.py             # Phase 2: Gaussian blur robustness
│   ├── calibration_ece.py             # Phase 3: ECE + reliability diagrams
│   ├── negation_probes.py             # Phase 4: negation stress tests
│   ├── llm_judge.py                   # Phase 5: GPT-4o taxonomy + Cohen's κ
│   └── update_report.py               # Phase 6: patch REPORT.md placeholders
│
├── figures/
│   ├── scripts/                       # Figure generation (all 300 DPI)
│   │   ├── taxonomy_chart.py          #   → taxonomy_distribution.png   (Fig 1)
│   │   ├── qualitative_grid.py        #   → failure_grid.png            (Fig 2)
│   │   ├── reliability_diagram.py     #   → reliability_diagram.png     (Fig 3)
│   │   ├── robustness_curve.py        #   → robustness_curve.png        (Fig 4)
│   │   └── negation_chart.py          #   → negation_chart.png          (Fig 5)
│   ├── taxonomy_distribution.png
│   ├── failure_grid.png
│   ├── reliability_diagram.png
│   ├── robustness_curve.png
│   ├── negation_chart.png
│   ├── calibration_reliability_diagram_*.png  (4 per-model diagrams)
│   ├── img_vqav2_*.png               # Cached images for qualitative grid
│   └── img_coco_*.png
│
├── results/                           # All experiment outputs
│   ├── vlm_inference_results.csv      # 4,000 rows of model outputs
│   ├── vlm_inference_checkpoint.json  # Checkpoint for resumable inference
│   ├── robustness_report.json         # ρ, per-model blur drop
│   ├── robustness_blurred_results.csv # Per-sample blur results
│   ├── calibration_results.json       # ECE + bin stats (M=10)
│   ├── negation_probes_results.csv    # Per-probe outcomes
│   ├── negation_probes_summary.json   # Aggregated success rates
│   ├── llm_judge_labels.json          # 400 GPT-4o taxonomy labels
│   └── cohen_kappa_report.json        # Inter-model κ values
│
├── docs/
│   ├── REPORT.md                      # Extended paper draft (Markdown)
│   ├── PROPOSAL.tex                   # Original research proposal (Turkish)
│   └── additional_literature_review.md
│
├── run_pipeline.sh                    # Automated pipeline runner (all phases)
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/knrl/<...>
cd workspace_2
pip install -r requirements.txt
```

> **GPU requirement:** NVIDIA GPU with ≥ 6 GiB VRAM and CUDA support.
> For PyTorch with CUDA, follow [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 2. Run the full pipeline

```bash
bash run_pipeline.sh
```

Or skip phases whose outputs already exist:

```bash
# Skip batch inference (if results/vlm_inference_results.csv already exists)
bash run_pipeline.sh --skip-batch

# With GPT-4o taxonomy judge (requires API key, ~$2–5 cost)
OPENAI_API_KEY=sk-... bash run_pipeline.sh --skip-batch

# Commit locally but do not push
bash run_pipeline.sh --no-push

# Dry-run without any git operations
bash run_pipeline.sh --no-git
```

### 3. Run phases individually

```bash
python scripts/edge_reliability_gap.py              # Phase 0: sanity check (~5 min)
python scripts/batch_inference.py                   # Phase 1: main inference, resumable (4–8 h)
python scripts/robustness_blur.py                   # Phase 2: Gaussian blur test (~15 min)
python scripts/calibration_ece.py                   # Phase 3: ECE + reliability diagrams (~5 min)
python scripts/negation_probes.py                   # Phase 4: negation stress tests (~30 min)
OPENAI_API_KEY=sk-... python scripts/llm_judge.py   # Phase 5: GPT-4o taxonomy (~10 min)
python scripts/update_report.py                     # Phase 6: patch REPORT.md (<1 s)
```

### 4. Regenerate figures

```bash
python figures/scripts/taxonomy_chart.py         # Fig 1: error taxonomy distribution
python figures/scripts/qualitative_grid.py       # Fig 2: qualitative failure grid
python figures/scripts/reliability_diagram.py    # Fig 3: calibration reliability diagrams
python figures/scripts/robustness_curve.py       # Fig 4: blur degradation curve
python figures/scripts/negation_chart.py         # Fig 5: negation probe success rates
```

---

## Pipeline Architecture

Phases 2–4 are **independent** and all depend only on Phase 1's output.
Phase 5 reads the inference CSV directly. Phase 6 reads all result JSONs.

```
edge_reliability_gap.py     (Phase 0 — sanity check, optional)
          │
          ▼
batch_inference.py          (Phase 1 → vlm_inference_results.csv)
          │
    ┌─────┼──────────────────────┐
    ▼     ▼                      ▼
Phase 2   Phase 3             Phase 4
blur      calibration         negation
    │     │                      │
    ▼     ▼                      ▼
  .json   .json + .png         .json
                   │
                   ▼
           llm_judge.py          (Phase 5 → taxonomy labels + κ)
                   │
                   ▼
           update_report.py      (Phase 6 → patches REPORT.md)
```

### Pipeline runner options

```
bash run_pipeline.sh [OPTIONS]

  --skip-batch       Skip Phase 1 (batch inference)
  --skip-blur        Skip Phase 2 (blur robustness)
  --skip-ece         Skip Phase 3 (calibration)
  --skip-negation    Skip Phase 4 (negation probes)
  --skip-llm         Skip Phase 5 (LLM judge)
  --skip-report      Skip Phase 6 (report update)
  --no-push          Commit locally but do not push
  --no-git           Disable all git operations
  --no-color         Disable ANSI color output
  -h, --help         Show full help text
```

---

## Models

| Property | Qwen2.5-VL-7B-Instruct | SmolVLM2-500M-Instruct |
|---|---|---|
| Parameters | ~7.6 B | ~0.5 B |
| Precision | 4-bit NF4 (bitsandbytes) | FP16 |
| VRAM | ~4.5 GiB | ~1.0 GiB |
| Vision encoder | Qwen2.5-VL native (dynamic res.) | SigLIP-400M |
| LLM backbone | Qwen2.5-7B | SmolLM2-135M |
| HF model ID | `Qwen/Qwen2.5-VL-7B-Instruct` | `HuggingFaceTB/SmolVLM2-500M-Instruct` |

Both models load simultaneously on a single GPU (combined VRAM ≈ 5.5–6 GiB).

## Datasets

| Dataset | HuggingFace path | Split | Samples | Task |
|---|---|---|---|---|
| VQAv2 | `lmms-lab/VQAv2` | `validation` | 2,000 | Open-ended VQA |
| COCO Captions | `lmms-lab/COCO-Caption` | `val` | 2,000 | Image captioning |

Datasets are **streamed** via HuggingFace `datasets` — never fully loaded into RAM.

---

## Figures

| # | File | Description |
|---|------|-------------|
| 1 | [`figures/taxonomy_distribution.png`](figures/taxonomy_distribution.png) | GPT-4o error-taxonomy labels (Object Blindness dominates) |
| 2 | [`figures/failure_grid.png`](figures/failure_grid.png) | Qualitative failure grid (3 examples × 4 columns) |
| 3 | [`figures/reliability_diagram.png`](figures/reliability_diagram.png) | Calibration reliability diagrams (VQAv2 + COCO) |
| 4 | [`figures/robustness_curve.png`](figures/robustness_curve.png) | Blur degradation curve (σ = 0–5) |
| 5 | [`figures/negation_chart.png`](figures/negation_chart.png) | Per-template negation probe success rates |

---

## Reproducibility

- **Deterministic:** All scripts share `SEED = 42`.
- **Resumable:** Phase 1 (`batch_inference.py`) checkpoints after each batch —
  re-run to continue from where it left off.
- **Offline-capable:** After initial model download, all phases except Phase 5
  (GPT-4o) run without network access.
- **Phase 5 cost:** LLM-as-Judge uses ~400 GPT-4o API calls (~$2–5).
  Set `OPENAI_API_KEY` to enable; omit to skip automatically.

### Tested environment

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| CUDA | 12.x |
| GPU | NVIDIA RTX 5090 (32 GiB) |
| OS | Ubuntu 22.04+ |

---

## Citation

```bibtex
@misc{edge_reliability_gap_2026,
  title   = {Edge Reliability Gap in Vision-Language Models:
             Quantifying Failure Modes of Compressed VLMs
             Under Visual Corruption},
  author  = {Anonymous},
  year    = {2026},
  url     = {https://github.com/knrl/workspace_2}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
