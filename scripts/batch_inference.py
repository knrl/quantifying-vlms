"""
Edge Reliability Gap in Vision–Language Models
===============================================
Batched inference over VQAv2 (validation) and COCO Captions.
Results are streamed to vlm_inference_results.csv with checkpoint
resumption so the run can be safely interrupted and restarted.

Models
------
1. Qwen/Qwen2.5-VL-7B-Instruct        → 4-bit NF4 (bitsandbytes)
2. HuggingFaceTB/SmolVLM2-500M-Instruct → FP16

Requirements
------------
    pip install torch torchvision transformers accelerate \
                bitsandbytes qwen-vl-utils pillow requests \
                datasets tqdm
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import gc
import io
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Generator

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
from PIL import Image, UnidentifiedImageError
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    SmolVLMForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_UTILS = True
except ImportError:
    _QWEN_UTILS = False

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("Install the `datasets` package: pip install datasets") from exc


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 2_000          # per dataset
MAX_NEW_TOKENS = 128
LOG_EVERY_N_BATCHES = 50
RESULTS_CSV = Path("results/vlm_inference_results.csv")
CHECKPOINT_FILE = Path("results/vlm_inference_checkpoint.json")
SKIP_LOG_FILE = Path("vlm_skipped_indices.log")

# Batch size auto-tuning limits
BATCH_MIN = 1
BATCH_MAX = 16

DATASET_CONFIGS = [
    {
        "name": "vqav2",
        "hf_path": "lmms-lab/VQAv2",
        "split": "validation",
        "prompt_field": "question",        # dataset column used as prompt
        "answer_field": "answers",         # ground-truth column
        "image_field": "image",
    },
    {
        "name": "coco_captions",
        "hf_path": "lmms-lab/COCO-Caption",
        "split": "val",
        "prompt_field": None,              # use fixed caption prompt
        "answer_field": "answer",          # list of reference captions
        "image_field": "image",
    },
]

COCO_PROMPT = "Generate a short caption for this image."

CSV_FIELDNAMES = [
    "dataset_name",
    "image_id",
    "task_prompt",
    "ground_truth",
    "smol_output",
    "qwen_output",
]


# ══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_inference.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════════════════════

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# VRAM helpers
# ══════════════════════════════════════════════════════════════════════════════

def vram_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(DEVICE) / 1024 ** 2


def vram_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(DEVICE) / 1024 ** 2


def free_vram_mb() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    props = torch.cuda.get_device_properties(DEVICE)
    return (props.total_memory - torch.cuda.memory_reserved(DEVICE)) / 1024 ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Batch-size auto-tuner
# ══════════════════════════════════════════════════════════════════════════════

def compute_initial_batch_size() -> int:
    """
    Heuristically pick a starting batch size based on free VRAM.
    Both models are already loaded so we work with remaining headroom.
    """
    free = free_vram_mb()
    # Very rough estimate: ~500 MiB per sample for combined forward passes
    estimate = max(BATCH_MIN, min(BATCH_MAX, int(free // 500)))
    log.info("Auto-selected batch size: %d  (free VRAM ≈ %.0f MiB)", estimate, free)
    return estimate


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict[str, int]:
    """Return {dataset_name: last_processed_index} or empty dict."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            log.info("Resuming from checkpoint: %s", data)
            return data
        except Exception:
            log.warning("Checkpoint file corrupt – starting from scratch.")
    return {}


def save_checkpoint(state: dict[str, int]):
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def log_skipped(dataset_name: str, index: int, reason: str):
    with SKIP_LOG_FILE.open("a") as fh:
        fh.write(f"{dataset_name}\t{index}\t{reason}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def open_csv_writer(path: Path):
    """Open CSV in append mode; write header only if file is new/empty."""
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


# ══════════════════════════════════════════════════════════════════════════════
# Dataset streaming helpers
# ══════════════════════════════════════════════════════════════════════════════

def validate_image(raw) -> Image.Image | None:
    """
    Accept a PIL Image, bytes, or file-like object.
    Returns an RGB PIL Image or None on failure.
    """
    try:
        if isinstance(raw, Image.Image):
            img = raw.convert("RGB")
        elif isinstance(raw, (bytes, bytearray)):
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif hasattr(raw, "read"):
            img = Image.open(raw).convert("RGB")
        elif isinstance(raw, dict) and "bytes" in raw:
            img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        else:
            return None
        # Minimal sanity: non-zero size
        if img.size[0] == 0 or img.size[1] == 0:
            return None
        return img
    except (UnidentifiedImageError, OSError, Exception):
        return None


class StreamingVLMDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming dataset, yielding dicts:
        {index, image_id, image, prompt, ground_truth}

    Skips corrupted samples and honours a resume offset.
    """

    def __init__(self, cfg: dict, resume_from: int = 0):
        self.cfg = cfg
        self.resume_from = resume_from
        self.name = cfg["name"]

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        cfg = self.cfg
        ds = load_dataset(
            cfg["hf_path"],
            split=cfg["split"],
            streaming=True,
            trust_remote_code=True,
        )

        yielded = 0
        global_idx = 0

        for raw in ds:
            if yielded >= MAX_SAMPLES:
                break

            global_idx_current = global_idx
            global_idx += 1

            # Resume: skip already-processed samples
            if global_idx_current < self.resume_from:
                continue

            # ── Extract image ────────────────────────────────────────────────
            image_field = cfg.get("image_field")
            image: Image.Image | None = None
            if image_field and image_field in raw:
                image = validate_image(raw[image_field])
                if image is None:
                    reason = "invalid_image"
                    log_skipped(self.name, global_idx_current, reason)
                    log.debug("Skipped %s idx=%d (%s)", self.name, global_idx_current, reason)
                    continue

            # ── Extract prompt ───────────────────────────────────────────────
            prompt_field = cfg.get("prompt_field")
            if prompt_field and prompt_field in raw:
                prompt = str(raw[prompt_field])
            else:
                prompt = COCO_PROMPT

            # ── Extract ground truth ─────────────────────────────────────────
            answer_field = cfg.get("answer_field")
            ground_truth = ""
            if answer_field and answer_field in raw:
                gt = raw[answer_field]
                if isinstance(gt, list):
                    # VQAv2 answers are list-of-dicts: [{"answer": "..."}, ...]
                    if gt and isinstance(gt[0], dict):
                        ground_truth = gt[0].get("answer", str(gt[0]))
                    else:
                        ground_truth = str(gt[0]) if gt else ""
                elif isinstance(gt, dict):
                    ground_truth = str(next(iter(gt.values()), ""))
                else:
                    ground_truth = str(gt)

            # ── Image ID ─────────────────────────────────────────────────────
            image_id = raw.get("question_id", raw.get("id", raw.get("image_id", global_idx_current)))

            yield {
                "index": global_idx_current,
                "image_id": str(image_id),
                "image": image,          # PIL.Image or None
                "prompt": prompt,
                "ground_truth": ground_truth,
            }
            yielded += 1


def collate_fn(batch: list[dict]) -> dict:
    """Keep heterogeneous items as a plain list-of-dicts (no tensor stacking)."""
    return {k: [item[k] for item in batch] for k in batch[0]}


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders  (identical config to sanity-check script)
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    log.info("Loading %s (4-bit NF4) …", model_id)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": DEVICE},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    log.info("Qwen loaded  | VRAM allocated: %.0f MiB", vram_allocated_mb())
    return model, processor


def load_smolvlm(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"):
    log.info("Loading %s (FP16) …", model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    log.info("SmolVLM2 loaded  | VRAM allocated: %.0f MiB", vram_allocated_mb())
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample inference
# ══════════════════════════════════════════════════════════════════════════════

def _qwen_single(model, processor, image: Image.Image | None, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    [{"type": "image", "image": image}]
                    if image is not None else []
                ),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if image is not None and _QWEN_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _smol_single(model, processor, image: Image.Image | None, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    [{"type": "image"}]
                    if image is not None else []
                ),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text_prompt,
        images=[image] if image is not None else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)
    raw = decoded[0] if decoded else ""
    if prompt in raw:
        raw = raw.split(prompt, 1)[-1].strip()
    return raw.strip()


def run_batch(
    batch: dict,
    qwen_model, qwen_proc,
    smol_model, smol_proc,
    dataset_name: str,
) -> list[dict]:
    """
    Process one batch. Each sample is inferred independently so a single
    failure never aborts the whole batch.
    Returns a list of result-row dicts ready for CSV.
    """
    rows = []
    indices = batch["index"]
    image_ids = batch["image_id"]
    images = batch["image"]
    prompts = batch["prompt"]
    ground_truths = batch["ground_truth"]

    for i in range(len(indices)):
        idx = indices[i]
        img = images[i]
        prompt = prompts[i]
        gt = ground_truths[i]
        image_id = image_ids[i]

        smol_out = ""
        qwen_out = ""

        # ── SmolVLM2 ──────────────────────────────────────────────────────────
        try:
            smol_out = _smol_single(smol_model, smol_proc, img, prompt)
        except torch.cuda.OutOfMemoryError:
            log.warning("[OOM-SmolVLM2] idx=%d – skipping sample", idx)
            torch.cuda.empty_cache()
            smol_out = "[OOM]"
        except Exception as exc:
            log.warning("[ERR-SmolVLM2] idx=%d – %s", idx, exc)
            smol_out = f"[ERROR: {type(exc).__name__}]"

        # ── Qwen ─────────────────────────────────────────────────────────────
        try:
            qwen_out = _qwen_single(qwen_model, qwen_proc, img, prompt)
        except torch.cuda.OutOfMemoryError:
            log.warning("[OOM-Qwen] idx=%d – skipping sample", idx)
            torch.cuda.empty_cache()
            qwen_out = "[OOM]"
        except Exception as exc:
            log.warning("[ERR-Qwen] idx=%d – %s", idx, exc)
            qwen_out = f"[ERROR: {type(exc).__name__}]"

        rows.append(
            {
                "dataset_name": dataset_name,
                "image_id": image_id,
                "task_prompt": prompt,
                "ground_truth": gt,
                "smol_output": smol_out,
                "qwen_output": qwen_out,
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset loop
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(
    cfg: dict,
    qwen_model, qwen_proc,
    smol_model, smol_proc,
    checkpoint: dict[str, int],
    csv_writer,
    csv_fh,
    batch_size: int,
):
    name = cfg["name"]
    resume_from = checkpoint.get(name, 0)
    log.info("=== Dataset: %s  |  resume_from=%d  |  batch_size=%d ===",
             name, resume_from, batch_size)

    dataset = StreamingVLMDataset(cfg, resume_from=resume_from)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,          # streaming datasets don't support multi-worker
        collate_fn=collate_fn,
        pin_memory=False,
    )

    batch_count = 0
    total_samples = 0
    total_inf_time = 0.0
    run_start = time.perf_counter()

    current_batch_size = batch_size

    for batch in loader:
        batch_start = time.perf_counter()

        try:
            rows = run_batch(
                batch,
                qwen_model, qwen_proc,
                smol_model, smol_proc,
                dataset_name=name,
            )
        except torch.cuda.OutOfMemoryError:
            # OOM at batch level – halve batch size for the rest of this dataset
            torch.cuda.empty_cache()
            gc.collect()
            new_bs = max(BATCH_MIN, current_batch_size // 2)
            log.warning(
                "[OOM] Batch-level OOM on %s. Reducing batch size %d → %d.",
                name, current_batch_size, new_bs,
            )
            current_batch_size = new_bs
            # Re-create loader with smaller batch size; resume is handled by
            # checkpoint (we haven't advanced the checkpoint yet for this batch)
            loader = DataLoader(
                StreamingVLMDataset(cfg, resume_from=resume_from + total_samples),
                batch_size=current_batch_size,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=False,
            )
            continue
        except Exception as exc:
            log.error("Unexpected error on batch %d of %s: %s", batch_count, name, exc)
            log.debug(traceback.format_exc())
            continue

        batch_time = time.perf_counter() - batch_start
        n = len(rows)
        total_samples += n
        total_inf_time += batch_time
        batch_count += 1

        # ── Write rows immediately ────────────────────────────────────────────
        for row in rows:
            csv_writer.writerow(row)
        csv_fh.flush()

        # ── Update checkpoint ─────────────────────────────────────────────────
        last_idx = batch["index"][-1] + 1   # exclusive upper bound
        checkpoint[name] = resume_from + last_idx
        save_checkpoint(checkpoint)

        # ── Periodic progress log ─────────────────────────────────────────────
        if batch_count % LOG_EVERY_N_BATCHES == 0:
            elapsed = time.perf_counter() - run_start
            avg_inf = total_inf_time / batch_count
            sps = total_samples / elapsed if elapsed > 0 else 0.0
            log.info(
                "[%s] batch=%d  samples=%d  avg_batch_time=%.2fs  "
                "samples/sec=%.2f  VRAM_alloc=%.0f MiB  VRAM_rsvd=%.0f MiB",
                name, batch_count, total_samples,
                avg_inf, sps,
                vram_allocated_mb(), vram_reserved_mb(),
            )

    elapsed_total = time.perf_counter() - run_start
    log.info(
        "[%s] DONE – %d samples in %.1f s (%.2f samples/sec)",
        name, total_samples, elapsed_total,
        total_samples / elapsed_total if elapsed_total > 0 else 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  Edge Reliability Gap – Batch Inference")
    log.info("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(DEVICE)
        log.info("GPU : %s  (%.1f GiB total)", props.name, props.total_memory / 1024 ** 3)
    log.info("Results CSV  : %s", RESULTS_CSV.resolve())
    log.info("Checkpoint   : %s", CHECKPOINT_FILE.resolve())
    log.info("Skip log     : %s", SKIP_LOG_FILE.resolve())

    # ── Load models ──────────────────────────────────────────────────────────
    try:
        qwen_model, qwen_proc = load_qwen()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load Qwen2.5-VL-7B – aborting.")
        return

    try:
        smol_model, smol_proc = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load SmolVLM2-500M – aborting.")
        return

    log.info("Both models loaded | VRAM allocated: %.0f MiB", vram_allocated_mb())

    batch_size = compute_initial_batch_size()
    checkpoint = load_checkpoint()

    # ── Open CSV (append) ────────────────────────────────────────────────────
    csv_fh, csv_writer = open_csv_writer(RESULTS_CSV)

    try:
        for cfg in DATASET_CONFIGS:
            process_dataset(
                cfg,
                qwen_model, qwen_proc,
                smol_model, smol_proc,
                checkpoint,
                csv_writer,
                csv_fh,
                batch_size,
            )
    finally:
        csv_fh.close()
        del qwen_model, smol_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Models unloaded, CUDA cache cleared.")
        log.info("Results written to: %s", RESULTS_CSV.resolve())


if __name__ == "__main__":
    main()
