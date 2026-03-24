"""
Edge Reliability Gap in Vision–Language Models
===============================================
Loads two VLMs simultaneously on a single RTX 5090 (32 GB VRAM) and
runs a single inference sanity-check on each.

Models
------
1. Qwen/Qwen2.5-VL-7B-Instruct   → 4-bit (bitsandbytes NF4)
2. HuggingFaceTB/SmolVLM2-500M-Instruct → FP16

Requirements
------------
    pip install torch torchvision transformers accelerate \
                bitsandbytes qwen-vl-utils pillow requests
"""

# ── Standard library ──────────────────────────────────────────────────────────
import gc
import io
import os
import random
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    SmolVLMForConditionalGeneration,
)

# ── Optional Qwen helper (falls back gracefully if not installed) ──────────────
try:
    from qwen_vl_utils import process_vision_info
    _QWEN_UTILS = True
except ImportError:
    _QWEN_UTILS = False
    print("[WARN] qwen-vl-utils not found – falling back to manual image handling.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Deterministic seeds
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROMPT_TEXT = "Describe this image in detail."
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
MAX_NEW_TOKENS = 256


# ══════════════════════════════════════════════════════════════════════════════
# 2. Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def vram_mb() -> float:
    """Return current GPU memory allocated in MiB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(DEVICE) / 1024 ** 2


def vram_peak_mb() -> float:
    """Return peak GPU memory allocated in MiB since last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)


def print_vram(label: str):
    print(f"  [{label}] VRAM allocated: {vram_mb():.1f} MiB")


def download_image(url: str, retries: int = 1) -> Image.Image | None:
    """Download an image from *url*, retrying once on failure."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            print(f"  Image downloaded successfully ({img.size[0]}×{img.size[1]} px).")
            return img
        except Exception as exc:
            if attempt < retries:
                print(f"  [WARN] Image download failed ({exc}). Retrying …")
                time.sleep(2)
            else:
                print(f"  [ERROR] Image download failed after {retries + 1} attempt(s): {exc}")
                return None


def print_result_block(
    model_name: str,
    load_time: float,
    inference_time: float,
    vram_used_mb: float,
    output_text: str,
):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"=== {model_name} ===")
    print(sep)
    print(f"Load Time      : {load_time:.2f} s")
    print(f"Inference Time : {inference_time:.2f} s")
    print(f"VRAM Used      : {vram_used_mb:.1f} MiB")
    print(f"Output         :\n  {output_text.strip()}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load Qwen2.5-VL-7B with 4-bit NF4 quantisation via bitsandbytes."""
    print(f"\n[Qwen] Loading {model_id} (4-bit NF4) …")
    print_vram("before load")
    reset_peak_vram()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": DEVICE},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0

    model.eval()
    print_vram("after load")
    return model, processor, load_time


def load_smolvlm(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"):
    """Load SmolVLM2-500M in FP16."""
    print(f"\n[SmolVLM2] Loading {model_id} (FP16) …")
    print_vram("before load")
    reset_peak_vram()

    t0 = time.perf_counter()
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0

    model.eval()
    print_vram("after load")
    return model, processor, load_time


# ══════════════════════════════════════════════════════════════════════════════
# 4. Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_qwen_inference(model, processor, image: Image.Image | None) -> tuple[float, str]:
    """Run one forward pass through the Qwen model."""
    reset_peak_vram()

    # Build the chat-style message expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} if image is not None
                else {"type": "text", "text": "[No image available]"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Build pixel tensors
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
        # Fallback: pass PIL image directly
        inputs = processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    inference_time = time.perf_counter() - t0

    # Trim input tokens from output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
    return inference_time, decoded[0] if decoded else ""


def run_smolvlm_inference(model, processor, image: Image.Image | None) -> tuple[float, str]:
    """Run one forward pass through the SmolVLM2 model."""
    reset_peak_vram()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"} if image is not None
                else {"type": "text", "text": "[No image available]"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    if image is not None:
        inputs = processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=text_prompt,
            return_tensors="pt",
        )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    inference_time = time.perf_counter() - t0

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
    # Strip the echoed prompt that some processors include
    raw = decoded[0] if decoded else ""
    if PROMPT_TEXT in raw:
        raw = raw.split(PROMPT_TEXT, 1)[-1].strip()
    return inference_time, raw


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Edge Reliability Gap in Vision–Language Models")
    print("  Single-GPU Sanity Check")
    print("=" * 60)
    print(f"\nDevice  : {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(DEVICE)
        total_vram = props.total_memory / 1024 ** 3
        print(f"GPU     : {props.name}  ({total_vram:.1f} GiB total VRAM)")
    print(f"Seed    : {SEED}\n")

    # ── Download test image ───────────────────────────────────────────────────
    print("── Downloading test image ──")
    image = download_image(TEST_IMAGE_URL)

    # ── Load both models ─────────────────────────────────────────────────────
    vram_before_any = vram_mb()
    print(f"\nVRAM before any model load: {vram_before_any:.1f} MiB")

    try:
        qwen_model, qwen_processor, qwen_load_time = load_qwen()
    except torch.cuda.OutOfMemoryError:
        print("[OOM] CUDA out of memory while loading Qwen2.5-VL-7B. Aborting.")
        return

    try:
        smol_model, smol_processor, smol_load_time = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        print("[OOM] CUDA out of memory while loading SmolVLM2-500M. Aborting.")
        return

    vram_after_both = vram_mb()
    print(f"\nVRAM after both models loaded: {vram_after_both:.1f} MiB")

    # ── Qwen inference ────────────────────────────────────────────────────────
    print("\n── Running Qwen2.5-VL-7B inference ──")
    try:
        qwen_inf_time, qwen_output = run_qwen_inference(qwen_model, qwen_processor, image)
        qwen_peak = vram_peak_mb()
    except torch.cuda.OutOfMemoryError:
        print("[OOM] CUDA out of memory during Qwen inference.")
        qwen_inf_time, qwen_output, qwen_peak = 0.0, "[OOM]", 0.0

    # ── SmolVLM2 inference ────────────────────────────────────────────────────
    print("── Running SmolVLM2-500M inference ──")
    try:
        smol_inf_time, smol_output = run_smolvlm_inference(smol_model, smol_processor, image)
        smol_peak = vram_peak_mb()
    except torch.cuda.OutOfMemoryError:
        print("[OOM] CUDA out of memory during SmolVLM2 inference.")
        smol_inf_time, smol_output, smol_peak = 0.0, "[OOM]", 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    print_result_block(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct (4-bit NF4)",
        load_time=qwen_load_time,
        inference_time=qwen_inf_time,
        vram_used_mb=qwen_peak,
        output_text=qwen_output,
    )

    print_result_block(
        model_name="HuggingFaceTB/SmolVLM2-500M-Instruct (FP16)",
        load_time=smol_load_time,
        inference_time=smol_inf_time,
        vram_used_mb=smol_peak,
        output_text=smol_output,
    )

    # ── Teardown ──────────────────────────────────────────────────────────────
    del qwen_model, smol_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n[Done] Models unloaded, CUDA cache cleared.")


if __name__ == "__main__":
    main()
