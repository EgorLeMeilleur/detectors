"""
Qwen2.5-VL — batch frame-folder captioning for deepfake detection.

New dataset structure supported:

<dataset>/
    <method_1>/
        real/ or fake/ or ff/ or cdf/
            <video_id_1>/
                frames/
                    000001.jpg
                    000002.jpg
                    ...
            <video_id_2>/
                frames/
                    ...
    <method_2>/
        ...

Each leaf sample is now a folder with extracted frames instead of a video file.

Output:
    <dataset>/captions/results.json   — successful descriptions
    <dataset>/captions/skipped.json   — failed/skipped samples with reasons

Usage:
    python extract_captions.py --dataset /home/dataset --quantize --num-frames 16

    # Resume interrupted run
    python extract_captions.py --dataset /home/dataset --quantize --resume
"""

import argparse
import json
import os
import sys
import gc
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ── Deepfake-focused prompt ────────────────────────────────────────────────────

DEEPFAKE_PROMPT = """Analyze these video frames for signs of deepfake manipulation.
Examine the person's face carefully across all provided frames.
Respond ONLY with a valid JSON object — no markdown, no extra text.

{
  "general_description": "<3-5 sentences: who is in the video, what they are doing, overall visual quality and any notable observations>",
  "face_boundaries": "<edges around face, hair, neck: sharp/blurry/unnatural artifacts?>",
  "skin_texture": "<skin consistency across frames: smooth/waxy/flickering/inconsistent?>",
  "eye_behavior": "<eye movement and blinking: natural/rare/asymmetric/frozen?>",
  "lip_sync": "<mouth movement vs speech: aligned/delayed/stiff/unnatural?>",
  "lighting_consistency": "<lighting on face vs background: consistent/mismatched/shadow artifacts?>",
  "temporal_stability": "<face appearance across frames: stable/flickering/warping/identity shifts?>",
  "overall_suspicion": "<none|low|medium|high>"
}"""

FALLBACK_PROMPT = """Look at these frames. Does the person's face look real or artificially generated/swapped?
Respond ONLY with this JSON, no other text:
{"face_boundaries":"?","skin_texture":"?","eye_behavior":"?","lip_sync":"?","lighting_consistency":"?","temporal_stability":"?","overall_suspicion":"none|low|medium|high"}"""


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Extract deepfake-focused captions for all frame folders in a dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory.")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID. Default: Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--quantize", action="store_true", help="Load model in 4-bit (for 16GB VRAM GPUs).")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu. Auto-detected if not set.")
    parser.add_argument(
        "--num-frames", type=int, default=8,
        help="Number of frames to sample per sample folder. Default: 16",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Kept for compatibility. For frame folders this is ignored; sampling uses --num-frames.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens for model output. Default: 512")
    parser.add_argument(
        "--max-pixels", type=int, default=602112,
        help=(
            "Max pixels per frame fed to the model (width x height). "
            "Controls visual token count — critical for VRAM. "
            "602112 = 896x672 (~16 tokens/frame budget). "
            "Lower = less VRAM, faster. Default: 602112"
        ),
    )
    parser.add_argument(
        "--output-dir-name", default="captions",
        help="Name of output directory inside dataset root. Default: captions",
    )
    parser.add_argument("--resume", action="store_true", help="Resume: skip samples already in results.json.")
    parser.add_argument(
        "--trim-edges", type=float, default=0.05,
        help="Fraction of frames to skip at start and end of each frame folder (e.g. 0.05 = skip first/last 5%%). Default: 0.05",
    )
    parser.add_argument(
        "--retry-attempts", type=int, default=2,
        help="Retries per sample on JSON parse failure. Default: 2",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Max seconds for model inference per sample. Default: 120",
    )
    parser.add_argument(
        "--captions-dir-name", default="captions",
        help="Output directory name inside dataset root. Default: captions",
    )
    return parser.parse_args()


# ── Utilities ──────────────────────────────────────────────────────────────────

def get_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_label_from_path(sample_path: Path, dataset_root: Path):
    parts = sample_path.relative_to(dataset_root).parts
    for p in parts:
        pl = p.lower()
        if pl in ["real", "fake", "ff", "cdf"]:
            return pl
    return "unknown"

def collect_frame_folders(dataset_root: Path, output_dir: Path):
    """
    Find all sample folders:
    dataset/.../frames/<video_id>/
    where <video_id> contains images.
    """
    samples = []

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirpath = Path(dirpath)

        # skip output dir
        if output_dir in dirpath.parents or dirpath == output_dir:
            continue

        # ищем папки frames
        if dirpath.name.lower() != "frames":
            continue

        # теперь берем ВСЕ подпапки внутри frames
        for sub in dirpath.iterdir():
            if not sub.is_dir():
                continue

            has_images = any(
                p.suffix.lower() in IMAGE_EXTENSIONS for p in sub.iterdir()
            )

            if has_images:
                samples.append(sub)

    return sorted(samples)


def load_json_safe(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_images_in_folder(frames_dir: Path):
    images = []
    for p in sorted(frames_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images

def sample_images_from_folder(
    frames_dir: Path,
    num_frames: int,
    trim_edges: float = 0.0,
    max_pixels: int = 602112,
):
    """
    Sample images from a 'frames' folder.
    - trim_edges: fraction [0.0, 0.45] to skip at each end.
    - If there are fewer images than num_frames, returns all usable images.
    - max_pixels: resize each image so width*height <= max_pixels before returning.
    Returns list of PIL.Image.
    """
    images = list_images_in_folder(frames_dir)
    if not images:
        raise RuntimeError(f"No image files found in: {frames_dir}")

    trim_edges = max(0.0, min(trim_edges, 0.45))
    n = len(images)

    first = round(n * trim_edges)
    last = n - 1 - round(n * trim_edges)

    if last < first:
        raise RuntimeError(f"After trim_edges={trim_edges}, no images remain in: {frames_dir}")

    usable = images[first:last + 1]

    if len(usable) <= num_frames:
        selected = usable
    else:
        selected = [usable[round(i * (len(usable) - 1) / (num_frames - 1))] for i in range(num_frames)]

    frames = []
    for img_path in selected:
        with Image.open(img_path) as img:
            pil = img.convert("RGB")
            w, h = pil.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            frames.append(pil)

    if not frames:
        raise RuntimeError(f"Could not read any images from: {frames_dir}")

    return frames


def parse_model_json(text: str) -> dict:
    """
    Extract and parse the first JSON object found in model output.
    Handles extra text, markdown code blocks, etc.
    """
    text = text.strip()

    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model output")

    return json.loads(text[start:end])

def free_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_prompt(label: str):
    if label == "fake":
        label_text = "FAKE (manipulated / deepfake video)"
        instruction = "Pay extra attention to subtle artifacts and inconsistencies."
    elif label == "real":
        label_text = "REAL (authentic video)"
        instruction = "Verify that facial behavior and appearance are natural and consistent."
    elif label in ["ff", "cdf"]:
        label_text = f"{label.upper()} (dataset-specific class)"
        instruction = "Inspect the frames carefully for manipulation artifacts or visual inconsistencies."
    else:
        label_text = "UNKNOWN"
        instruction = ""

    return f"""Analyze these video frames for signs of deepfake manipulation.
Ground truth label: {label_text}.
{instruction}

Examine the person's face carefully across all provided frames.
Respond ONLY with a valid JSON object — no markdown, no extra text.

{{
  "general_description": "<3-5 sentences: who is in the video, what they are doing, overall visual quality and any notable observations>",
  "face_boundaries": "<edges around face, hair, neck: sharp/blurry/unnatural artifacts?>",
  "skin_texture": "<skin consistency across frames: smooth/waxy/flickering/inconsistent?>",
  "eye_behavior": "<eye movement and blinking: natural/rare/asymmetric/frozen?>",
  "lip_sync": "<mouth movement vs speech: aligned/delayed/stiff/unnatural?>",
  "lighting_consistency": "<lighting on face vs background: consistent/mismatched/shadow artifacts?>",
  "temporal_stability": "<face appearance across frames: stable/flickering/warping/identity shifts?>",
  "overall_suspicion": "<none|low|medium|high>"
}}"""


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, quantize: bool, device: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model: {model_id}")

    if quantize:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            if allocated_gb > 10:
                print(f"  [!] Warning: bitsandbytes loaded model at {allocated_gb:.1f} GB — quantization likely failed.")
                print(f"  [!] Falling back to float16.")
                del model
                torch.cuda.empty_cache()
                raise RuntimeError("bitsandbytes quantization did not reduce model size")
            print(f"Quantization: 4-bit  ({allocated_gb:.1f} GB)")
        except Exception as e:
            print(f"  bitsandbytes failed ({e}), loading in float16 instead.")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"Quantization: float16 fallback  ({allocated_gb:.1f} GB)")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"Quantization: float16  ({allocated_gb:.1f} GB)")

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("Model loaded.\n")
    return model, processor


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(model, processor, frames: list, prompt: str, max_new_tokens: int, max_pixels: int) -> str:
    """Run Qwen2.5-VL on a list of PIL frames with a text prompt."""
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=frames,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output[0].strip()


def process_sample(sample_path: Path, model, processor, args) -> dict:
    """
    Full pipeline for one sample folder:
    1. Load and sample extracted frames
    2. Run inference (with retry on JSON parse failure)
    3. Return parsed result dict
    """
    t_start = time.time()

    frames = sample_images_from_folder(
        sample_path,
        args.num_frames,
        args.trim_edges,
        args.max_pixels,
    )

    raw_output = None
    parsed = None
    last_error = None

    label = get_label_from_path(sample_path, args.dataset)
    main_prompt = build_prompt(label)
    prompts = [main_prompt] + [FALLBACK_PROMPT] * (args.retry_attempts - 1)

    if len(prompts) < args.retry_attempts:
        prompts += [FALLBACK_PROMPT] * (args.retry_attempts - len(prompts))

    for attempt, prompt in enumerate(prompts[:args.retry_attempts], 1):
        try:
            raw_output = run_inference(model, processor, frames, prompt, args.max_new_tokens, args.max_pixels)
            parsed = parse_model_json(raw_output)
            break
        except Exception as e:
            last_error = str(e)
            if attempt < args.retry_attempts:
                free_cuda_memory()

    if parsed is None:
        raise ValueError(
            f"JSON parse failed after {args.retry_attempts} attempts. "
            f"Last error: {last_error}. Raw output: {repr(raw_output)}"
        )

    elapsed = round(time.time() - t_start, 2)

    return {
        "parsed": parsed,
        "raw_output": raw_output,
        "frames_sampled": len(frames),
        "processing_time_sec": elapsed,
        "label": label,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.is_dir():
        print(f"[ERROR] Dataset path does not exist: {dataset_root}")
        sys.exit(1)

    output_dir = dataset_root / args.captions_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    skipped_path = output_dir / "skipped.json"

    device = get_device(args.device)

    run_meta = {
        "_meta": {
            "model": args.model,
            "quantize": args.quantize,
            "num_frames": args.num_frames,
            "fps_sampling": args.fps,
            "trim_edges": args.trim_edges,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_root),
            "input_type": "frames_folders",
        }
    }

    results = load_json_safe(results_path) if args.resume else {}
    skipped = load_json_safe(skipped_path) if args.resume else {}

    if not results:
        results = run_meta.copy()

    if args.resume:
        print(f"Resuming: {len(results) - 1} already processed, {len(skipped)} skipped.\n")

    sample_folders = collect_frame_folders(dataset_root, output_dir)
    if not sample_folders:
        print("No 'frames' folders found. Exiting.")
        sys.exit(0)

    print(f"Found {len(sample_folders)} frame folder(s).\n")

    if args.resume:
        done_keys = (set(results.keys()) | set(skipped.keys())) - {"_meta"}
        sample_folders = [p for p in sample_folders if str(p.relative_to(dataset_root)) not in done_keys]
        print(f"Remaining to process: {len(sample_folders)}\n")

    if not sample_folders:
        print("Nothing to process. Exiting.")
        sys.exit(0)

    if args.fps is not None:
        print("[INFO] --fps is ignored for pre-extracted frame folders; using --num-frames sampling.\n")

    model, processor = load_model(args.model, args.quantize, device)

    counters = {"success": 0, "skipped": 0}

    for i, sample_path in enumerate(tqdm(sample_folders, desc="Samples", unit="sample"), 1):
        rel_key = str(sample_path.relative_to(dataset_root))
        tqdm.write(f"\n[{i}/{len(sample_folders)}] {rel_key}")

        try:
            result = process_sample(sample_path, model, processor, args)

            results[rel_key] = {
                "path": rel_key,
                "label": result["label"],
                "depth": result["parsed"],
                "raw_output": result["raw_output"],
                "frames_sampled": result["frames_sampled"],
                "processing_time_sec": result["processing_time_sec"],
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            tqdm.write(f"  ✓ suspicion={result['parsed'].get('overall_suspicion', '?')}  ({result['processing_time_sec']}s)")
            counters["success"] += 1

        except Exception as e:
            reason = traceback.format_exc()
            skipped[rel_key] = {
                "path": rel_key,
                "reason": str(e),
                "traceback": reason,
                "skipped_at": datetime.now(timezone.utc).isoformat(),
            }
            tqdm.write(f"  ✗ Skipped: {e}")
            counters["skipped"] += 1

        finally:
            free_cuda_memory()
            save_json(results, results_path)
            if skipped:
                save_json(skipped, skipped_path)

    print(f"\n{'─'*50}")
    print(f"Done.  Success: {counters['success']} | Skipped: {counters['skipped']}")
    print(f"Results → {results_path}")
    if skipped:
        print(f"Skipped → {skipped_path}")


if __name__ == "__main__":
    main()