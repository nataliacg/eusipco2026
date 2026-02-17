import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.functional import normalize
from transformers import ClapModel, ClapProcessor

# =====================
# CLI configuration
# =====================
parser = argparse.ArgumentParser(
    description="BioLingual/CLAP zero-shot evaluation using a manifest CSV."
)

parser.add_argument(
    "--manifest",
    type=Path,
    required=True,
    help="Path to manifest CSV with columns: clip_path,true_scientific_name",
)
parser.add_argument(
    "--model_path",
    type=Path,
    required=True,
    help="Path to a local BioLingual/CLAP model directory (Hugging Face format).",
)
parser.add_argument(
    "--results_dir",
    type=Path,
    required=True,
    help="Directory where outputs will be saved (CSVs, confusion matrix, summary).",
)

parser.add_argument("--target_sr", type=int, default=48000, help="Target sample rate (Hz).")
parser.add_argument("--batch_audio", type=int, default=16, help="Audio batch size.")
parser.add_argument("--batch_text", type=int, default=64, help="Text batch size.")
parser.add_argument("--topk_save", type=int, default=5, help="Top-K predictions to store per clip.")

parser.add_argument(
    "--prompt",
    type=str,
    default="A field recording of the bird {sci}.",
    help="Prompt template. Use {sci} as placeholder for the scientific name.",
)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda"],
    help="Compute device to use.",
)

t_script0 = time.perf_counter()
args = parser.parse_args()

# Resolve and validate paths
manifest_path = args.manifest.expanduser().resolve()
model_path = args.model_path.expanduser().resolve()
results_dir = args.results_dir.expanduser().resolve()
results_dir.mkdir(parents=True, exist_ok=True)

if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest not found: {manifest_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model path not found: {model_path}")

# Runtime config
TARGET_SR = int(args.target_sr)
BATCH_AUDIO = int(args.batch_audio)
BATCH_TEXT = int(args.batch_text)
TOPK_SAVE = int(args.topk_save)
PROMPT_TEMPLATE = args.prompt

# Device selection
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

# =====================
# 1) Cargar manifest TEST 
# =====================
df = pd.read_csv(manifest_path)
required_cols = {"clip_path", "true_scientific_name"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"El manifest TEST debe tener columnas {required_cols}. Faltan: {missing}")

df["clip_path"] = df["clip_path"].astype(str)
df["true_scientific_name"] = df["true_scientific_name"].astype(str).str.strip()

print("Clips evaluados:", len(df))

# =====================
# 2) Build label space from the test manifest
# =====================
classes = sorted(df["true_scientific_name"].unique().tolist())
print("Number of classes (TEST):", len(classes))

name_to_idx = {name: i for i, name in enumerate(classes)}
idx_to_name = {i: name for name, i in name_to_idx.items()}

# Build text prompts for each class
# Folder names use "_" but prompts are more natural with spaces
prompts = [PROMPT_TEMPLATE.format(sci=name.replace("_", " ")) for name in classes]

# =====================
# 3) Load BioLingual / CLAP model
# =====================
processor = ClapProcessor.from_pretrained(str(model_path), local_files_only=True)
model = ClapModel.from_pretrained(str(model_path), local_files_only=True).to(device)
model.eval()

# =====================
# 4) Pre-encode text prompts
# =====================
t_text0 = time.perf_counter()
text_embeds_list = []

with torch.no_grad():
    for i in range(0, len(prompts), BATCH_TEXT):
        batch_prompts = prompts[i:i + BATCH_TEXT]
        text_inputs = processor(text=batch_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        emb = model.get_text_features(**text_inputs)
        emb = normalize(emb, dim=-1)
        text_embeds_list.append(emb)

text_embeds = torch.cat(text_embeds_list, dim=0)
t_text = time.perf_counter() - t_text0
print(f"Text encoding time: {t_text:.2f}s")


# =====================
# 5) Audio loader (mono + resample)
# =====================
resampler_cache = {}

def load_audio_mono_resample(path: str) -> np.ndarray:
    """Load audio, convert to mono, and resample to TARGET_SR if needed."""
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        key = (sr, TARGET_SR)
        if key not in resampler_cache:
            resampler_cache[key] = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=TARGET_SR
            )
        wav = resampler_cache[key](wav)

    return wav.squeeze(0).numpy()

# =====================
# 6) Batch evaluation
# =====================
y_true, y_pred = [], []
pred_rows = []

paths = df["clip_path"].tolist()
true_names = df["true_scientific_name"].tolist()

t1 = time.perf_counter()

num = len(paths)
for start in range(0, num, BATCH_AUDIO):
    end = min(start + BATCH_AUDIO, num)

    batch_paths = paths[start:end]
    batch_true_names = true_names[start:end]

    audios = [load_audio_mono_resample(p) for p in batch_paths]

    inputs = processor(audios=audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_embeds = model.get_audio_features(**inputs)
        audio_embeds = normalize(audio_embeds, dim=-1)

        sims = audio_embeds @ text_embeds.T
        topk_scores, topk_idx = torch.topk(sims, k=min(TOPK_SAVE, sims.shape[1]), dim=1)

    topk_scores_cpu = topk_scores.detach().cpu().numpy()
    topk_idx_cpu = topk_idx.detach().cpu().numpy()

    for i in range(end - start):
        true_name = batch_true_names[i]
        true_idx = name_to_idx[true_name]

        pred_idx = int(topk_idx_cpu[i, 0])
        pred_name = idx_to_name[pred_idx]
        pred_score = float(topk_scores_cpu[i, 0])

        y_true.append(true_idx)
        y_pred.append(pred_idx)

        tk_idx = topk_idx_cpu[i].tolist()
        tk_scores = topk_scores_cpu[i].tolist()

        pred_rows.append({
            "clip_path": batch_paths[i],
            "true_scientific_name": true_name,
            "pred_scientific_name": pred_name,
            "pred_score": pred_score,
            "topk_pred_scientific": ";".join([idx_to_name[j] for j in tk_idx]),
            "topk_scores": ";".join([f"{s:.6f}" for s in tk_scores]),
        })

    if end % 500 == 0 or end == num:
        elapsed = time.perf_counter() - t1
        cps = end / elapsed if elapsed > 0 else 0
        print(f"Processed {end}/{num} | {cps:.2f} clips/s")

t_audio = time.perf_counter() - t1
t_total = time.perf_counter() - t_script0

# =====================
# 7) Metrics + saving
# =====================
acc = accuracy_score(y_true, y_pred)

report_dict = classification_report(
    y_true, y_pred, target_names=classes, zero_division=0, output_dict=True
)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(results_dir / "classification_report.csv", encoding="utf-8")

cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    results_dir / "confusion_matrix.csv", encoding="utf-8"
)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(results_dir / "predictions.csv", index=False, encoding="utf-8")

clips_per_sec = len(df) / t_audio if t_audio > 0 else 0.0
avg_ms_clip = (t_audio / len(df)) * 1000 if len(df) > 0 else 0.0

with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Num clips: {len(df)}\n")
    f.write(f"Num classes: {len(classes)}\n")
    f.write(f"Prompt template: {PROMPT_TEMPLATE}\n")
    f.write(f"TARGET_SR: {TARGET_SR}\n")
    f.write(f"BATCH_AUDIO: {BATCH_AUDIO}\n")
    f.write(f"BATCH_TEXT: {BATCH_TEXT}\n")
    f.write(f"TOPK_SAVE: {TOPK_SAVE}\n\n")

    f.write(f"Accuracy: {acc:.6f}\n\n")

    if "macro avg" in report_dict:
        f.write(
            "Macro avg (precision/recall/f1): "
            f"{report_dict['macro avg']['precision']:.4f} / "
            f"{report_dict['macro avg']['recall']:.4f} / "
            f"{report_dict['macro avg']['f1-score']:.4f}\n"
        )
    if "weighted avg" in report_dict:
        f.write(
            "Weighted avg (precision/recall/f1): "
            f"{report_dict['weighted avg']['precision']:.4f} / "
            f"{report_dict['weighted avg']['recall']:.4f} / "
            f"{report_dict['weighted avg']['f1-score']:.4f}\n"
        )

    f.write("\n--- Timing ---\n")
    f.write(f"Text encoding time (s): {t_text:.2f}\n")
    f.write(f"Audio eval time (s): {t_audio:.2f}\n")
    f.write(f"Total time (s): {t_total:.2f}\n")
    f.write(f"Clips/sec: {clips_per_sec:.2f}\n")
    f.write(f"Avg ms/clip: {avg_ms_clip:.2f}\n")

print("\nAccuracy:", acc)
print("Saved to:", results_dir)
print(f"Total time: {t_total:.2f}s | {clips_per_sec:.2f} clips/s | {avg_ms_clip:.2f} ms/clip")