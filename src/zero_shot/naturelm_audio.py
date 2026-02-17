import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from NatureLM.infer import Pipeline
from NatureLM.models import NatureLM

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# =====================
# 1) CLI Configuration
# =====================
import argparse

parser = argparse.ArgumentParser(
    description="NatureLM evaluation using a manifest CSV (closed-set query)."
)

parser.add_argument(
    "--manifest",
    type=Path,
    required=True,
    help="Path to manifest CSV with columns: clip_path,true_scientific_name",
)

parser.add_argument(
    "--results_dir",
    type=Path,
    required=True,
    help="Directory where outputs will be saved.",
)

parser.add_argument("--window_sec", type=int, default=3)
parser.add_argument("--hop_sec", type=int, default=3)
parser.add_argument("--batch", type=int, default=8)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda"],
    help="Compute device to use.",
)

args = parser.parse_args()

manifest_path = args.manifest
results_dir = args.results_dir
results_dir.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = args.window_sec
HOP_SEC = args.hop_sec
BATCH = args.batch

if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

print("Device:", device)

# =====================
# 2) Helpers
# =====================
def clean_pred(raw: str) -> str:
    """Convert '#0.00s - 3.00s#: Acrocephalus arundinaceus\\n' -> 'Acrocephalus arundinaceus'."""
    if raw is None:
        return ""
    raw = str(raw)
    raw = re.sub(r"#.*?#:\s*", "", raw)
    raw = raw.replace("\n", " ").strip()
    raw = raw.strip().strip(".")
    return raw


def to_manifest_label(name_spaces: str) -> str:
    """Convert 'Acrocephalus arundinaceus' -> 'Acrocephalus_arundinaceus'."""
    return name_spaces.strip().replace(" ", "_")



# =====================
# 3) Load test manifest
# =====================
df = pd.read_csv(manifest_path)

required_cols = {"clip_path", "true_scientific_name"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Manifest must contain columns {required_cols}. Missing: {missing}"
    )

df["clip_path"] = df["clip_path"].astype(str)
df["true_scientific_name"] = df["true_scientific_name"].astype(str).str.strip()

print("Num clips:", len(df))

# =====================
# 4) Class list + query
# =====================
classes_unders = sorted(df["true_scientific_name"].unique().tolist())
CLASSES_SPACES = [x.replace("_", " ") for x in classes_unders]

allowed_spaces = set(CLASSES_SPACES)
allowed_unders = set(to_manifest_label(x) for x in CLASSES_SPACES)

classes = classes_unders
name_to_idx = {name: i for i, name in enumerate(classes)}
idx_to_name = {i: name for name, i in name_to_idx.items()}

PROMPT_TEMPLATE = (
    "Which bird species is the focal species in the audio recording?\n"
    "Choose exactly one scientific name from the following list:\n"
    "{options}.\n"
    "Answer with exactly one scientific name and no additional text."
)

QUERY = PROMPT_TEMPLATE.format(options=", ".join(CLASSES_SPACES))

print("Num classes:", len(classes))

# =====================
# 5) Load model + pipeline
# =====================
t0 = time.perf_counter()

model = NatureLM.from_pretrained(
    "EarthSpeciesProject/NatureLM-audio"
).eval().to(device)

try:
    gc = model.generation_config
    if gc.pad_token_id is None and gc.eos_token_id is not None:
        gc.pad_token_id = gc.eos_token_id
except Exception:
    pass

pipeline = Pipeline(model=model)


# =====================
# 6) Evaluation setup
# =====================
y_true, y_pred = [], []
pred_rows = []
raw_rows = []
failed_paths = []

paths = df["clip_path"].tolist()
true_names = df["true_scientific_name"].tolist()

t_audio0 = time.perf_counter()
num_out_of_set = 0

total_batches = (len(paths) + BATCH - 1) // BATCH

# =====================
# 7) Batched inference
# =====================
for i in tqdm(
    range(0, len(paths), BATCH),
    total=total_batches,
    desc="NatureLM zero-shot (closed-set)",
):
    batch_paths = paths[i:i + BATCH]
    batch_gt = true_names[i:i + BATCH]

    send_paths = []
    send_gt = []
    send_idx_map = []

    for j, (audio_path, gt) in enumerate(zip(batch_paths, batch_gt)):
        if gt not in name_to_idx:
            pred_rows.append({
                "clip_path": audio_path,
                "true_scientific_name": gt,
                "pred_scientific_name": "GT_NOT_IN_CLASSES",
                "pred_score": "",
                "topk_pred_scientific": "",
                "topk_scores": "",
            })
        else:
            send_paths.append(audio_path)
            send_gt.append(gt)
            send_idx_map.append(j)

    if not send_paths:
        continue

    batch_queries = [QUERY] * len(send_paths)

    try:
        raw_preds = pipeline(
            send_paths,
            batch_queries,
            window_length_seconds=WINDOW_SEC,
            hop_length_seconds=HOP_SEC,
        )
    except Exception:
        for audio_path, gt in zip(send_paths, send_gt):
            failed_paths.append(audio_path)
            pred_rows.append({
                "clip_path": audio_path,
                "true_scientific_name": gt,
                "pred_scientific_name": "ERROR_PIPELINE",
                "pred_score": "",
                "topk_pred_scientific": "",
                "topk_scores": "",
            })
        continue

    for audio_path, gt, raw_pred in zip(send_paths, send_gt, raw_preds):
        pred_clean = clean_pred(raw_pred)

        raw_rows.append({
            "clip_path": audio_path,
            "true_scientific_name": gt,
            "raw_output": raw_pred,
            "clean_output": pred_clean,
        })

        pred_raw_name = pred_clean
        pred_label = to_manifest_label(pred_clean) if pred_clean in allowed_spaces else None

        pred_rows.append({
            "clip_path": audio_path,
            "true_scientific_name": gt,
            "pred_scientific_name": pred_raw_name,
            "pred_score": "",
            "topk_pred_scientific": "",
            "topk_scores": "",
        })

        gt_idx = name_to_idx[gt]
        y_true.append(gt_idx)

        if pred_label is not None and pred_label in name_to_idx:
            y_pred.append(name_to_idx[pred_label])
        else:
            num_out_of_set += 1
            wrong_idx = (gt_idx + 1) % len(classes)
            y_pred.append(wrong_idx)

t_audio = time.perf_counter() - t_audio0
t_total = time.perf_counter() - t0

# =====================
# 8) Metrics + saving
# =====================
n_eval = len(y_true)
acc = accuracy_score(y_true, y_pred) if n_eval > 0 else 0.0

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    zero_division=0,
    output_dict=True,
) if n_eval > 0 else {}

report_df = pd.DataFrame(report_dict).T if report_dict else pd.DataFrame()
report_df.to_csv(results_dir / "classification_report.csv", encoding="utf-8")

cm = confusion_matrix(y_true, y_pred) if n_eval > 0 else [[0] * len(classes) for _ in range(len(classes))]
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    results_dir / "confusion_matrix.csv",
    encoding="utf-8",
)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(results_dir / "predictions.csv", index=False, encoding="utf-8")

if raw_rows:
    pd.DataFrame(raw_rows).to_csv(results_dir / "raw_outputs.csv", index=False, encoding="utf-8")

if failed_paths:
    (results_dir / "failed_files.txt").write_text("\n".join(failed_paths), encoding="utf-8")


# =====================
# 9) Summary + final prints
# =====================
clips_per_sec = len(df) / t_audio if t_audio > 0 else 0.0
avg_ms_clip = (t_audio / len(df)) * 1000 if len(df) > 0 else 0.0

with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Num clips: {len(df)}\n")
    f.write(f"Num classes: {len(classes)}\n")
    f.write(f"Prompt template: {PROMPT_TEMPLATE}\n")
    f.write(f"QUERY: {QUERY}\n")
    f.write(f"WINDOW_SEC: {WINDOW_SEC}\n")
    f.write(f"HOP_SEC: {HOP_SEC}\n\n")

    f.write(f"Num evaluated: {n_eval}\n")
    f.write(f"Num out-of-set predictions: {num_out_of_set}\n")
    f.write(f"Num failed pipeline: {len(failed_paths)}\n")
    f.write(f"Accuracy: {acc:.6f}\n\n")

    if report_dict:
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
    f.write(f"Audio eval time (s): {t_audio:.2f}\n")
    f.write(f"Total time (s): {t_total:.2f}\n")
    f.write(f"Clips/sec: {clips_per_sec:.2f}\n")
    f.write(f"Avg ms/clip: {avg_ms_clip:.2f}\n")

print("\nAccuracy:", acc)
print("Saved to:", results_dir)
print(f"Total time: {t_total:.2f}s | {clips_per_sec:.2f} clips/s | {avg_ms_clip:.2f} ms/clip")