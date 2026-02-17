import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset
from transformers import ClapModel, ClapProcessor

# =====================
# 1) CLI Configuration
# =====================
parser = argparse.ArgumentParser(
    description="BioLingual/CLAP audio embeddings + MLP classifier training."
)

parser.add_argument("--manifest_train", type=Path, required=True)
parser.add_argument("--manifest_val", type=Path, required=True)
parser.add_argument("--manifest_test", type=Path, required=True)

parser.add_argument("--model_path", type=Path, required=True, help="Local BioLingual/CLAP model directory")
parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")

parser.add_argument("--target_sr", type=int, default=48000)
parser.add_argument("--batch_audio_emb", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=0)

parser.add_argument("--batch_mlp", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--dropout", type=float, default=0.4)

parser.add_argument("--cache_emb", action="store_true", help="Cache train/val embeddings as .npy")

args = parser.parse_args()

MANIFEST_TRAIN = args.manifest_train
MANIFEST_VAL = args.manifest_val
MANIFEST_TEST = args.manifest_test

MODEL_PATH = args.model_path

OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = args.target_sr
BATCH_AUDIO_EMB = args.batch_audio_emb
NUM_WORKERS = args.num_workers

BATCH_MLP = args.batch_mlp
EPOCHS = args.epochs
LR = args.lr
WEIGHT_DECAY = args.weight_decay
PATIENCE = args.patience
DROPOUT = args.dropout

CACHE_EMB = args.cache_emb

# =====================
# 2) Utils
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

resampler_cache = {}

def read_manifest(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    req = {"clip_path", "true_scientific_name"}
    if not req.issubset(df.columns):
        raise ValueError(f"{p} must contain columns {req}. Found: {df.columns.tolist()}")
    df["clip_path"] = df["clip_path"].astype(str)
    df["true_scientific_name"] = df["true_scientific_name"].astype(str).str.strip()
    return df


def load_audio_mono_resample(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        key = (sr, TARGET_SR)
        if key not in resampler_cache:
            resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        wav = resampler_cache[key](wav)

    return wav.squeeze(0).numpy()


@torch.no_grad()
def compute_embeddings(df: pd.DataFrame, processor, clap_model, batch_size: int) -> np.ndarray:
    """Return L2-normalized audio embeddings with shape [N, D]."""
    paths = df["clip_path"].tolist()
    all_embs = []
    n = len(paths)

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        audios = [load_audio_mono_resample(p) for p in batch_paths]

        inputs = processor(audios=audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        emb = clap_model.get_audio_features(**inputs)
        emb = normalize(emb, dim=-1)
        all_embs.append(emb.detach().cpu().numpy())

        if end % 500 == 0 or end == n:
            elapsed = time.perf_counter() - t0
            cps = end / elapsed if elapsed > 0 else 0.0
            print(f"Embeddings: {end}/{n} | {cps:.2f} clips/s")

    return np.vstack(all_embs)


def make_label_mapping(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    classes = sorted(df_train["true_scientific_name"].unique().tolist())

    extra_val = set(df_val["true_scientific_name"]) - set(classes)
    extra_test = set(df_test["true_scientific_name"]) - set(classes)
    if extra_val or extra_test:
        raise ValueError(f"VAL/TEST contain labels not present in TRAIN: val={extra_val}, test={extra_test}")

    name_to_idx = {n: i for i, n in enumerate(classes)}
    return classes, name_to_idx


def df_to_y(df: pd.DataFrame, name_to_idx: dict) -> np.ndarray:
    return np.array([name_to_idx[n] for n in df["true_scientific_name"].tolist()], dtype=np.int64)

# =====================
# 3) MLP model
# =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    all_logits = []
    all_y = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            all_logits.append(logits.detach().cpu())
            all_y.append(yb.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    return y_true, y_pred, logits.numpy()

# =====================
# 4) Main
# =====================
def main():
    # 4.1) Load manifests
    df_train = read_manifest(MANIFEST_TRAIN)
    df_val = read_manifest(MANIFEST_VAL)
    df_test = read_manifest(MANIFEST_TEST)

    print("Train clips:", len(df_train))
    print("Val clips  :", len(df_val))
    print("Test clips :", len(df_test))

    # 4.2) Label mapping
    classes, name_to_idx = make_label_mapping(df_train, df_val, df_test)
    print("Num classes:", len(classes))

    y_train = df_to_y(df_train, name_to_idx)
    y_val = df_to_y(df_val, name_to_idx)
    y_test = df_to_y(df_test, name_to_idx)

    # 4.3) Load CLAP (BioLingual)
    processor = ClapProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    clap = ClapModel.from_pretrained(str(MODEL_PATH), local_files_only=True).to(device).eval()

    # =====================
    # 5) Compute embeddings
    # =====================
    cache_dir = OUT_DIR / "cache_embeddings"
    cache_dir.mkdir(exist_ok=True, parents=True)

    def cache_path(split: str) -> Path:
        return cache_dir / f"X_{split}.npy"

    if CACHE_EMB and cache_path("train").exists():
        X_train = np.load(cache_path("train"))
        print("Loaded cached train embeddings:", X_train.shape)
    else:
        X_train = compute_embeddings(df_train, processor, clap, BATCH_AUDIO_EMB)
        if CACHE_EMB:
            np.save(cache_path("train"), X_train)

    if CACHE_EMB and cache_path("val").exists():
        X_val = np.load(cache_path("val"))
        print("Loaded cached val embeddings:", X_val.shape)
    else:
        X_val = compute_embeddings(df_val, processor, clap, BATCH_AUDIO_EMB)
        if CACHE_EMB:
            np.save(cache_path("val"), X_val)

    t_test_emb0 = time.perf_counter()
    X_test = compute_embeddings(df_test, processor, clap, BATCH_AUDIO_EMB)
    t_test_emb = time.perf_counter() - t_test_emb0
    print(f"Test embedding time (s): {t_test_emb:.2f}")

    input_dim = X_train.shape[1]

    # =====================
    # 6) DataLoaders
    # =====================
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_MLP, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_MLP, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_MLP, shuffle=False, num_workers=NUM_WORKERS
    )

    # =====================
    # 7) Train MLP (early stopping on VAL accuracy)
    # =====================
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(classes)), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    mlp = MLPClassifier(input_dim, len(classes), dropout=DROPOUT).to(device)
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print("Trainable params (MLP):", trainable_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_epoch = -1
    bad = 0
    history = {"train_loss": [], "val_acc": []}

    t_train0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        mlp.train()
        running = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = mlp(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        yv_true, yv_pred, _ = evaluate(mlp, val_loader)
        val_acc = accuracy_score(yv_true, yv_pred)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            bad = 0
            torch.save(mlp.state_dict(), OUT_DIR / "mlp_best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping. Best epoch={best_epoch} val_acc={best_val_acc:.4f}")
                break

    t_train = time.perf_counter() - t_train0
    print(f"Train time total (s): {t_train:.2f}")

    # =====================
    # 8) Test evaluation + outputs
    # =====================
    mlp.load_state_dict(torch.load(OUT_DIR / "mlp_best.pt", map_location=device))

    t_test0 = time.perf_counter()
    yt_true, yt_pred, logits = evaluate(mlp, test_loader)
    t_test = time.perf_counter() - t_test0

    t_inference_total = t_test_emb + t_test
    avg_ms_clip = (t_inference_total / len(df_test)) * 1000 if len(df_test) else 0.0
    clips_per_sec = len(df_test) / t_inference_total if t_inference_total > 0 else 0.0

    test_acc = accuracy_score(yt_true, yt_pred)

    report_dict = classification_report(
        yt_true, yt_pred, target_names=classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).T.to_csv(
        OUT_DIR / "classification_report_test.csv", encoding="utf-8"
    )

    cm = confusion_matrix(yt_true, yt_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        OUT_DIR / "confusion_matrix_test.csv", encoding="utf-8"
    )

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    topk = min(5, probs.shape[1])
    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    topk_scores = np.take_along_axis(probs, topk_idx, axis=1)

    pred_rows = []
    for i in range(len(df_test)):
        pred_rows.append({
            "clip_path": df_test.iloc[i]["clip_path"],
            "true_scientific_name": df_test.iloc[i]["true_scientific_name"],
            "pred_scientific_name": classes[int(yt_pred[i])],
            "pred_prob": float(probs[i, yt_pred[i]]),
            "topk_pred_scientific": ";".join([classes[j] for j in topk_idx[i].tolist()]),
            "topk_probs": ";".join([f"{s:.6f}" for s in topk_scores[i].tolist()]),
        })

    pd.DataFrame(pred_rows).to_csv(
        OUT_DIR / "predictions_test.csv", index=False, encoding="utf-8"
    )

    # =====================
    # 9) Training curves
    # =====================
    plt.figure()
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("MLP training loss")
    plt.savefig(OUT_DIR / "train_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Val accuracy")
    plt.title("MLP validation accuracy")
    plt.savefig(OUT_DIR / "val_acc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =====================
    # 10) Summary + metadata
    # =====================
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== BioLingual + MLP (fixed embeddings) ===\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Train clips: {len(df_train)}\n")
        f.write(f"Val clips:   {len(df_val)}\n")
        f.write(f"Test clips:  {len(df_test)}\n")
        f.write(f"Num classes: {len(classes)}\n\n")

        f.write(f"Embedding batch size: {BATCH_AUDIO_EMB}\n")
        f.write(f"MLP batch size: {BATCH_MLP}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Epochs max: {EPOCHS}\n")
        f.write(f"Early stopping patience: {PATIENCE}\n\n")

        f.write(f"Best VAL acc: {best_val_acc:.6f} (epoch {best_epoch})\n")
        f.write(f"TEST accuracy: {test_acc:.6f}\n\n")

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
        f.write(f"Train time (s): {t_train:.2f}\n")
        f.write(f"Test embedding time (s): {t_test_emb:.2f}\n")
        f.write(f"Test classifier time (s): {t_test:.2f}\n")
        f.write(f"Inference total (s): {t_inference_total:.2f}\n")
        f.write(f"Avg ms/clip: {avg_ms_clip:.2f}\n")
        f.write(f"Clips/sec: {clips_per_sec:.2f}\n")
        f.write(f"Trainable parameters (MLP): {trainable_params}\n")

    with open(OUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    print("\n✅ DONE")
    print("Best VAL acc:", best_val_acc, "epoch:", best_epoch)
    print("TEST acc:", test_acc)
    print("Outputs in:", OUT_DIR)