import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from NatureLM.infer import sliding_window_inference_modificado as swi
from NatureLM.models import NatureLM
from NatureLM.config import Config
from NatureLM.processors import NatureLMAudioProcessor
from typing import Optional, Tuple, Dict, List

# =====================
# 1) CLI Configuration
# =====================
parser = argparse.ArgumentParser(
    description="NatureLM audio encoder embeddings + MLP closed-set training/evaluation."
)

parser.add_argument("--manifest_train", type=Path, required=True)
parser.add_argument("--manifest_val", type=Path, required=True)
parser.add_argument("--manifest_test", type=Path, required=True)

parser.add_argument("--out_dir", type=Path, required=True)
parser.add_argument("--config_path", type=str, required=True, help="Path to NatureLM inference.yml")

parser.add_argument("--cache_emb", action="store_true", help="Cache embeddings as .npy in out_dir")

parser.add_argument("--input_sr", type=int, default=16000)
parser.add_argument("--window_sec", type=float, default=3.0)
parser.add_argument("--hop_sec", type=float, default=3.0)

parser.add_argument("--debug_n", type=int, default=None, help="Optional: limit rows per split for debugging")

parser.add_argument("--batch_mlp", type=int, default=32)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--dropout", type=float, default=0.05)

args = parser.parse_args()

# =====================
# 2) Resolved configuration
# =====================
MANIFEST_TRAIN: Path = args.manifest_train
MANIFEST_VAL: Path = args.manifest_val
MANIFEST_TEST: Path = args.manifest_test

OUT_DIR: Path = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH: str = args.config_path
CACHE_EMB: bool = args.cache_emb

INPUT_SR: int = args.input_sr
WINDOW_SEC: float = args.window_sec
HOP_SEC: float = args.hop_sec

DEBUG_N = args.debug_n

BATCH_MLP: int = args.batch_mlp
EPOCHS: int = args.epochs
LR: float = args.lr
WEIGHT_DECAY: float = args.weight_decay
PATIENCE: int = args.patience
DROPOUT: float = args.dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================
# 3) RAM/VRAM helpers
# =====================
_proc = psutil.Process(os.getpid())

def ram_gb_process() -> float:
    """Resident Set Size (RSS) in GB."""
    return _proc.memory_info().rss / (1024**3)

def ram_gb_system():
    """Return (used_gb, total_gb, percent)."""
    vm = psutil.virtual_memory()
    return (vm.used / (1024**3), vm.total / (1024**3), vm.percent)

def vram_stats_gb():
    """Return (alloc_gb, reserved_gb, peak_alloc_gb). If no CUDA -> zeros."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return alloc, reserved, peak
from typing import Optional, Tuple, Dict, List

# =====================
# 4) Helpers
# =====================
def read_manifest(p: Path, debug_n: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(p)

    required = {"clip_path", "true_scientific_name"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{p} must contain columns {required}. Found: {df.columns.tolist()}"
        )

    df["clip_path"] = df["clip_path"].astype(str)
    df["true_scientific_name"] = df["true_scientific_name"].astype(str).str.strip()

    if debug_n is not None:
        df = df.head(debug_n).copy()

    return df


def make_label_mapping(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(df_train["true_scientific_name"].unique().tolist())

    extra_val = set(df_val["true_scientific_name"]) - set(classes)
    extra_test = set(df_test["true_scientific_name"]) - set(classes)
    if extra_val or extra_test:
        raise ValueError(
            f"VAL/TEST contain labels not present in TRAIN: val={sorted(extra_val)}, test={sorted(extra_test)}"
        )

    name_to_idx = {name: i for i, name in enumerate(classes)}
    return classes, name_to_idx


def df_to_y(df: pd.DataFrame, name_to_idx: Dict[str, int]) -> np.ndarray:
    return np.array([name_to_idx[n] for n in df["true_scientific_name"].tolist()], dtype=np.int64)


@torch.no_grad()
def compute_embeddings(
    df: pd.DataFrame,
    processor: NatureLMAudioProcessor,
    model: NatureLM,
    cfg: Config,
    *,
    window_sec: float,
    hop_sec: float,
    input_sr: int,
    device: torch.device,
    desc: str = "Extract embeddings",
) -> np.ndarray:
    
    X: List[np.ndarray] = []
    paths = df["clip_path"].tolist()

    for path in tqdm(paths, desc=desc):
        emb = swi(
            path,
            "",
            processor,
            model,
            cfg,
            window_length_seconds=window_sec,
            hop_length_seconds=hop_sec,
            input_sr=input_sr,
            device=str(device),
        )

        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(f"NatureLM inference did not return a torch.Tensor for: {path}")

        # emb: [T, D] -> mean pool -> [D]
        vec = emb.mean(dim=0)
        vec = normalize(vec.float(), dim=-1)

        X.append(vec.detach().cpu().numpy())

    return np.stack(X, axis=0).astype(np.float32)

# =====================
# 5)  MLP model
# =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
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
# 6) Main
# =====================
def main():
    # =====================
    # Load manifests
    # =====================
    t_total0 = time.perf_counter()

    df_train = read_manifest(MANIFEST_TRAIN, debug_n=DEBUG_N)
    df_val   = read_manifest(MANIFEST_VAL,   debug_n=DEBUG_N)
    df_test  = read_manifest(MANIFEST_TEST,  debug_n=DEBUG_N)

    print("Train clips (debug):", len(df_train))
    print("Val clips (debug):  ", len(df_val))
    print("Test clips (debug): ", len(df_test))

    # =====================
    # Label mapping
    # =====================
    classes, name_to_idx = make_label_mapping(df_train, df_val, df_test)

    y_train = df_to_y(df_train, name_to_idx)
    y_val   = df_to_y(df_val, name_to_idx)
    y_test  = df_to_y(df_test, name_to_idx)

    # =====================
    # Load NatureLM + config
    # =====================
    cfg = Config.from_sources(CONFIG_PATH)
    processor = NatureLMAudioProcessor(sample_rate=INPUT_SR, max_length_seconds=10)
    model_nlm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio").eval().to(device)
    # =====================
    # 4) Embedding extraction (with cache)
    # =====================

    X_train_path = OUT_DIR / "X_train.npy"
    X_val_path   = OUT_DIR / "X_val.npy"
    X_test_path  = OUT_DIR / "X_test.npy"

    # Reset CUDA peak memory stats before embedding stage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_emb0 = time.perf_counter()
    t_test_emb = 0.0
    if CACHE_EMB and X_train_path.exists() and X_val_path.exists() and X_test_path.exists():

        print("Loading embeddings from cache...")

        X_train = np.load(X_train_path)
        X_val   = np.load(X_val_path)
        X_test  = np.load(X_test_path)

        if (
            len(X_train) != len(df_train) or
            len(X_val)   != len(df_val)   or
            len(X_test)  != len(df_test)
        ):
            raise ValueError(
                "Cached embeddings do not match manifest sizes. "
                "Delete .npy files or set CACHE_EMB=False."
            )

    else:

        print("Computing embeddings with NatureLM...")

        X_train = compute_embeddings(df_train, processor, model_nlm, cfg)
        X_val   = compute_embeddings(df_val,   processor, model_nlm, cfg)

        t_test_emb0 = time.perf_counter()
        X_test  = compute_embeddings(df_test, processor, model_nlm, cfg)
        t_test_emb = time.perf_counter() - t_test_emb0

        t_emb = time.perf_counter() - t_emb0
        print(f"Embedding time total: {t_emb:.2f}s")

        np.save(X_train_path, X_train)
        np.save(X_val_path,   X_val)
        np.save(X_test_path,  X_test)

    # Final embedding timing
    t_emb = time.perf_counter() - t_emb0

    print(
        f"Embedding stage total: {t_emb:.2f}s "
        f"(train={len(df_train)}, val={len(df_val)}, test={len(df_test)})"
    )

    # =====================
    # Resource usage after embedding stage
    # =====================

    emb_alloc, emb_reserved, emb_peak = vram_stats_gb()
    ram_proc_emb = ram_gb_process()
    used_sys_emb, total_sys_emb, pct_emb = ram_gb_system()

    # =====================
    # Save manifests used for reproducibility
    # =====================

    df_train[["clip_path", "true_scientific_name"]].to_csv(
        OUT_DIR / "train_manifest_used.csv", index=False
    )

    df_val[["clip_path", "true_scientific_name"]].to_csv(
        OUT_DIR / "val_manifest_used.csv", index=False
    )

    df_test[["clip_path", "true_scientific_name"]].to_csv(
        OUT_DIR / "test_manifest_used.csv", index=False
    )    
    # =====================
    # DataLoaders
    # =====================

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_MLP,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_MLP,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_MLP,
        shuffle=False,
        num_workers=0
    )


    # =====================
    # Loss + Model setup
    # =====================

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(len(classes)),
        y=y_train
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    mlp = MLPClassifier(
        X_train.shape[1],
        len(classes),
        dropout=DROPOUT
    ).to(device)

    trainable_params = sum(
        p.numel() for p in mlp.parameters() if p.requires_grad
    )

    print("Trainable parameters (MLP):", trainable_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        mlp.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    # =====================
    # Training loop
    # =====================

    best_val_acc = -1.0
    best_epoch = -1
    bad = 0

    t_train0 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, EPOCHS + 1):

        mlp.train()
        running = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = criterion(mlp(xb), yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        yv_true, yv_pred, _ = evaluate(mlp, val_loader)
        val_acc = accuracy_score(yv_true, yv_pred)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            bad = 0
            torch.save(mlp.state_dict(), OUT_DIR / "mlp_best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(
                    f"Early stopping. "
                    f"Best epoch={best_epoch} "
                    f"val_acc={best_val_acc:.4f}"
                )
                break


    # =====================
    # Training timing + resources
    # =====================

    t_train = time.perf_counter() - t_train0
    print(f"Train time total: {t_train:.2f}s")

    train_alloc, train_reserved, train_peak = vram_stats_gb()
    ram_proc_train = ram_gb_process()
    used_sys_train, total_sys_train, pct_train = ram_gb_system()

    # =====================
    # test evaluation
    # =====================

    mlp.load_state_dict(
        torch.load(OUT_DIR / "mlp_best.pt", map_location=device)
    )

    t_test0 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    yt_true, yt_pred, logits = evaluate(mlp, test_loader)

    t_test = time.perf_counter() - t_test0

    # Inference timing (embeddings + classifier)
    t_infer_total = t_test_emb + t_test
    avg_ms_clip_total = (t_infer_total / len(df_test)) * 1000
    clips_per_sec_total = len(df_test) / t_infer_total

    # Resource stats
    test_alloc, test_reserved, test_peak = vram_stats_gb()
    ram_proc_test = ram_gb_process()
    used_sys_test, total_sys_test, pct_test = ram_gb_system()

    cps = len(df_test) / t_test if t_test > 0 else 0.0
    avg_ms = (t_test / len(df_test)) * 1000 if len(df_test) > 0 else 0.0

    print(
        f"Test eval time: {t_test:.2f}s | "
        f"{cps:.2f} clips/s | "
        f"{avg_ms:.2f} ms/clip"
    )


    # =====================
    # Metrics
    # =====================

    test_acc = accuracy_score(yt_true, yt_pred)

    report_dict = classification_report(
        yt_true,
        yt_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).T.to_csv(
        OUT_DIR / "classification_report_test.csv",
        encoding="utf-8"
    )

    cm = confusion_matrix(yt_true, yt_pred)

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    ).to_csv(
        OUT_DIR / "confusion_matrix_test.csv",
        encoding="utf-8"
    )


    # =====================
    # Predictions export
    # =====================

    probs = torch.softmax(
        torch.from_numpy(logits),
        dim=1
    ).numpy()

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
            "topk_pred_scientific": ";".join(
                [classes[j] for j in topk_idx[i].tolist()]
            ),
            "topk_probs": ";".join(
                [f"{s:.6f}" for s in topk_scores[i].tolist()]
            ),
        })

    pd.DataFrame(pred_rows).to_csv(
        OUT_DIR / "predictions_test.csv",
        index=False,
        encoding="utf-8"
    )

    with open(OUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)


    # =====================
    # Summary
    # =====================

    t_total = time.perf_counter() - t_total0

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:

        f.write("=== NatureLM audio encoder + MLP (debug2) ===\n\n")

        f.write(f"Best VAL acc: {best_val_acc:.6f} (epoch {best_epoch})\n")
        f.write(f"TEST acc: {test_acc:.6f}\n")

        f.write("\n--- Timing ---\n")
        f.write(f"Embedding stage total (s): {t_emb:.2f}\n")
        f.write(f"Train time total (s): {t_train:.2f}\n")
        f.write(f"Test eval time (s): {t_test:.2f}\n")

        f.write("\n--- RAM ---\n")
        f.write(f"[Emb]  proc RSS(GB): {ram_proc_emb:.2f} | sys: {used_sys_emb:.1f}/{total_sys_emb:.1f} ({pct_emb}%)\n")
        f.write(f"[Train] proc RSS(GB): {ram_proc_train:.2f} | sys: {used_sys_train:.1f}/{total_sys_train:.1f} ({pct_train}%)\n")
        f.write(f"[Test]  proc RSS(GB): {ram_proc_test:.2f} | sys: {used_sys_test:.1f}/{total_sys_test:.1f} ({pct_test}%)\n")

        f.write("\n--- VRAM (CUDA) ---\n")
        f.write(f"[Emb]  peak alloc(GB): {emb_peak:.2f} | alloc: {emb_alloc:.2f} | reserved: {emb_reserved:.2f}\n")
        f.write(f"[Train] peak alloc(GB): {train_peak:.2f} | alloc: {train_alloc:.2f} | reserved: {train_reserved:.2f}\n")
        f.write(f"[Test]  peak alloc(GB): {test_peak:.2f} | alloc: {test_alloc:.2f} | reserved: {test_reserved:.2f}\n")

        f.write(f"\nTotal time (s): {t_total:.2f}\n")

        f.write("\n--- Model size ---\n")
        f.write(f"Trainable parameters (MLP): {trainable_params}\n")

        f.write("\n--- Inference (TEST) ---\n")
        f.write(f"Test embedding time (s): {t_test_emb:.2f}\n")
        f.write(f"Test classifier time (s): {t_test:.2f}\n")
        f.write(f"Inference total (s): {t_infer_total:.2f}\n")
        f.write(f"Avg ms/clip (inference total): {avg_ms_clip_total:.2f}\n")
        f.write(f"Clips/sec (inference total): {clips_per_sec_total:.2f}\n")


    print("\n✅ DONE")
    print("Best VAL acc:", best_val_acc, "epoch:", best_epoch)
    print("TEST acc:", test_acc)
    print("Outputs in:", OUT_DIR)