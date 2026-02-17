import argparse
import time
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================
# 1) CLI Configuration
# =====================
parser = argparse.ArgumentParser(
    description="BirdNET few-shot training and closed-set evaluation."
)

parser.add_argument("--python_exe", type=str, required=True,
                    help="Python executable inside BirdNET-Analyzer environment.")

parser.add_argument("--train_dir", type=Path, required=True)
parser.add_argument("--val_dir", type=Path)
parser.add_argument("--test_dir", type=Path, required=True)

parser.add_argument("--out_root", type=Path, required=True)

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--val_split", type=float, default=0.2)
parser.add_argument("--crop_mode", type=str, default="center")
parser.add_argument("--model_format", type=str, default="tflite")

parser.add_argument("--min_conf", type=float, default=0.0)
parser.add_argument("--top_n", type=int)
parser.add_argument("--show_progress", action="store_true")

args = parser.parse_args()

# =====================
# 2) Utils
# =====================

def build_filename_to_label(test_dir: Path) -> dict:
    mapping = {}
    for wav in test_dir.rglob("*.wav"):
        mapping[wav.name] = wav.parent.name
    return mapping


def infer_gt_from_birdnet_filecol(filecol_value: str, fname_to_label: dict) -> str:
    p = Path(str(filecol_value))

    label = p.parent.name
    if label and label != ".":
        return label

    return fname_to_label.get(p.name, "")


def infer_gt_from_path(file_path: str) -> str:
    folder = Path(file_path).parent.name
    return folder.split("_", 1)[0].strip()


def run(cmd, cwd=None):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=cwd)


def folder_classes(folder: Path):
    return sorted([p.name for p in folder.iterdir() if p.is_dir()])


def read_birdnet_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    colmap = {c.strip().lower(): c for c in df.columns}
    required = ["scientific name", "confidence", "file"]

    for k in required:
        if k not in colmap:
            raise ValueError(
                f"Unexpected CSV format ({csv_path}). Columns: {df.columns.tolist()}"
            )

    sci = colmap["scientific name"]
    conf = colmap["confidence"]
    fcol = colmap["file"]

    df["_sci"] = df[sci].astype(str).str.strip()
    df["_conf"] = pd.to_numeric(df[conf], errors="coerce").fillna(0.0)
    df["_file"] = df[fcol].astype(str)

    return df


def make_summary(
    out_dir: Path,
    acc,
    report_dict,
    n_files: int,
    t_train: float,
    t_inf: float,
    t_post: float,
    t_total: float,
):

    clips_per_sec_inf = n_files / t_inf if t_inf > 0 else 0.0
    avg_ms_inf = (t_inf / n_files) * 1000 if n_files > 0 else 0.0

    clips_per_sec_total = n_files / t_total if t_total > 0 else 0.0
    avg_ms_total = (t_total / n_files) * 1000 if n_files > 0 else 0.0

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Num files: {n_files}\n")
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

        f.write("\n--- Timing breakdown ---\n")
        f.write(f"Train time (s): {t_train:.2f}\n")
        f.write(f"Inference time (analyze) (s): {t_inf:.2f}\n")
        f.write(f"Postprocess time (s): {t_post:.2f}\n")
        f.write(f"Total time (s): {t_total:.2f}\n")

        f.write("\n--- Throughput (inference only) ---\n")
        f.write(f"Clips/sec: {clips_per_sec_inf:.2f}\n")
        f.write(f"Avg ms/clip: {avg_ms_inf:.2f}\n")

        f.write("\n--- Throughput (total pipeline) ---\n")
        f.write(f"Clips/sec: {clips_per_sec_total:.2f}\n")
        f.write(f"Avg ms/clip: {avg_ms_total:.2f}\n")
# =========================
# MAIN
# =========================
def main():

    # -------------------------
    # 1) Read CLI arguments
    # -------------------------
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    TEST_DIR = args.test_dir
    OUT_ROOT = args.out_root
    PYTHON_EXE = args.python_exe

    EPOCHS = args.epochs
    BATCH = args.batch
    LR = args.lr
    VAL_SPLIT = args.val_split
    CROP_MODE = args.crop_mode
    MODEL_FORMAT = args.model_format

    MIN_CONF = args.min_conf
    TOP_N = args.top_n
    SHOW_PROGRESS = args.show_progress

    assert TRAIN_DIR.exists(), f"TRAIN_DIR does not exist: {TRAIN_DIR}"
    assert TEST_DIR.exists(), f"TEST_DIR does not exist: {TEST_DIR}"

    train_labels = folder_classes(TRAIN_DIR)
    print("Train classes:", len(train_labels))
    print(train_labels)

    has_val = VAL_DIR is not None and VAL_DIR.exists() and any(VAL_DIR.iterdir())
    has_test = TEST_DIR.exists() and any(TEST_DIR.iterdir())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_dir = OUT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_out_dir = run_dir / "custom_classifier"
    print("\nOutput run:", run_dir)

    # -------------------------
    # 2) TRAIN
    # -------------------------
    t0 = time.perf_counter()

    train_cmd = [
        PYTHON_EXE, "-m", "birdnet_analyzer.train",
        str(TRAIN_DIR),
        "-o", str(model_out_dir),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH),
        "--learning_rate", str(LR),
        "--crop_mode", CROP_MODE,
        "--model_format", MODEL_FORMAT,
        "--model_save_mode", "append",
    ]

    if has_val:
        train_cmd += ["--test_data", str(VAL_DIR)]
    else:
        train_cmd += ["--val_split", str(VAL_SPLIT)]

    t_train0 = time.perf_counter()
    run(train_cmd)
    t_train = time.perf_counter() - t_train0
    print(f"\nTrain time: {t_train:.2f}s")

    # Locate model
    tflites = list(run_dir.rglob("*.tflite"))
    if not tflites:
        tflites = list(model_out_dir.parent.rglob("*.tflite"))

    if not tflites:
        raise FileNotFoundError(f"No .tflite model found in {run_dir}")

    model_path = tflites[0]
    print("\n✅ Model:", model_path)


    # -------------------------
    # 3) INFERENCE
    # -------------------------
    if not has_test:
        print("\n⚠️ No TEST_DIR found. Training finished.")
        return

    pred_out = run_dir / "test_predictions"
    pred_out.mkdir(parents=True, exist_ok=True)

    analyze_cmd = [
        PYTHON_EXE, "-m", "birdnet_analyzer.analyze",
        str(TEST_DIR),
        "-o", str(pred_out),
        "--classifier", str(model_path),
        "--rtype", "csv",
        "--min_conf", str(MIN_CONF),
        "--combine_results",
    ]

    if SHOW_PROGRESS:
        analyze_cmd += ["--show_progress"]

    if TOP_N is not None:
        analyze_cmd += ["--top_n", str(TOP_N)]

    t_inf0 = time.perf_counter()
    run(analyze_cmd)
    t_inf = time.perf_counter() - t_inf0
    print(f"\nInference time: {t_inf:.2f}s")

    csvs = list(pred_out.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV generated in {pred_out}")

    out_csv = max(csvs, key=lambda p: p.stat().st_size)
    print("\n✅ Predictions CSV:", out_csv)

    # -------------------------
    # 4) Postprocess (closed-set evaluation)
    # -------------------------
    t_post0 = time.perf_counter()

    det = read_birdnet_csv(out_csv)
    det["_sci"] = det["_sci"].astype(str).apply(lambda s: s.split("_", 1)[0].strip())

    fname_to_label = build_filename_to_label(TEST_DIR)
    det["_gt"] = det["_file"].apply(lambda x: infer_gt_from_birdnet_filecol(x, fname_to_label))
    det["_gt"] = det["_gt"].astype(str).apply(lambda s: s.split("_", 1)[0].strip())

    det_sorted = det.sort_values(["_file", "_conf"], ascending=[True, False])
    top1 = det_sorted.groupby("_file", as_index=False).head(1).copy()

    top1.rename(columns={
        "_sci": "pred_label",
        "_conf": "pred_conf",
        "_gt": "true_label"
    }, inplace=True)

    classes = sorted([lbl.split("_", 1)[0].strip() for lbl in train_labels])
    name_to_idx = {c: i for i, c in enumerate(classes)}
    OTHER_IDX = len(classes)

    y_true = [name_to_idx[t] for t in top1["true_label"].tolist()]

    y_pred = []
    out_of_set = 0
    for p in top1["pred_label"].tolist():
        if p in name_to_idx:
            y_pred.append(name_to_idx[p])
        else:
            out_of_set += 1
            y_pred.append(OTHER_IDX)

    extended_classes = classes + ["OTHER"]
    labels_ext = list(range(len(extended_classes)))

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_ext,
        target_names=extended_classes,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report_dict).T.to_csv(run_dir / "classification_report.csv")
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels_ext),
        index=extended_classes,
        columns=extended_classes
    ).to_csv(run_dir / "confusion_matrix.csv")

    top1[["true_label", "pred_label", "pred_conf", "_file"]].rename(
        columns={"_file": "clip_path"}
    ).to_csv(run_dir / "predictions_top1.csv", index=False)

    t_post = time.perf_counter() - t_post0
    t_total = time.perf_counter() - t0

    make_summary(
        run_dir,
        acc,
        report_dict,
        n_files=len(top1),
        t_train=t_train,
        t_inf=t_inf,
        t_post=t_post,
        t_total=t_total
    )

    print("\n=== DONE ===")
    print("Accuracy:", acc)
    print("Out-of-set preds:", out_of_set)
    print("Saved to:", run_dir)

if __name__ == "__main__":
    main()