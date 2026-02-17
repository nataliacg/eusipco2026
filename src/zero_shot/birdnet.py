import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# =====================
# 1) CLI Configuration
# =====================
import argparse

parser = argparse.ArgumentParser(
    description="BirdNET-Analyzer evaluation using a manifest CSV."
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

parser.add_argument(
    "--min_conf",
    type=float,
    default=0.0,
    help="Minimum confidence threshold for BirdNET detections.",
)

parser.add_argument(
    "--topk",
    type=int,
    default=5,
    help="Top-K predictions to store per clip.",
)

args = parser.parse_args()

MANIFEST_TEST = args.manifest
RESULTS_DIR = args.results_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONF = args.min_conf
TOPK = args.topk

# =====================
# 2) Helpers
# =====================
def to_spaces(label_underscore: str) -> str:
    """Convert underscore scientific name to space-separated format."""
    return str(label_underscore).strip().replace("_", " ")


def safe_copy(src: Path, dst: Path):
    """Copy file ensuring destination directory exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def pick_clip_prediction(df_clip: pd.DataFrame):
    """
    Aggregate BirdNET detections for a single clip.

    Returns:
        pred_name (str),
        pred_score (float),
        topk_names (list[str]),
        topk_scores (list[float])
    """
    if df_clip.empty:
        return "", 0.0, [], []

    agg = df_clip.groupby("Scientific name", as_index=False)["Confidence"].max()
    agg = agg.sort_values("Confidence", ascending=False)

    pred_name = str(agg.iloc[0]["Scientific name"])
    pred_score = float(agg.iloc[0]["Confidence"])

    top = agg.head(TOPK)
    topk_names = [str(x) for x in top["Scientific name"].tolist()]
    topk_scores = [float(x) for x in top["Confidence"].tolist()]

    return pred_name, pred_score, topk_names, topk_scores

def main():
    t0 = time.perf_counter()

    # =====================
    # 3) Load and validate manifest
    # =====================
    df = pd.read_csv(MANIFEST_TEST)
    req = {"clip_path", "true_scientific_name"}
    if not req.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {req}. Found: {df.columns.tolist()}")

    df["clip_path"] = df["clip_path"].astype(str)
    df["true_scientific_name"] = df["true_scientific_name"].astype(str).str.strip()

    print("Num clips:", len(df))



    # =====================
    # 4) Prepare temporary folder + file mapping
    # =====================
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input_wavs"
        out_dir = tmpdir / "birdnet_out"
        input_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        map_rows = []
        for i, row in df.iterrows():
            src = Path(row["clip_path"])
            if not src.exists():
                raise FileNotFoundError(f"File not found: {src}")

            tmp_name = f"{i:06d}__{src.name}"
            dst = input_dir / tmp_name
            safe_copy(src, dst)

            map_rows.append({
                "idx": i,
                "tmp_name": tmp_name,
                "orig_clip_path": str(src),
                "gt_underscore": row["true_scientific_name"],
                "gt_spaces": to_spaces(row["true_scientific_name"]),
            })

        map_df = pd.DataFrame(map_rows)
        map_df.to_csv(RESULTS_DIR / "tmp_file_mapping.csv", index=False, encoding="utf-8")


        # =====================
        # 5) Run BirdNET Analyzer (single run on folder)
        # =====================
        print("\nRunning BirdNET once on folder...")
        t_bird0 = time.perf_counter()

        cmd = [
            sys.executable, "-m", "birdnet_analyzer.analyze",
            str(input_dir),
            "-o", str(out_dir),
            "--rtype", "csv",
            "--min_conf", str(MIN_CONF),
            "--combine_results",
            "--show_progress",
        ]
        subprocess.run(cmd, check=True)

        t_bird = time.perf_counter() - t_bird0
        print(f"BirdNET analysis time: {t_bird:.2f}s")

        # =====================
        # 6) Load BirdNET outputs
        # =====================
        csvs = list(out_dir.rglob("*.csv"))
        if not csvs:
            raise RuntimeError(f"No CSVs were generated in {out_dir}")

        det_all = []
        for p in csvs:
            try:
                d = pd.read_csv(p)
            except Exception:
                continue

            cols = {c.strip().lower(): c for c in d.columns}
            needed = ["scientific name", "confidence", "file"]
            if not all(k in cols for k in needed):
                continue

            d = d.rename(columns={
                cols["scientific name"]: "Scientific name",
                cols["confidence"]: "Confidence",
                cols["file"]: "File"
            })
            det_all.append(d[["Scientific name", "Confidence", "File"]].copy())

        if not det_all:
            raise RuntimeError("No CSV found with columns (Scientific name, Confidence, File).")

        det = pd.concat(det_all, ignore_index=True)
        det["File"] = det["File"].astype(str).apply(lambda x: Path(x).name)

        # =====================
        # 7) Score clips (top-1 + top-k)
        # =====================
        pred_rows = []
        y_true_labels = []
        y_pred_labels = []
        num_no_detections = 0

        for _, r in tqdm(map_df.iterrows(), total=len(map_df), desc="Scoring clips"):
            tmp_name = r["tmp_name"]
            df_clip = det[det["File"] == tmp_name]

            pred_name, pred_score, topk_names, topk_scores = pick_clip_prediction(df_clip)

            gt_label = r["gt_underscore"]

            if pred_name == "":
                pred_label = "NO_DETECTION"
                num_no_detections += 1
            else:
                pred_label = pred_name.strip().replace(" ", "_")

            y_true_labels.append(gt_label)
            y_pred_labels.append(pred_label)

            pred_rows.append({
                "clip_path": r["orig_clip_path"],
                "true_scientific_name": r["gt_underscore"],
                "pred_scientific_name": pred_name,
                "pred_prob": float(pred_score),
                "topk_pred_scientific": ";".join(topk_names),
                "topk_probs": ";".join([f"{s:.6f}" for s in topk_scores]),
            })

        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(RESULTS_DIR / "predictions_test.csv", index=False, encoding="utf-8")

        # =====================
        # 8) Metrics (closed-set + OUT_OF_SET)
        # =====================
        known_labels = sorted(set(y_true_labels))
        known_set = set(known_labels)

        y_pred_closed = []
        num_out_of_set = 0
        for p in y_pred_labels:
            if p in known_set or p == "NO_DETECTION":
                y_pred_closed.append(p)
            else:
                y_pred_closed.append("OUT_OF_SET")
                num_out_of_set += 1

        labels_eval = known_labels.copy()
        if "NO_DETECTION" in set(y_pred_closed):
            labels_eval.append("NO_DETECTION")
        labels_eval.append("OUT_OF_SET")

        acc = accuracy_score(y_true_labels, y_pred_closed)

        report_dict = classification_report(
            y_true_labels,
            y_pred_closed,
            labels=labels_eval,
            target_names=labels_eval,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report_dict).T.to_csv(
            RESULTS_DIR / "classification_report_CLOSEDSET_WITH_OOS.csv",
            index_label="label",
            encoding="utf-8"
        )

        cm = confusion_matrix(y_true_labels, y_pred_closed, labels=labels_eval)
        pd.DataFrame(cm, index=labels_eval, columns=labels_eval).to_csv(
            RESULTS_DIR / "confusion_matrix_CLOSEDSET_WITH_OOS.csv",
            encoding="utf-8"
        )

        outside_counts = pd.Series(
            [p for p in y_pred_labels if p not in known_set and p != "NO_DETECTION"]
        ).value_counts()

        outside_counts.to_csv(
            RESULTS_DIR / "pred_outside_manifest_counts.csv",
            header=["count"],
            encoding="utf-8"
        )

        # =====================
        # 9) Summary + final prints
        # =====================
        oos_rate = num_out_of_set / len(y_pred_labels) if len(y_pred_labels) else 0.0
        t_total = time.perf_counter() - t0
        clips_per_sec = len(df) / t_bird if t_bird > 0 else 0.0
        avg_ms_clip = (t_bird / len(df)) * 1000 if len(df) > 0 else 0.0

        with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
            f.write("=== BirdNET Analyzer (open-set) on TEST manifest ===\n\n")
            f.write(f"Num clips: {len(df)}\n")
            f.write(f"Num labels (manifest): {len(known_labels)}\n")
            f.write(f"Num labels (eval): {len(labels_eval)}\n")
            f.write(f"Accuracy (closed-set + OOS): {acc:.6f}\n")
            f.write(f"Num OUT_OF_SET predictions: {num_out_of_set}\n")
            f.write(f"OUT_OF_SET rate: {oos_rate:.6f}\n")
            f.write(f"MIN_CONF: {MIN_CONF}\n")
            f.write(f"TOPK saved: {TOPK}\n\n")

            f.write(f"Num clips with no detections in CSV: {num_no_detections}\n\n")

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
            f.write(f"BirdNET analysis time (s): {t_bird:.2f}\n")
            f.write(f"Total script time (s): {t_total:.2f}\n")
            f.write(f"Clips/sec (BirdNET stage): {clips_per_sec:.2f}\n")
            f.write(f"Avg ms/clip (BirdNET stage): {avg_ms_clip:.2f}\n")

        print("\n✅ DONE")
        print("Saved in:", RESULTS_DIR)
        print("Accuracy:", acc)
        print(f"BirdNET time: {t_bird:.2f}s | {clips_per_sec:.2f} clips/s | {avg_ms_clip:.2f} ms/clip")
