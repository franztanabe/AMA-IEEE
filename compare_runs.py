import json, csv
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_metrics(run_dir):
    p2 = pd.read_csv(run_dir / "phase2" / "phase2_detections.csv")
    p3 = pd.read_csv(run_dir / "phase3" / "phase3_classifications.csv")
    p4 = pd.read_csv(run_dir / "phase4" / "phase4_reid.csv")
    return {
        "phase2_recall": p2.query("class == 'animal'").shape[0] / max(1, p2.shape[0]),
        "phase3_f1": compute_macro_f1(p3),
        "phase4_rank1": (p4["match_id"].notna()).mean(),
    }

def compute_macro_f1(df):
    classes = df["top1_class"].unique()
    f1s = []
    for c in classes:
        tp = ((df["top1_class"] == c) & (df["true_class"] == c)).sum()
        fp = ((df["top1_class"] == c) & (df["true_class"] != c)).sum()
        fn = ((df["top1_class"] != c) & (df["true_class"] == c)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1s.append(f1)
    return sum(f1s) / max(1, len(f1s))

def main(principal_dir, baseline_dir, out_csv):
    m_p = load_metrics(Path(principal_dir))
    m_b = load_metrics(Path(baseline_dir))
    rows = []
    for k in ("phase2_recall", "phase3_f1", "phase4_rank1"):
        delta = m_p[k] - m_b[k]
        rows.append({
            "metric": k,
            "principal": round(m_p[k], 3),
            "baseline": round(m_b[k], 3),
            "delta": round(delta, 3),
            "magnitude": classify(delta),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def classify(delta):
    a = abs(delta)
    if a < 0.03: return "pequena"
    if a < 0.10: return "moderada"
    return "grande"

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])