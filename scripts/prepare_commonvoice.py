"""
Best-effort helper to build a CSV from Common Voice English with accent metadata.
You may need to tweak 'ACCENT_MAP' depending on the dataset version/tag formats.

Usage:
  python scripts/prepare_commonvoice.py --cv-root /path/to/cv_corpus/en/ --out data/processed/commonvoice_raw.csv
"""

import argparse, os, pandas as pd

ACCENT_MAP = {
    "northern english": "Northern English",
    "northern": "Northern English",
    "southern english": "Southern English",
    "southern": "Southern English",
    "midlands": "Midlands English",
    "scottish": "Scottish English",
    "welsh": "Welsh English",
    "irish": "Irish English",
}

def normalise_accent(a: str) -> str | None:
    if not isinstance(a, str) or not a.strip():
        return None
    key = a.strip().lower()
    return ACCENT_MAP.get(key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True, help="Path to Common Voice English folder containing 'validated.tsv' and 'clips/'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tsv_path = os.path.join(args.cv_root, "validated.tsv")
    clips_dir = os.path.join(args.cv_root, "clips")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Could not find {tsv_path}")
    if not os.path.exists(clips_dir):
        raise FileNotFoundError(f"Could not find {clips_dir}")

    df = pd.read_csv(tsv_path, sep="\t")
    keep_cols = [c for c in ["path", "client_id", "accent"] if c in df.columns]
    df = df[keep_cols].rename(columns={"client_id": "speaker_id"})
    df["label"] = df["accent"].map(normalise_accent)
    df = df[~df["label"].isna()].copy()
    df["path"] = df["path"].apply(lambda p: os.path.join(clips_dir, p))
    df = df[["path", "label", "speaker_id"]]
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()

