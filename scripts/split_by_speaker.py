"""
Speaker-disjoint split: takes an input CSV (path,label,speaker_id) and writes train/val/test CSVs.
Usage:
  python scripts/split_by_speaker.py --in data/processed/commonvoice_raw.csv --out-dir data/processed --seed 42
"""

import argparse, os, random, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    speakers = list(df["speaker_id"].astype(str).unique())
    random.Random(args.seed).shuffle(speakers)

    n = len(speakers); n_train = int(n * args.train); n_val = int(n * args.val)
    spk_train = set(speakers[:n_train])
    spk_val = set(speakers[n_train:n_train+n_val])
    spk_test = set(speakers[n_train+n_val:])

    tr = df[df["speaker_id"].astype(str).isin(spk_train)]
    va = df[df["speaker_id"].astype(str).isin(spk_val)]
    te = df[df["speaker_id"].astype(str).isin(spk_test)]

    os.makedirs(args.out_dir, exist_ok=True)
    tr.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    va.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    te.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print(f"Speakers: train {len(spk_train)}, val {len(spk_val)}, test {len(spk_test)}")
    print(f"Rows:     train {len(tr)}, val {len(va)}, test {len(te)}")

if __name__ == "__main__":
    main()

