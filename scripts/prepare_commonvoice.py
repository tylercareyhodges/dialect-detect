# scripts/prepare_commonvoice.py
"""
Prepare a CSV (path,label,speaker_id) from Common Voice English validated.tsv.

- Works with 'accents' (or 'accent') and uses 'variant' as an extra hint.
- Normalises free-text to six UK labels:
  Northern English, Southern English, Midlands English, Scottish English, Welsh English, Irish English.
- Handles comma/semicolon separated values found in CV Delta 22.0.

Usage:
  python scripts/prepare_commonvoice.py \
    --cv-root /path/to/cv-corpus-22.0-delta-2025-06-20/en \
    --out data/processed/commonvoice_raw.csv
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Iterable, Optional, Tuple

import pandas as pd


UK_LABELS = {
    "Northern English",
    "Southern English",
    "Midlands English",
    "Scottish English",
    "Welsh English",
    "Irish English",
}

# Broad UK keyword rules → label (matched case-insensitively)
# Add to these if you see unmapped UK examples printed at the end.
_KEYWORD_RULES: Tuple[Tuple[str, str], ...] = (
    # --- Scotland ---
    (r"\bscotland|scottish|glasgow|glaswegian|edinburgh|doric|highlands|aberdeen|dundee|fife\b", "Scottish English"),
    # --- Wales ---
    (r"\bwales|welsh|cardiff|swansea|carmarthenshire|gwynedd|north wales|south wales\b", "Welsh English"),
    # --- Ireland (NI + ROI grouped) ---
    (r"\bireland|irish|dublin|belfast|northern ireland|derry|galway|cork|limerick|kildare\b", "Irish English"),
    # --- Midlands ---
    (r"\bmidlands|east midlands|west midlands|birmingham|brummie|coventry|wolverhampton|nottingham|derby|leicester\b",
     "Midlands English"),
    # --- Northern England (regions + cities) ---
    (r"\bnorthern english|northern\b", "Northern English"),
    (r"\bnorth ?west(?: england)?|cumbria|lancashire|manchester|salford|liverpool|merseyside|cheshire\b",
     "Northern English"),
    (r"\bnorth ?east(?: england)?|tyne|tees|wear|newcastle|sunderland|middlesbrough\b", "Northern English"),
    (r"\byorkshire(?: and the humber)?|york|leeds|sheffield|bradford|hull|west yorkshire|south yorkshire|north yorkshire\b",
     "Northern English"),
    (r"\bmancunian|scouse\b", "Northern English"),
    # --- Southern England (regions + cities) ---
    (r"\bsouthern english|southern\b", "Southern English"),
    (r"\buk southern english\b", "Southern English"),
    (r"\bsouth ?west(?: england)?|cornwall|cornish|devon|somerset|bristol|bath|west country\b", "Southern English"),
    (r"\bsouth ?east(?: england)?|kent|surrey|sussex|hampshire|berkshire|oxfordshire|buckinghamshire\b",
     "Southern English"),
    (r"\blondon|greater london|estuary|received pronunciation|rp|home counties|cambridge|oxford\b",
     "Southern English"),
    # --- Broad fallbacks (be conservative) ---
    (r"\bengland english\b", "Southern English"),
    (r"\bbritish english\b", "Southern English"),
)

# Precompile once
KEYWORD_RULES = tuple((re.compile(pat, re.I), lab) for pat, lab in _KEYWORD_RULES)


def _split_tokens(text: str) -> Iterable[str]:
    """Split a cell into tokens on commas/semicolons/vertical bars/slashes."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Common Voice Delta 22.0 uses commas; we handle a few separators anyway.
    parts = re.split(r"[;,|/]", text)
    return [p.strip() for p in parts if p.strip()]


def _map_text_to_label(text: str) -> Optional[str]:
    """Return one of UK_LABELS if any keyword matches; else None."""
    if not isinstance(text, str) or not text.strip():
        return None
    for rx, label in KEYWORD_RULES:
        if rx.search(text):
            return label
    return None


def _best_label_from_row(accents: str, variant: str) -> Optional[str]:
    """
    Try mapping:
      1) each token in 'accents'
      2) the whole 'accents' string
      3) the 'variant' field
    Returns the first UK label found, else None.
    """
    # 1) token-wise search in accents
    for tok in _split_tokens(accents):
        lab = _map_text_to_label(tok)
        if lab in UK_LABELS:
            return lab
    # 2) whole accents string
    lab = _map_text_to_label(accents)
    if lab in UK_LABELS:
        return lab
    # 3) variant fallback
    lab = _map_text_to_label(variant)
    if lab in UK_LABELS:
        return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True, help="Folder containing validated.tsv and clips/")
    ap.add_argument("--out", required=True, help="Output CSV path (path,label,speaker_id)")
    ap.add_argument("--show-unmapped", action="store_true", help="Print a few unmapped examples for debugging")
    args = ap.parse_args()

    tsv_path = os.path.join(args.cv_root, "validated.tsv")
    clips_dir = os.path.join(args.cv_root, "clips")
    if not os.path.exists(tsv_path):
        sys.exit(f"[ERROR] Could not find {tsv_path}")
    if not os.path.exists(clips_dir):
        sys.exit(f"[ERROR] Could not find {clips_dir}")

    df = pd.read_csv(tsv_path, sep="\t")

    # Choose accent/variant columns
    accent_col = "accents" if "accents" in df.columns else ("accent" if "accent" in df.columns else None)
    if accent_col is None:
        print("[ERROR] No 'accents' or 'accent' column in this release.")
        print("Columns present:", list(df.columns))
        sys.exit(1)
    variant_col = "variant" if "variant" in df.columns else None

    # Map to UK labels
    labels = []
    for _, row in df.iterrows():
        accents = str(row.get(accent_col, "") or "")
        variant = str(row.get(variant_col, "") or "")
        lab = _best_label_from_row(accents, variant)
        labels.append(lab)
    df["label"] = labels

    # Keep only mapped rows
    before = len(df)
    df_mapped = df[~df["label"].isna()].copy()
    after = len(df_mapped)
    kept_pct = 100.0 * after / max(1, before)
    print(f"Mapped UK-labelled rows: {after}/{before} ({kept_pct:.1f}%)")

    # Ensure essential columns
    if "path" not in df_mapped.columns:
        sys.exit("[ERROR] Could not find 'path' column with audio filenames.")
    spk_col = None
    for cand in ("client_id", "speaker_id", "user_id"):
        if cand in df_mapped.columns:
            spk_col = cand
            break
    if spk_col is None:
        sys.exit("[ERROR] Could not find a speaker id column (client_id/speaker_id/user_id).")

    # Absolute audio paths + minimal columns
    def to_abs(p: str) -> str:
        p = str(p)
        return os.path.join(clips_dir, p) if not os.path.isabs(p) else p

    out_df = df_mapped.rename(columns={spk_col: "speaker_id"}).copy()
    out_df["path"] = out_df["path"].apply(to_abs)
    out_df = out_df[["path", "label", "speaker_id"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("Per-class counts:\n", out_df["label"].value_counts())
    print(f"\nWrote {len(out_df)} rows → {args.out}")

    # Optional: show a few unmapped examples to help extend rules
    if args.show_unmapped:
        print("\nExamples that did NOT map (accent | variant):")
        shown = 0
        for _, row in df.iterrows():
            if row.get("label") is not None:
                continue
            a = str(row.get(accent_col, "") or "")
            v = str(row.get(variant_col, "") or "")
            # Print only rows that look UK-ish to keep output relevant
            if re.search(r"\b(england|scotland|wales|ireland|uk|british|london|midlands|yorkshire)\b", (a + " " + v), re.I):
                print(f"  - {a!r} | {v!r}")
                shown += 1
                if shown >= 12:
                    break


if __name__ == "__main__":
    main()
