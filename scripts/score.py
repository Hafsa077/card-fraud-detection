"""Batch scoring: read a CSV of transactions, write fraud scores and alert flags.

    python scripts/score.py transactions.csv scored.csv

The input needs the same columns as the training data: Time, Amount and V1-V28.
The model and the alert threshold are read from models/, which the notebook writes.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.features import add_features  # noqa: E402

REQUIRED = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score card transactions for fraud.")
    parser.add_argument("input", help="CSV of transactions")
    parser.add_argument("output", help="where to write the scored CSV")
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgb_fraud.json")

    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"input is missing columns: {missing}")

    scores = model.predict_proba(add_features(df)[meta["features"]])[:, 1]
    out = df.copy()
    out["fraud_score"] = scores.round(6)
    out["alert"] = (scores >= meta["threshold"]).astype(int)
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} transactions: {int(out['alert'].sum())} alerts at threshold {meta['threshold']}.")


if __name__ == "__main__":
    main()
