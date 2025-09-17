import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PATIENTS = Path("data/processed/patients.csv")
SPLITS_DIR = Path("data/processed/splits")

def main():
    df = pd.read_csv(PATIENTS)
    tr, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    tr, va = train_test_split(tr, test_size=0.1765, stratify=tr["label"], random_state=42)  # ~15% вал
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(SPLITS_DIR/"train.csv", index=False)
    va.to_csv(SPLITS_DIR/"val.csv", index=False)
    te.to_csv(SPLITS_DIR/"test.csv", index=False)
    print("Saved splits to:", SPLITS_DIR)

if __name__ == "__main__":
    main()
