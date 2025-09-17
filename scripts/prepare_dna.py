import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/dna")
OUT_CSV = Path("data/raw/dna/cancer_dna.csv")

def main():
    # найдём первый подходящий CSV (можно явно указать имя)
    csvs = list(RAW_DIR.glob("*.csv"))
    assert csvs, "CSV с геномикой не найден"
    df = pd.read_csv(csvs[0])

    # Унифицируем имена столбцов, если нужно
    cols = {c.lower(): c for c in df.columns}
    # пытаемся найти Patient_ID и Class
    # если нет уникальных ID — сгенерируем
    if "patient_id" not in [c.lower() for c in df.columns]:
        df.insert(0, "Patient_ID", [f"D_{i:06d}" for i in range(len(df))])
    else:
        df.rename(columns={cols["patient_id"]: "Patient_ID"}, inplace=True)

    # если нет Class — создадим фиктивный столбец (для демо)
    if "class" not in [c.lower() for c in df.columns]:
        df["Class"] = "unknown"
    else:
        df.rename(columns={cols["class"]: "Class"}, inplace=True)

    # приведём типы, уберём явный текст в числах, если нужно
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, "Rows:", len(df))

if __name__ == "__main__":
    main()
