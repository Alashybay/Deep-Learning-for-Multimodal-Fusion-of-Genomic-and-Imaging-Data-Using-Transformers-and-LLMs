import pandas as pd
import random
from pathlib import Path

IMG_ROOT = Path("data/raw/images_by_class")
DNA_CSV = Path("data/raw/dna/cancer_dna.csv")
OUT_PATIENTS = Path("data/processed/patients.csv")

random.seed(42)

def main():
    # читаем ДНК-таблицу
    dna_df = pd.read_csv(DNA_CSV)
    dna_idx = list(dna_df.index)

    rows = []
    # пройдём по всем изображениям всех классов
    for cls_dir in IMG_ROOT.iterdir():
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name  # glioma / meningioma / pituitary / no_tumor
        for img_path in cls_dir.glob("*.*"):
            j = random.choice(dna_idx)
            dna_row = dna_df.iloc[j]
            # создаём синтетического "пациента"
            pid = f"SYN_{img_path.stem}"
            rows.append({
                "patient_id": pid,
                "label": label,                 # целевая метка -> из MRI
                "image_path": str(img_path),    # путь к изображению
                "dna_row_index": int(j)         # индекс в общей ДНК-таблице
            })

    out = pd.DataFrame(rows)
    OUT_PATIENTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATIENTS, index=False)
    print("Saved:", OUT_PATIENTS, "Rows:", len(out))

if __name__ == "__main__":
    main()
