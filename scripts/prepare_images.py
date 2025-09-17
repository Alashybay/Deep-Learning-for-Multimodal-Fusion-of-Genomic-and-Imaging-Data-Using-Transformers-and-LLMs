import shutil, os
from pathlib import Path

SRC_ROOT = Path("data/raw/images")
DST_ROOT = Path("data/raw/images_by_class")

# нормализуем имена классов
NAME_MAP = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "pituitary_tumor": "pituitary",
    "no_tumor": "no_tumor",
}

def collect_images():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    # ищем и в Training, и в Testing
    for split in ["Training", "Testing"]:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = NAME_MAP.get(cls_dir.name, cls_dir.name)
            out_dir = DST_ROOT / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in cls_dir.glob("*.*"):
                # уникальное имя файла при копии
                dst = out_dir / f"{split}__{cls_dir.name}__{p.name}"
                if not dst.exists():
                    shutil.copy2(p, dst)

if __name__ == "__main__":
    collect_images()
    print("Done. Images normalized into:", DST_ROOT)
