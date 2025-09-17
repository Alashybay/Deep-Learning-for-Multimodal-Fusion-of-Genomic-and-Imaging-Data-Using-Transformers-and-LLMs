import tensorflow as tf
import pandas as pd
import numpy as np

def _load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def detect_dna_feature_columns(dna_df, exclude=("Patient_ID", "Class")):
    # берём числовые колонки, исключая ID/класс
    num_cols = dna_df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

def make_dataset(split_csv_path, dna_csv_path, img_size=224, batch_size=16, shuffle=True):
    # читаем сплит (train/val/test)
    split_df = pd.read_csv(split_csv_path)
    # читаем полный ДНК CSV
    dna_df = pd.read_csv(dna_csv_path)

    # авто-выбор фич ДНК
    feature_cols = detect_dna_feature_columns(dna_df)

    # формируем X_dna в том же порядке, что строки в split_df, используя dna_row_index
    assert "dna_row_index" in split_df.columns, "В split CSV должен быть столбец dna_row_index"
    dna_rows = split_df["dna_row_index"].astype(int).values
    X_dna = dna_df.loc[dna_rows, feature_cols].to_numpy(dtype="float32")

    # метки
    y = split_df["label"].astype("category").cat.codes.to_numpy()  # 0..C-1

    # пути к изображениям
    img_paths = split_df["image_path"].astype(str).values

    # tf.data
    ds_imgs = tf.data.Dataset.from_tensor_slices(img_paths).map(
        lambda p: _load_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_dna = tf.data.Dataset.from_tensor_slices(X_dna)
    ds_y   = tf.data.Dataset.from_tensor_slices(y)

    ds = tf.data.Dataset.zip(((ds_imgs, ds_dna), ds_y))
    if shuffle:
        ds = ds.shuffle(len(split_df), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # возвращаем датасет и полезные метаданные
    num_classes = int(split_df["label"].astype("category").cat.categories.size)
    meta = {
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "num_classes": num_classes,
    }
    return ds, meta
