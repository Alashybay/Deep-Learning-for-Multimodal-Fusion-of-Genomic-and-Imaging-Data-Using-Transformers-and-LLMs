import os, yaml
from tensorflow import keras
from tensorflow.keras import callbacks as C
from src.data.dataset_tf import make_dataset
from src.models.fusion import build_early_fusion_model

def main():
    cfg = yaml.safe_load(open("configs/experiment_baseline.yaml"))
    paths = cfg["paths"]
    img_size = cfg["images"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    dropout = cfg["model"]["dropout"]

    # датасеты
    train_ds, meta_train = make_dataset(
        os.path.join(paths["splits_dir"], "train.csv"),
        paths["dna_csv"],
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds, meta_val = make_dataset(
        os.path.join(paths["splits_dir"], "val.csv"),
        paths["dna_csv"],
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    assert meta_train["num_features"] == meta_val["num_features"]
    num_features = meta_train["num_features"]
    num_classes  = meta_train["num_classes"]

    # модель
    model = build_early_fusion_model(
        img_size=img_size,
        num_features=num_features,
        num_classes=num_classes,
        dropout=dropout
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(paths["outputs_dir"], exist_ok=True)
    ckpt = C.ModelCheckpoint(
        os.path.join(paths["outputs_dir"], "best.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True
    )
    es = C.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[ckpt, es])

    # простая оценка на test
    test_ds, _ = make_dataset(
        os.path.join(paths["splits_dir"], "test.csv"),
        paths["dna_csv"],
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    with open(os.path.join(paths["outputs_dir"], "metrics.txt"), "w") as f:
        f.write(f"test_loss={test_loss:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
    print(f"[OK] Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
