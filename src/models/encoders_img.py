from tensorflow import keras
from tensorflow.keras import layers as L

def build_img_encoder(img_size=224):
    inp = L.Input((img_size, img_size, 3))
    x = L.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(128, activation="relu")(x)
    return keras.Model(inp, x, name="img_encoder")