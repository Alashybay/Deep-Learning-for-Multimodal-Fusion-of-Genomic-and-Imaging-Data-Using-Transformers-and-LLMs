from tensorflow import keras
from tensorflow.keras import layers as L

def build_dna_encoder(num_features: int):
    inp = L.Input((num_features,))
    x = L.Dense(256, activation="relu")(inp)
    x = L.Dropout(0.2)(x)
    x = L.Dense(128, activation="relu")(x)
    return keras.Model(inp, x, name="dna_encoder")