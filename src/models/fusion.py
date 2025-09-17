from tensorflow import keras
from tensorflow.keras import layers as L
from .encoders_img import build_img_encoder
from .encoders_dna import build_dna_encoder

def build_early_fusion_model(img_size, num_features, num_classes, dropout=0.2):
    img_enc = build_img_encoder(img_size)
    dna_enc = build_dna_encoder(num_features)

    inp_img = L.Input((img_size, img_size, 3))
    inp_dna = L.Input((num_features,))
    z_img = img_enc(inp_img)
    z_dna = dna_enc(inp_dna)

    z = L.Concatenate()([z_img, z_dna])
    z = L.Dropout(dropout)(z)
    out = L.Dense(num_classes, activation="softmax")(z)
    return keras.Model([inp_img, inp_dna], out, name="early_fusion")
