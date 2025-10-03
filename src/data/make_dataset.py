import os, json, math, argparse
from typing import Tuple
import numpy as np

def _prepare_images(n: int, img_h: int, img_w: int, rng: np.random.Generator):
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    cy, cx = img_h/2.0, img_w/2.0
    r = min(img_h, img_w)/4.0
    circle = (((yy-cy)**2 + (xx-cx)**2) <= (r**2)).astype(np.float32)
    imgs = np.zeros((n, img_h, img_w), np.float32)
    signal = np.zeros(n, np.float32)
    for i in range(n):
        base = rng.normal(0, 0.35, size=(img_h, img_w)).astype(np.float32)
        ang = rng.uniform(0, 2*math.pi)
        gx = np.cos(ang)*(xx-cx)/img_w
        gy = np.sin(ang)*(yy-cy)/img_h
        k = rng.uniform(0.5, 1.3)
        lesion = rng.uniform(0.25, 0.95)
        img = base + 0.5*k*(gx+gy) + lesion*circle
        img = (img - img.mean())/(img.std()+1e-6)
        imgs[i] = img
        signal[i] = 0.7*lesion + 0.3*k + rng.normal(0,0.05)
    imgs = (imgs - imgs.min())/(imgs.max()-imgs.min()+1e-6)
    return imgs, signal

def _prepare_genes(n: int, n_genes: int, rng: np.random.Generator):
    # three latent modules for structure
    genes = rng.normal(0, 1, size=(n, n_genes)).astype(np.float32)
    block = n_genes // 5 if n_genes >= 100 else max(10, n_genes//5)
    A = rng.normal(0, 1, size=(n, 1)).astype(np.float32)
    B = rng.normal(0, 1, size=(n, 1)).astype(np.float32)
    C = rng.normal(0, 1, size=(n, 1)).astype(np.float32)
    genes[:, 0:block]              += 1.4*A
    genes[:, block:2*block]        += 1.1*B
    genes[:, 2*block:3*block]      += 0.9*C
    genes += rng.normal(0, 0.25, size=(n, 1)).astype(np.float32)  # mild global correlation
    sig = (1.1*A.squeeze()+0.85*B.squeeze()+0.6*C.squeeze()
           + 0.25*(A.squeeze()*B.squeeze())
           + rng.normal(0,0.1,size=n).astype(np.float32))
    return genes, sig

def _apply_batch_effects(imgs, genes, n_batches: int, rng: np.random.Generator):
    if n_batches <= 0:
        return imgs, genes
    n = imgs.shape[0]
    batch_ids = rng.integers(0, n_batches, size=n)
    # image bias per batch (low-freq offset), gene offset per batch
    img_bias = rng.normal(0, 0.1, size=(n_batches, 1, 1)).astype(np.float32)
    gene_bias = rng.normal(0, 0.2, size=(n_batches, 1)).astype(np.float32)
    imgs = imgs + img_bias[batch_ids]
    genes = genes + gene_bias[batch_ids]
    # re-normalize images to [0,1] after offsets
    imgs = (imgs - imgs.min())/(imgs.max() - imgs.min() + 1e-6)
    return imgs, genes

def _make_labels(img_s, gen_s, rng: np.random.Generator, pos_rate: float):
    # logistic mix with interaction; pos_rate tunes global threshold for class balance
    z = 0.9*img_s + 1.1*gen_s + 0.8*img_s*gen_s + rng.normal(0,0.2,size=img_s.shape)
    p = 1/(1+np.exp(-z))
    # shift threshold to match desired prevalence
    thr = np.quantile(p, 1.0 - pos_rate)
    y = (p > thr).astype(np.int64)
    return y

def build_dataset(
    n_samples: int = 4000,
    img_h: int = 32,
    img_w: int = 32,
    n_genes: int = 200,
    n_tokens: int = 20,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    pos_rate: float = 0.5,
    n_batches: int = 0,
    seed: int = 17,
    out_npz: str = "data/paired_mm.npz",
    out_meta: str = "data/metadata.json",
):
    assert n_genes % n_tokens == 0, "n_genes must be divisible by n_tokens"
    rng = np.random.default_rng(seed)

    Ximg, Simg = _prepare_images(n_samples, img_h, img_w, rng)
    Xgen, Sgen = _prepare_genes(n_samples, n_genes, rng)
    Ximg, Xgen = _apply_batch_effects(Ximg, Xgen, n_batches, rng)
    y = _make_labels(Simg, Sgen, rng, pos_rate)

    idx = np.arange(n_samples); rng.shuffle(idx)
    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    va, te, tr = idx[:n_val], idx[n_val:n_val+n_test], idx[n_val+n_test:]

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz,
             Ximg_tr=Ximg[tr], Ximg_va=Ximg[va], Ximg_te=Ximg[te] if n_test>0 else np.empty((0,img_h,img_w)),
             Xgen_tr=Xgen[tr], Xgen_va=Xgen[va], Xgen_te=Xgen[te] if n_test>0 else np.empty((0,n_genes)),
             y_tr=y[tr], y_va=y[va], y_te=y[te] if n_test>0 else np.empty((0,), dtype=np.int64))

    meta = {
        "n_samples": n_samples, "img_h": img_h, "img_w": img_w,
        "n_genes": n_genes, "n_tokens": n_tokens,
        "val_ratio": val_ratio, "test_ratio": test_ratio,
        "pos_rate": pos_rate, "n_batches": n_batches,
        "seed": seed, "out_npz": out_npz
    }
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    return out_npz, out_meta

def main():
    ap = argparse.ArgumentParser(description="Build paired imaging+genomics dataset.")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--img-h", type=int, default=32)
    ap.add_argument("--img-w", type=int, default=32)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--n-tokens", type=int, default=20)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--pos-rate", type=float, default=0.5, help="target positive class prevalence (0..1)")
    ap.add_argument("--n-batches", type=int, default=0, help="cohort-like batch effects")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=str, default="data/paired_mm.npz")
    ap.add_argument("--meta", type=str, default="data/metadata.json")
    args = ap.parse_args()
    out_npz, out_meta = build_dataset(
        n_samples=args.n_samples, img_h=args.img_h, img_w=args.img_w,
        n_genes=args.n_genes, n_tokens=args.n_tokens, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, pos_rate=args.pos_rate, n_batches=args.n_batches,
        seed=args.seed, out_npz=args.out, out_meta=args.meta
    )
    print(f"Saved: {out_npz}\nMeta:  {out_meta}")

if __name__ == "__main__":
    main()
