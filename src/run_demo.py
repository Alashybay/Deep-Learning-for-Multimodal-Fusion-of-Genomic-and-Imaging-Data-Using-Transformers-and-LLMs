"""
Multimodal Fusion: Genomics + Imaging with CNN/Transformer (PyTorch)
- Builds/loads a paired dataset (via src.data.make_dataset)
- Trains 4 models: image-only, genomics-only, early fusion, intermediate fusion
- Per-epoch validation with AUC/ACC, StepLR, and Early Stopping
- Saves charts & metrics to artifacts/
"""

import os, json, math, random, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

from src.utils import log, accuracy, fmt_losses
from src.data.make_dataset import build_dataset

# ---------------- Config via CLI ----------------
def get_args():
    ap = argparse.ArgumentParser(description="Train multimodal fusion models.")
    ap.add_argument("--rebuild-data", action="store_true", help="(re)build dataset before training")
    ap.add_argument("--n-samples", type=int, default=4000, help="dataset size if rebuilding")
    ap.add_argument("--img-h", type=int, default=32)
    ap.add_argument("--img-w", type=int, default=32)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--n-tokens", type=int, default=20)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--pos-rate", type=float, default=0.5)
    ap.add_argument("--n-batches", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--data-path", type=str, default="data/paired_mm.npz")
    ap.add_argument("--meta-path", type=str, default="data/metadata.json")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    return ap.parse_args()

# ---------------- Repro ----------------
def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return rng

# ---------------- Dataset ----------------
class DS(Dataset):
    def __init__(self, Ximg, Xgen, y, device):
        self.Ximg = torch.tensor(Ximg, dtype=torch.float32).unsqueeze(1)
        self.Xgen = torch.tensor(Xgen, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.device = device
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        # Move to device inside the collate loop (keep tensors CPU until batch)
        return self.Ximg[i], self.Xgen[i], self.y[i]

# ---------------- Models ----------------
class CNN(nn.Module):
    def __init__(self, out_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(32, out_dim)
    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.proj(h)

class GenT(nn.Module):
    def __init__(self, d=48, heads=4, layers=1, n_genes=200, n_tokens=20):
        super().__init__()
        assert n_genes % n_tokens == 0, "n_genes must be divisible by n_tokens"
        self.T = n_tokens
        self.Dtok = n_genes // n_tokens
        self.proj = nn.Linear(self.Dtok, d)
        self.pos  = nn.Parameter(torch.randn(1, self.T, d) * 0.01)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=96, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
    def forward(self, x):
        B = x.size(0)
        tokens = x.view(B, self.T, self.Dtok)
        h = self.proj(tokens) + self.pos
        h = self.enc(h)
        return h.mean(1)

class Head(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2)
        )
    def forward(self, z): return self.net(z)

class EarlyFusion(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.pool = nn.AvgPool2d(4)  # 32 -> 8
        self.mlp  = Head(8*8 + n_genes, 128)
    def forward(self, img, gen):
        x = self.pool(img).view(img.size(0), -1)
        z = torch.cat([x, gen], 1)
        return self.mlp(z)

class InterFusion(nn.Module):
    def __init__(self, n_genes, n_tokens):
        super().__init__()
        self.cnn  = CNN(48)
        self.gen  = GenT(48, heads=4, layers=1, n_genes=n_genes, n_tokens=n_tokens)
        self.head = Head(96, 96)
    def forward(self, img, gen):
        zi = self.cnn(img)
        zg = self.gen(gen)
        z  = torch.cat([zi, zg], 1)
        logits = self.head(z)
        return logits, zi, zg, z

class UniImg(nn.Module):
    def __init__(self): super().__init__(); self.cnn=CNN(48); self.head=Head(48,64)
    def forward(self, img): z=self.cnn(img); return self.head(z), z

class UniGen(nn.Module):
    def __init__(self, n_genes, n_tokens):
        super().__init__()
        self.gen = GenT(48, heads=4, layers=1, n_genes=n_genes, n_tokens=n_tokens)
        self.head= Head(48,64)
    def forward(self, gen): z=self.gen(gen); return self.head(z), z

# ---------------- Train/Eval ----------------
def train_epoch(model, loader, opt, crit, device, mode=None):
    model.train(); tot=0.0
    for img, gen, y in loader:
        img, gen, y = img.to(device), gen.to(device), y.to(device)
        opt.zero_grad()
        if mode == "early":
            logits = model(img, gen)
        elif mode == "img":
            logits, _ = model(img)
        elif mode == "gen":
            logits, _ = model(gen)
        else:
            logits, *_ = model(img, gen)  # intermediate
        loss = crit(logits, y); loss.backward(); opt.step()
        tot += loss.item() * y.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_probs(model, loader, device, mode=None):
    model.eval(); pr=[]; yall=[]; zf=[]
    for img, gen, y in loader:
        img, gen = img.to(device), gen.to(device)
        z = None
        if mode == "early":
            logits = model(img, gen)
            p = torch.softmax(logits, 1)[:,1]
        elif mode == "img":
            logits, z = model(img)
            p = torch.softmax(logits, 1)[:,1]
        elif mode == "gen":
            logits, z = model(gen)
            p = torch.softmax(logits, 1)[:,1]
        else:
            logits, zi, zg, z = model(img, gen)
            p = torch.softmax(logits, 1)[:,1]
        pr.append(p.detach().cpu().numpy())
        yall.append(y.numpy())
        if z is not None:
            zf.append(z.detach().cpu().numpy())
    p = np.concatenate(pr); yv = np.concatenate(yall)
    Z = np.concatenate(zf) if zf else None
    return p, yv, Z

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    log(f"Device: {device}")

    # Build or load dataset
    if args.rebuild_data:
        log("Rebuilding dataset ...")
        build_dataset(
            n_samples=args.n_samples, img_h=args.img_h, img_w=args.img_w,
            n_genes=args.n_genes, n_tokens=args.n_tokens,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            pos_rate=args.pos_rate, n_batches=args.n_batches,
            seed=args.seed, out_npz=args.data_path, out_meta=args.meta_path
        )
    log(f"Loading dataset from {args.data_path}")
    D = np.load(args.data_path, allow_pickle=False)
    Ximg_tr, Ximg_va = D["Ximg_tr"], D["Ximg_va"]
    Xgen_tr, Xgen_va = D["Xgen_tr"], D["Xgen_va"]
    y_tr, y_va = D["y_tr"], D["y_va"]
    log(f"Train={len(y_tr)}, Val={len(y_va)}")

    # Save sample images panel
    os.makedirs("artifacts", exist_ok=True)
    plt.figure()
    k = min(6, len(Ximg_tr))
    for i in range(k):
        plt.subplot(2,3,i+1); plt.imshow(Ximg_tr[i], cmap="gray"); plt.axis("off")
    plt.suptitle("Sample images"); plt.tight_layout()
    plt.savefig("artifacts/sample_images.png"); plt.close()

    # Loaders
    train = DS(Ximg_tr, Xgen_tr, y_tr, device)
    val   = DS(Ximg_va, Xgen_va, y_va, device)
    tr_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(val,   batch_size=max(128, args.batch_size), shuffle=False)
    log(f"Batches: train={len(tr_loader)}, val={len(va_loader)}")

    # Models
    log("Building models ...")
    early = EarlyFusion(args.n_genes).to(device)
    inter = InterFusion(args.n_genes, args.n_tokens).to(device)
    uimg  = UniImg().to(device)
    ugen  = UniGen(args.n_genes, args.n_tokens).to(device)

    crit = nn.CrossEntropyLoss()
    opt_early = torch.optim.Adam(early.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_inter = torch.optim.Adam(inter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_uimg  = torch.optim.Adam(uimg.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
    opt_ugen  = torch.optim.Adam(ugen.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

    sch_early = torch.optim.lr_scheduler.StepLR(opt_early, step_size=10, gamma=0.5)
    sch_inter = torch.optim.lr_scheduler.StepLR(opt_inter, step_size=10, gamma=0.5)
    sch_uimg  = torch.optim.lr_scheduler.StepLR(opt_uimg,  step_size=10, gamma=0.5)
    sch_ugen  = torch.optim.lr_scheduler.StepLR(opt_ugen,  step_size=10, gamma=0.5)
    log("Models & optimizers ready.")

    # Training loop with per-epoch validation & early stopping (on intermediate AUC)
    best_auc, bad_epochs = -1.0, 0
    hist = {"inter":[], "early":[], "img":[], "gen":[]}
    val_hist = {"inter_auc":[], "early_auc":[], "img_auc":[], "gen_auc":[], "late_auc":[], "inter_acc":[]}
    for ep in range(1, args.epochs+1):
        h_inter = train_epoch(inter, tr_loader, opt_inter, crit, device, None)
        h_early = train_epoch(early, tr_loader, opt_early, crit, device, "early")
        h_img   = train_epoch(uimg,  tr_loader, opt_uimg,  crit, device, "img")
        h_gen   = train_epoch(ugen,  tr_loader, opt_ugen,  crit, device, "gen")
        hist["inter"].append(h_inter); hist["early"].append(h_early)
        hist["img"].append(h_img);     hist["gen"].append(h_gen)

        # val eval
        p_early, yv, _   = eval_probs(early, va_loader, device, "early")
        p_inter, _,  zf  = eval_probs(inter, va_loader, device, None)
        p_img,   _,  _   = eval_probs(uimg,  va_loader, device, "img")
        p_gen,   _,  _   = eval_probs(ugen,  va_loader, device, "gen")
        p_late = 0.5*p_img + 0.5*p_gen

        aucs = {
            "inter": float(roc_auc_score(yv, p_inter)),
            "early": float(roc_auc_score(yv, p_early)),
            "img":   float(roc_auc_score(yv, p_img)),
            "gen":   float(roc_auc_score(yv, p_gen)),
            "late":  float(roc_auc_score(yv, p_late))
        }
        acc_inter = accuracy(yv, p_inter)
        val_hist["inter_auc"].append(aucs["inter"])
        val_hist["early_auc"].append(aucs["early"])
        val_hist["img_auc"].append(aucs["img"])
        val_hist["gen_auc"].append(aucs["gen"])
        val_hist["late_auc"].append(aucs["late"])
        val_hist["inter_acc"].append(acc_inter)

        log(fmt_losses(ep, args.epochs, {"inter":h_inter,"early":h_early,"img":h_img,"gen":h_gen}))
        log(f"Val AUCs: inter={aucs['inter']:.3f} | early={aucs['early']:.3f} | "
            f"img={aucs['img']:.3f} | gen={aucs['gen']:.3f} | late={aucs['late']:.3f} | "
            f"inter-ACC={acc_inter:.3f}")

        # LR step
        sch_inter.step(); sch_early.step(); sch_uimg.step(); sch_ugen.step()

        # Early stopping on intermediate AUC
        if aucs["inter"] > best_auc + 1e-4:
            best_auc = aucs["inter"]; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"Early stopping triggered at epoch {ep} (best inter AUC={best_auc:.3f}).")
                break

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    # Loss curves
    plt.figure()
    for k in ["inter","early","img","gen"]:
        plt.plot(hist[k], label=k)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training losses"); plt.legend()
    plt.tight_layout(); plt.savefig("artifacts/loss_curves.png"); plt.close()

    # ROC (final eval already computed in last epoch loop)
    fpr_tprs = {}
    for name, p in [("Intermediate", p_inter), ("Early", p_early),
                    ("Image-only", p_img), ("Genomics-only", p_gen),
                    ("Late", p_late)]:
        fpr, tpr, _ = roc_curve(yv, p); fpr_tprs[name] = (fpr, tpr)
    plt.figure()
    for name, (fpr, tpr) in fpr_tprs.items():
        plt.plot(fpr, tpr, label=name)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (validation)"); plt.legend()
    plt.tight_layout(); plt.savefig("artifacts/roc_validation.png"); plt.close()

    # Embedding PCA (intermediate)
    if zf is not None:
        em = PCA(n_components=2).fit_transform(zf)
        plt.figure()
        m0 = (yv==0); m1=(yv==1)
        plt.scatter(em[m0,0], em[m0,1], marker="o", alpha=0.7, label="Class 0")
        plt.scatter(em[m1,0], em[m1,1], marker="^", alpha=0.7, label="Class 1")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Fused embedding space (val)")
        plt.legend(); plt.tight_layout(); plt.savefig("artifacts/embedding_pca.png"); plt.close()

    # Save metrics & predictions
    metrics = {
        "best_intermediate_auc": best_auc,
        "history": val_hist
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        "prob_early": p_early,
        "prob_intermediate": p_inter,
        "prob_image": p_img,
        "prob_genomics": p_gen,
        "prob_late": p_late,
        "y_true": yv
    }).to_csv("artifacts/val_predictions.csv", index=False)

    log("Done. See artifacts/ and data/.")

if __name__ == "__main__":
    main()
