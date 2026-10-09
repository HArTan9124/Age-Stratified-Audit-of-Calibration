"""Track 2 — retrain InceptionTime with Gitau (2025) super-diag config, to push macro-AUC
past the current 0.923 single / 0.927 blended.

  depth 9 · kernels (40,20,10) · class-weighted BCE · concat-pool head · ReduceLROnPlateau
  · early stopping on VAL macro-AUC · light Gaussian-noise augmentation.

Writes outputs/08_inception_v2_best.pth (best-VAL checkpoint) + a JSON training log.
CPU run — expect several hours; it early-stops when VAL macro-AUC plateaus.
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DP   = ROOT / "data" / "processed"
OUT  = ROOT / "outputs"
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(4)

DEPTH, NF, KS = 9, 32, (41, 21, 11)  # odd => exact same-length padding (Gitau used ~40/20/10)
MAX_EPOCHS, BATCH, LR = 18, 64, 1e-3
ES_PATIENCE, PLATEAU_PATIENCE = 6, 3
NOISE_SIGMA, AUG_PROB = 0.01, 0.5
CKPT_OUT = OUT / "08_inception_v2_best.pth"
LOG_OUT  = OUT / "08_inception_v2_trainlog.json"

# ---------------- model: InceptionTime + concat-pool head ----------------
class InceptionBlock1D(nn.Module):
    def __init__(self, ic, nf=32, ks=(41, 21, 11)):
        super().__init__()
        self.bottleneck = nn.Conv1d(ic, nf, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks])
        self.pool_conv = nn.Conv1d(ic, nf, 1, bias=False)
        self.pool = nn.MaxPool1d(3, 1, 1)
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1)); self.act = nn.ReLU()
    def forward(self, x):
        b = self.bottleneck(x)
        return self.act(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool_conv(self.pool(x))], 1)))

class InceptionTimeV2(nn.Module):
    def __init__(self, nc=5, ic=12, nf=32, depth=9, ks=(41, 21, 11), drop=0.3):
        super().__init__()
        ch = [ic] + [nf * 4] * depth
        self.blocks = nn.Sequential(*[InceptionBlock1D(ch[i], nf, ks) for i in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1); self.gmp = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(nf * 4 * 2, nc)          # concat-pool -> 2x width
    def forward(self, x):
        x = self.blocks(x)
        x = torch.cat([self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)], 1)
        return self.fc(self.drop(x))

# ---------------- data ----------------
def load(stem): return np.transpose(np.load(DP / f"{stem}.npy"), (0, 2, 1)).astype(np.float32)
Xtr, ytr = load("X_train_clean_100"), np.load(DP / "y_train_diag_100.npy").astype(np.float32)
Xva, yva = load("X_val_clean_100"),   np.load(DP / "y_valid_diag_100.npy").astype(np.float32)
print(f"[DATA] train {Xtr.shape}  val {Xva.shape}  device {DEV}", flush=True)

pos = ytr.sum(0); neg = len(ytr) - pos
pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32, device=DEV)
print(f"[LOSS] pos_weight = {pos_weight.cpu().numpy().round(2)}", flush=True)

tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                   batch_size=BATCH, shuffle=True, drop_last=True)
va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                   batch_size=128, shuffle=False)

model = InceptionTimeV2(5, 12, NF, DEPTH, KS).to(DEV)
n_par = sum(p.numel() for p in model.parameters())
print(f"[MODEL] InceptionTimeV2 depth={DEPTH} ks={KS} params={n_par:,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                   patience=PLATEAU_PATIENCE)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

@torch.no_grad()
def val_macro_auc():
    model.eval(); ps, ys = [], []
    for xb, yb in va_dl:
        ps.append(torch.sigmoid(model(xb.to(DEV))).cpu().numpy()); ys.append(yb.numpy())
    P, Y = np.vstack(ps), np.vstack(ys)
    return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(5)]))

best, best_ep, wait, log = 0.0, -1, 0, []
t0 = time.time()
for ep in range(1, MAX_EPOCHS + 1):
    model.train(); run = 0.0
    for xb, yb in tr_dl:
        xb = xb.to(DEV)
        if np.random.rand() < AUG_PROB:
            xb = xb + torch.randn_like(xb) * NOISE_SIGMA
        yb = yb.to(DEV)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward(); opt.step()
        run += loss.item()
    auc = val_macro_auc(); sched.step(auc)
    lr_now = opt.param_groups[0]["lr"]
    log.append({"epoch": ep, "train_loss": run / len(tr_dl), "val_macro_auc": auc,
                "lr": lr_now, "elapsed_s": round(time.time() - t0)})
    print(f"[EP {ep:2d}] loss {run/len(tr_dl):.4f}  val macro-AUC {auc:.4f}  "
          f"lr {lr_now:.1e}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if auc > best + 1e-4:
        best, best_ep, wait = auc, ep, 0
        torch.save(model.state_dict(), CKPT_OUT)
        print(f"        -> new best, saved {CKPT_OUT.name}", flush=True)
    else:
        wait += 1
        if wait >= ES_PATIENCE:
            print(f"[STOP] no VAL improvement for {ES_PATIENCE} epochs.", flush=True)
            break

json.dump({"config": {"depth": DEPTH, "nf": NF, "ks": KS, "batch": BATCH, "lr": LR,
                      "max_epochs": MAX_EPOCHS, "noise_sigma": NOISE_SIGMA},
           "params": n_par, "best_val_macro_auc": best, "best_epoch": best_ep,
           "log": log}, open(LOG_OUT, "w"), indent=2)
print(f"\n[DONE] best VAL macro-AUC {best:.4f} @ epoch {best_ep}  ->  {CKPT_OUT}", flush=True)
