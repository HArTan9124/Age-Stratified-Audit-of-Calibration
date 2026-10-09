"""
NB12 — Paired attribution overlays for a high- and a low-confidence elderly case.

The manuscript needs the reader to see, not just be told, what an attribution map
looks like when the model is confident about an 80+ patient and when it is not.
For CD and HYP (the two classes the model discriminates worst) this picks the
most- and least-confident correctly-labelled 80+ test record and draws the
Integrated-Gradients overlay plus the Grad-CAM-1D temporal strip side by side,
using the same attribution code as NB06.

Outputs: outputs/figures/nb12_xai_pairs/12_xai_pair_{CD,HYP}.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'outputs' / 'figures' / 'nb12_xai_pairs'
OUT.mkdir(parents=True, exist_ok=True)

SUPER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SIG_LEN, FS, IG_STEPS = 1000, 100.0, 64
DEVICE = torch.device('cpu')

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, nb_filters=32, kernel_sizes=(9, 19, 39)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, nb_filters, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(nb_filters, nb_filters, k, padding=k // 2, bias=False)
                                    for k in kernel_sizes])
        self.pool_conv = nn.Conv1d(in_channels, nb_filters, 1, bias=False)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()
    def forward(self, x):
        b = self.bottleneck(x)
        return self.act(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool_conv(self.pool(x))], 1)))

class InceptionTime1D(nn.Module):
    def __init__(self, num_classes=5, in_channels=12, nb_filters=32, depth=4):
        super().__init__()
        ch = [in_channels] + [nb_filters * 4] * depth
        self.blocks = nn.Sequential(*[InceptionBlock1D(ch[i], nb_filters) for i in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, num_classes)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        return self.fc(self.drop(self.gap(self.blocks(x)).squeeze(-1)))

X_test = np.transpose(np.load(DATA / 'X_test_clean_100.npy'), (0, 2, 1)).astype(np.float32)
y_test = np.load(DATA / 'y_test_diag_100.npy').astype(np.float32)
age_test = np.load(DATA / 'y_test_age_100.npy').astype(np.float32)

model = InceptionTime1D(5)
model.load_state_dict(torch.load(ROOT / 'outputs' / '05_optimized_InceptionTime1D_best.pth',
                                 map_location=DEVICE, weights_only=True))
model.eval().to(DEVICE)

@torch.no_grad()
def predict(X, bs=128):
    z = np.vstack([model(torch.from_numpy(X[i:i + bs])).numpy() for i in range(0, len(X), bs)])
    return 1.0 / (1.0 + np.exp(-z))

P = predict(X_test)
pe = np.clip(P, 1e-7, 1 - 1e-7)
entropy = (-(pe * np.log(pe) + (1 - pe) * np.log(1 - pe))).mean(1)

def integrated_gradients(x_np, c, steps=IG_STEPS):
    xt = torch.tensor(x_np[None]); bt = torch.zeros_like(xt)
    a = torch.linspace(0, 1, steps).view(-1, 1, 1)
    path = (bt + a * (xt - bt)).requires_grad_(True)
    g = torch.autograd.grad(model(path)[:, c].sum(), path)[0].mean(0)
    return ((xt - bt)[0] * g).detach().numpy()

def grad_cam_1d(x_np, c, block_idx=3):
    acts, grads = {}, {}
    layer = model.blocks[block_idx]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    x = torch.tensor(x_np[None], requires_grad=True)
    model.zero_grad(); model(x)[0, c].backward()
    h1.remove(); h2.remove()
    cam = torch.relu((grads['g'][0].mean(1)[:, None] * acts['a'][0]).sum(0)).detach().numpy()
    return cam / (cam.max() + 1e-9)

elderly = age_test >= 80
picked = {}
t = np.arange(SIG_LEN) / FS
for k, sc in enumerate(SUPER):
    if sc not in ('CD', 'HYP'):
        continue
    pos = np.where(elderly & (y_test[:, k] == 1) & (P[:, k] >= 0.5))[0]
    if len(pos) < 2:
        print(f'[skip] {sc}: only {len(pos)} confident 80+ true positives'); continue
    hi = pos[np.argmin(entropy[pos])]       # model most certain overall
    lo = pos[np.argmax(entropy[pos])]       # model least certain overall
    picked[sc] = dict(high=int(hi), low=int(lo),
                      p_high=float(P[hi, k]), p_low=float(P[lo, k]),
                      entropy_high=float(entropy[hi]), entropy_low=float(entropy[lo]),
                      age_high=float(age_test[hi]), age_low=float(age_test[lo]),
                      labels_high=[SUPER[i] for i in np.where(y_test[hi] == 1)[0]],
                      labels_low=[SUPER[i] for i in np.where(y_test[lo] == 1)[0]])

    fig, axes = plt.subplots(13, 2, figsize=(13.5, 14), sharex=True,
                             gridspec_kw={'height_ratios': [1] * 12 + [0.55]})
    for col, (tag, j) in enumerate([('high-confidence', hi), ('low-confidence', lo)]):
        ig = integrated_gradients(X_test[j], k)
        cam = grad_cam_1d(X_test[j], k)
        a = np.abs(ig); a = a / (a.max() + 1e-9)
        for l in range(12):
            ax = axes[l, col]
            ax.plot(t, X_test[j][l], color='0.15', lw=0.6)
            ax.scatter(t, X_test[j][l], c=a[l], cmap='Reds', s=4, vmin=0, vmax=1)
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(LEADS[l], rotation=0, ha='right', va='center', fontsize=8)
        axes[12, col].imshow(cam[None], aspect='auto', cmap='viridis', extent=[t[0], t[-1], 0, 1])
        axes[12, col].set_yticks([]); axes[12, col].set_xlabel('time (s)')
        if col == 0:
            axes[12, col].set_ylabel('Grad-CAM', rotation=0, ha='right', va='center', fontsize=8)
        axes[0, col].set_title(f'{tag}: TEST idx {j}, age {age_test[j]:.0f}, '
                               f'p({sc})={P[j, k]:.2f}, entropy {entropy[j]:.2f}\n'
                               f'true labels: {", ".join(SUPER[i] for i in np.where(y_test[j] == 1)[0])}',
                               fontsize=9)
    fig.suptitle(f'{sc}: Integrated-Gradients overlay and Grad-CAM-1D for a confident and an '
                 f'unconfident correctly-labelled 80+ patient', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f'12_xai_pair_{sc}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {sc}: high idx {hi} (p={P[hi,k]:.2f}) vs low idx {lo} (p={P[lo,k]:.2f})')

with open(ROOT / 'outputs' / 'dataset_validation' / '12_xai_pair_cases.json', 'w') as f:
    json.dump(picked, f, indent=1)
print(json.dumps(picked, indent=1))
