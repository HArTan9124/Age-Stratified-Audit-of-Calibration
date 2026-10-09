"""Review Item 5, residual — full three-architecture ensemble seed variance.

NB13 reseeded only the anchor (InceptionTime1D) because training here is CPU-only, and
said so plainly: two of the ensemble's three members were identical across every row of
its Table 8, so the +/-0.0005 in that table's ensemble column is a **lower bound, not an
estimate**. The supervisor accepted that scoping and asked for the remaining two members
when the compute was available. This script is that run.

It retrains DualBranchECGNet and ResNet1D101 at seeds 1, 7 and 13, pairs each with the
anchor checkpoint NB13 already produced for the same seed, and pushes every seed through
the identical downstream pipeline:

    ensemble re-weighting -> per-class temperature scaling -> global + Mondrian
    conformal thresholds -> prediction-set size by age band -> trend statistic.

Each seed row therefore has all three members trained at that seed, which is what makes
the ensemble column a real estimate of ensemble seed variance rather than a lower bound.

Recipe fidelity — the point that decides whether this run means anything
-----------------------------------------------------------------------
Seed 42 for these two members *is* the NB02 checkpoint, so the retrained seeds must use
**NB02's** recipe, not NB13's. The two differ:

    NB02 (these two members)  AdamW lr 1e-3, wd 1e-4, CosineAnnealingLR(T_max=10),
                              10 fixed epochs, no early stopping, best-on-VAL checkpoint
    NB13 (the anchor)         AdamW lr 1e-3, wd 1e-4, ReduceLROnPlateau(0.5, patience 2),
                              max 10 epochs, early stopping (patience 4)

Training the new seeds under NB13's schedule would confound seed variance with a schedule
change and make the spread uninterpretable. So this script reproduces NB02's loop verbatim
for DualBranchECGNet and ResNet1D101, and reuses NB13's anchor checkpoints untouched for
InceptionTime1D. Neither architecture is retrained under a recipe its seed-42 counterpart
did not use.

Guard before cost
-----------------
Module 5 re-derives NB13's seed-42 row from the released checkpoints and asserts it matches
`13_multiseed_replication.json` to 1e-9 before any training starts. If the data, the
checkpoints or the pipeline have moved, the run aborts in about a minute instead of
spending twelve hours producing numbers that cannot be compared with the published ones.

Resume-safe: every (model, seed) checkpoint is written separately and reloaded if present,
so an interrupted run continues where it stopped rather than restarting.

Cost
----
Roughly 12 h on CPU for 6 trainings (3 seeds x 2 members), against the 4.42 h NB13 spent on
3 anchor trainings. ResNet1D101 is the expensive one. Run it detached.

Outputs
-------
outputs/dataset_validation/14_full_ensemble_multiseed.json   all per-seed metrics
outputs/14_seed{S}_{Model}_best.pth                          per-seed best-VAL checkpoints

Run:  nohup .venv/bin/python analysis/14_full_ensemble_multiseed.py > outputs/14_run.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------------- config
ANCHOR_SEED = 42                 # the released checkpoints, for all three members
NEW_SEEDS = (1, 7, 13)           # must match NB13's, or the anchor checkpoints will not exist
SUPER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LABEL_COLS = ['is_NORM', 'is_MI', 'is_STTC', 'is_CD', 'is_HYP']
NC = 5
ALPHA = 0.10
BAND_NAMES = ['<40', '40-65', '65-80', '80+']
BAND_MID = {'<40': 32.5, '40-65': 52.5, '65-80': 72.5, '80+': 85.0}
DEVICE = torch.device('cpu')
ORDER = ['DualBranchECGNet', 'InceptionTime1D', 'ResNet1D101']
RESEEDED = ['DualBranchECGNet', 'ResNet1D101']   # what this script trains

# NB02's recipe — the one the seed-42 checkpoints for these two members were trained under.
# Deliberately different from NB13's; see the module docstring.
MAX_EPOCHS, BATCH, LR, WD = 10, 64, 1e-3, 1e-4
COSINE_TMAX = MAX_EPOCHS

# how closely the seed-42 row must reproduce NB13 before the expensive part starts
GUARD_TOL = 1e-9

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'outputs'
OUT_VAL = OUT / 'dataset_validation'
OUT_VAL.mkdir(parents=True, exist_ok=True)
NB13_JSON = OUT_VAL / '13_multiseed_replication.json'

torch.set_num_threads(int(os.environ.get('NB14_THREADS', '4')))


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ----------------------------------------------------------------------------- data
def load_signals(stem: str) -> np.ndarray:
    """(N, T, 12) on disk -> (N, 12, T) float32, as every notebook in the project loads it."""
    return np.transpose(np.load(DATA / f'{stem}.npy'), (0, 2, 1)).astype(np.float32)


y_train = np.load(DATA / 'y_train_diag_100.npy').astype(np.float32)
y_val = np.load(DATA / 'y_valid_diag_100.npy').astype(np.float32)
y_cal = np.load(DATA / 'y_calib_diag_100.npy').astype(np.float32)
y_test = np.load(DATA / 'y_test_diag_100.npy').astype(np.float32)
age_cal = np.load(DATA / 'y_calib_age_100.npy').astype(np.float32)
age_test = np.load(DATA / 'y_test_age_100.npy').astype(np.float32)

meta = pd.read_csv(DATA / 'MASTER_RESEARCH_METADATA.csv')
df_train = meta[meta['split'] == 'TRAIN'].reset_index(drop=True)
df_val = meta[meta['split'] == 'VALIDATION'].reset_index(drop=True)
df_cal = meta[meta['split'] == 'CONFORMAL_CALIBRATION'].reset_index(drop=True)
df_test = meta[meta['split'] == 'TEST'].reset_index(drop=True)
age_mean, age_std = df_train['age'].mean(), df_train['age'].std()


def demographics(df: pd.DataFrame) -> np.ndarray:
    a = ((df['age'].fillna(age_mean) - age_mean) / (age_std + 1e-6)).values
    s = (df['sex'] == 1).values.astype(np.float32)
    return np.column_stack([a, s]).astype(np.float32)


# the dual-branch member needs demographics on every split it sees, training included
demo_train, demo_val = demographics(df_train), demographics(df_val)
demo_cal, demo_test = demographics(df_cal), demographics(df_test)
assert np.array_equal(y_test, df_test[LABEL_COLS].values.astype(np.float32))
assert np.array_equal(y_cal, df_cal[LABEL_COLS].values.astype(np.float32))
assert np.array_equal(y_train, df_train[LABEL_COLS].values.astype(np.float32))
assert np.array_equal(y_val, df_val[LABEL_COLS].values.astype(np.float32))

bands_cal = {'<40': age_cal < 40, '40-65': (age_cal >= 40) & (age_cal < 65),
             '65-80': (age_cal >= 65) & (age_cal < 80), '80+': age_cal >= 80}
bands_test = {'<40': age_test < 40, '40-65': (age_test >= 40) & (age_test < 65),
              '65-80': (age_test >= 65) & (age_test < 80), '80+': age_test >= 80}


# ----------------------------------------------------------------------------- models (verbatim from NB05/NB13)
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
        outs = [c(b) for c in self.convs] + [self.pool_conv(self.pool(x))]
        return self.act(self.bn(torch.cat(outs, dim=1)))


class InceptionTime1D(nn.Module):
    def __init__(self, num_classes=5, in_channels=12, nb_filters=32, depth=4):
        super().__init__()
        ch = [in_channels] + [nb_filters * 4] * depth
        self.blocks = nn.Sequential(*[InceptionBlock1D(ch[i], nb_filters) for i in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.drop(x))


class DualBranchECGNet(nn.Module):
    def __init__(self, num_classes=5, in_channels=12):
        super().__init__()
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.4),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.4),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.4),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.4),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.demo_branch = nn.Sequential(
            nn.Linear(2, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(100, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(528, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes))

    def forward(self, ecg, demo):
        return self.classifier(torch.cat([self.ecg_branch(ecg).squeeze(-1), self.demo_branch(demo)], dim=1))


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                                      nn.BatchNorm1d(out_channels))
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class ResNet1D101(nn.Module):
    def __init__(self, num_classes=5, in_channels=12, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1))

        def stage(i, o, n, stride=2):
            return nn.Sequential(ResBlock1D(i, o, stride), *[ResBlock1D(o, o) for _ in range(n - 1)])

        self.stage1 = stage(64, 64, 8, stride=1)
        self.stage2 = stage(64, 128, 8, stride=2)
        self.stage3 = stage(128, 256, 8, stride=2)
        self.stage4 = stage(256, 512, 8, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


ARCH = {'DualBranchECGNet': (DualBranchECGNet, True),
        'InceptionTime1D': (InceptionTime1D, False),
        'ResNet1D101': (ResNet1D101, False)}

# seed 42 = the released checkpoints, exactly the ones NB13 scored
CKPT42 = {'DualBranchECGNet': OUT / 'ensemble_DualBranchECGNet_best.pth',
          'ResNet1D101': OUT / 'ensemble_ResNet1D101_best.pth',
          'InceptionTime1D': OUT / '05_optimized_InceptionTime1D_best.pth'}


def ckpt_path(name: str, seed: int) -> Path:
    """Where seed S's checkpoint for a member lives."""
    if seed == ANCHOR_SEED:
        return CKPT42[name]
    if name == 'InceptionTime1D':
        return OUT / f'13_seed{seed}_InceptionTime1D_best.pth'   # NB13 produced these
    return OUT / f'14_seed{seed}_{name}_best.pth'                 # this script produces these


# ----------------------------------------------------------------------------- shared pipeline pieces
@torch.no_grad()
def infer(model, X, demo=None, bs=64):
    model.eval().to(DEVICE)
    logits = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        lb = model(xb) if demo is None else model(xb, torch.from_numpy(demo[i:i + bs]).to(DEVICE))
        logits.append(lb.cpu().numpy())
    z = np.vstack(logits)
    return z, 1.0 / (1.0 + np.exp(-z))


def macro_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])]))


EPS = 1e-6


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))


def fit_temperature(z, y):
    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-z / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.25, 5.0), method='bounded').x)


def conformal_thresholds(p_cal_, y_cal_, alpha=ALPHA):
    thr = np.zeros(NC)
    for k in range(NC):
        s = 1.0 - p_cal_[y_cal_[:, k] == 1, k]
        n = len(s)
        if n == 0:
            thr[k] = 0.0
            continue
        qlevel = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        thr[k] = 1.0 - np.quantile(s, qlevel, method='higher')
    return thr


def downstream(members):
    """Everything after the three members, for one seed.

    NB13's `downstream()` took the anchor's probabilities and closed over two fixed
    members. Here every member varies by seed, so the whole set is passed in. The body
    is otherwise identical, which is what keeps NB14's rows comparable with NB13's.
    """
    cal_aucs = np.array([macro_auc(y_cal, members[n]['p_cal']) for n in ORDER])
    w = cal_aucs / cal_aucs.sum()
    p_ens_cal = sum(w[i] * members[n]['p_cal'] for i, n in enumerate(ORDER))
    p_ens_test = sum(w[i] * members[n]['p_test'] for i, n in enumerate(ORDER))

    z_cal, z_test = logit(p_ens_cal), logit(p_ens_test)
    T = np.array([fit_temperature(z_cal[:, k], y_cal[:, k]) for k in range(NC)])
    p_cal_c = 1.0 / (1.0 + np.exp(-z_cal / T))
    p_test_c = 1.0 / (1.0 + np.exp(-z_test / T))

    g_thr = conformal_thresholds(p_cal_c, y_cal)
    m_thr = {b: conformal_thresholds(p_cal_c[m], y_cal[m]) for b, m in bands_cal.items()}

    ss_global = (p_test_c >= g_thr).sum(1).astype(float)
    ss_mondrian = np.zeros(len(p_test_c))
    for b, m in bands_test.items():
        ss_mondrian[m] = (p_test_c[m] >= m_thr[b]).sum(1)

    yhat = (p_test_c >= g_thr).astype(np.float32)
    exact = (yhat == y_test).all(1).astype(float)

    mids = np.array([BAND_MID[b] for b in BAND_NAMES])

    def by_band(v):
        return {b: float(np.mean(v[m])) for b, m in bands_test.items()}

    def trend(v):
        rho, p = spearmanr(age_test, v)
        band_means = [np.mean(v[bands_test[b]]) for b in BAND_NAMES]
        return dict(spearman_rho_vs_age=float(rho), spearman_p=float(p),
                    band_means=by_band(v),
                    slope_per_decade=float(np.polyfit(mids, band_means, 1)[0] * 10),
                    monotone=bool(all(np.diff(band_means) > 0)))

    return dict(
        ensemble_weights={n: float(w[i]) for i, n in enumerate(ORDER)},
        member_macro_auc_test={n: macro_auc(y_test, members[n]['p_test']) for n in ORDER},
        member_macro_auc_cal={n: macro_auc(y_cal, members[n]['p_cal']) for n in ORDER},
        temperature={c: float(T[k]) for k, c in enumerate(SUPER)},
        macro_auc_test_ensemble=macro_auc(y_test, p_ens_test),
        macro_auc_test_anchor=macro_auc(y_test, members['InceptionTime1D']['p_test']),
        macro_auc_cal_anchor=macro_auc(y_cal, members['InceptionTime1D']['p_cal']),
        global_thr={c: float(g_thr[k]) for k, c in enumerate(SUPER)},
        set_size_global=trend(ss_global),
        set_size_mondrian=trend(ss_mondrian),
        exact_match=trend(exact),
    )


# ----------------------------------------------------------------------------- training (NB02's recipe)
def train_member(name: str, seed: int, Xtr, Xva):
    """Retrain one ensemble member under NB02's loop — cosine schedule, 10 fixed epochs.

    Not NB13's schedule. Seed 42 for this member is the NB02 checkpoint, so anything else
    would mix a schedule change into what is supposed to be pure seed variance.
    """
    cls, needs_demo = ARCH[name]
    dest = ckpt_path(name, seed)

    if dest.exists():                       # resume-safe: a 12 h run must survive a stop
        model = cls(NC)
        model.load_state_dict(torch.load(dest, map_location=DEVICE, weights_only=True))
        _, pv = infer(model, Xva, demo_val if needs_demo else None)
        log(f'  {name} seed {seed}: checkpoint present, reusing (VAL {macro_auc(y_val, pv):.4f})')
        return model, dict(resumed=True, best_val_macro_auc=macro_auc(y_val, pv),
                           checkpoint=dest.name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)

    pos = y_train.sum(0)
    pos_weight = torch.tensor((len(y_train) - pos) / (pos + 1e-6),
                              dtype=torch.float32, device=DEVICE)   # NB02's formula

    tensors = [torch.from_numpy(Xtr)]
    if needs_demo:
        tensors.append(torch.from_numpy(demo_train))
    tensors.append(torch.from_numpy(y_train))
    tr_dl = DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True, generator=g)

    model = cls(NC).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_TMAX)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in tr_dl:
            xb, yb = batch[0].to(DEVICE), batch[-1].to(DEVICE)
            opt.zero_grad()
            out = model(xb, batch[1].to(DEVICE)) if needs_demo else model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()                        # NB02 steps the cosine schedule per epoch

        _, pv = infer(model, Xva, demo_val if needs_demo else None)
        v = macro_auc(y_val, pv)
        history.append(dict(epoch=ep, loss=tot / max(1, nb), val_macro_auc=v,
                            lr=opt.param_groups[0]['lr']))
        log(f'  {name} seed {seed} epoch {ep:02d}/{MAX_EPOCHS} '
            f'loss {tot / max(1, nb):.4f} val {v:.4f} ({(time.time() - t0) / 60:.1f} min)')
        if v > best:                        # NB02 keeps the best-on-VAL state, no early stop
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, dest)
    log(f'  {name} seed {seed}: best VAL {best:.4f} at epoch {best_ep}, '
        f'{(time.time() - t0) / 60:.1f} min -> {dest.name}')
    return model, dict(resumed=False, best_val_macro_auc=best, best_epoch=best_ep,
                       epochs_run=len(history), history=history,
                       minutes=round((time.time() - t0) / 60, 1), checkpoint=dest.name)


# ----------------------------------------------------------------------------- preflight
def preflight():
    """Everything that must be true before twelve hours of CPU is worth spending."""
    missing = [str(p) for p in CKPT42.values() if not p.exists()]
    for s in NEW_SEEDS:
        p = ckpt_path('InceptionTime1D', s)
        if not p.exists():
            missing.append(f'{p}  (NB13 produces this — run analysis/13 first)')
    if missing:
        raise SystemExit('[ABORT] missing checkpoints:\n  ' + '\n  '.join(missing))
    if not NB13_JSON.exists():
        raise SystemExit(f'[ABORT] {NB13_JSON} not found; NB14 is compared against it.')
    log('preflight: all nine checkpoints and the NB13 result file are present')
    return json.loads(NB13_JSON.read_text())


def score_seed(seed: int, X_cal, X_test, Xtr=None, Xva=None):
    """Load or train every member at this seed, then run the downstream pipeline once."""
    members, training = {}, {}
    for name in ORDER:
        cls, needs_demo = ARCH[name]
        if seed == ANCHOR_SEED or name not in RESEEDED:
            model = cls(NC)
            model.load_state_dict(torch.load(ckpt_path(name, seed), map_location=DEVICE,
                                             weights_only=True))
            src = 'released checkpoint' if seed == ANCHOR_SEED else 'NB13 checkpoint'
        else:
            model, tinfo = train_member(name, seed, Xtr, Xva)
            training[name] = tinfo
            src = 'retrained here'
        _, pc = infer(model, X_cal, demo_cal if needs_demo else None)
        _, pt = infer(model, X_test, demo_test if needs_demo else None)
        members[name] = dict(p_cal=pc, p_test=pt, source=src)
        log(f'  {name:<18s} seed {seed:<3d} {src:<20s} CAL {macro_auc(y_cal, pc):.4f} '
            f'TEST {macro_auc(y_test, pt):.4f}')
    row = dict(member_source={n: members[n]['source'] for n in ORDER}, **downstream(members))
    if training:
        row['training'] = training
    return row


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    nb13 = preflight()

    X_cal, X_test = load_signals('X_cal_clean_100'), load_signals('X_test_clean_100')
    seeds: dict[str, dict] = {}

    # --- guard: reproduce NB13's seed 42 exactly before spending anything
    log(f'seed {ANCHOR_SEED} (released checkpoints) — scoring as the reproduction guard')
    row42 = score_seed(ANCHOR_SEED, X_cal, X_test)
    ref = nb13['seeds'][str(ANCHOR_SEED)]
    drift = {k: (row42[k], ref[k]) for k in ('macro_auc_test_ensemble', 'macro_auc_test_anchor')
             if abs(row42[k] - ref[k]) > GUARD_TOL}
    drift.update({f'weight_{n}': (row42['ensemble_weights'][n], ref['ensemble_weights'][n])
                  for n in ORDER if abs(row42['ensemble_weights'][n] - ref['ensemble_weights'][n]) > GUARD_TOL})
    if drift:
        for k, (got, want) in drift.items():
            log(f'  MISMATCH {k}: got {got!r}, NB13 recorded {want!r}')
        raise SystemExit('[ABORT] seed 42 does not reproduce NB13; nothing was trained.')
    log(f'  guard passed — ensemble TEST {row42["macro_auc_test_ensemble"]:.10f} '
        f'matches NB13 to {GUARD_TOL:g}')
    seeds[str(ANCHOR_SEED)] = row42

    # --- the expensive part: retrain the two heavy members at each new seed
    Xtr, Xva = load_signals('X_train_clean_100'), load_signals('X_val_clean_100')
    log(f'train {Xtr.shape}  val {Xva.shape}  cal {X_cal.shape}  test {X_test.shape}')
    log(f'training {len(RESEEDED)} members x {len(NEW_SEEDS)} seeds — expect roughly 12 h on CPU')
    for s in NEW_SEEDS:
        log(f'seed {s} — retraining {", ".join(RESEEDED)}; anchor reused from NB13')
        seeds[str(s)] = score_seed(s, X_cal, X_test, Xtr, Xva)
        log(f'  seed {s} anchor TEST {seeds[str(s)]["macro_auc_test_anchor"]:.4f} '
            f'ensemble TEST {seeds[str(s)]["macro_auc_test_ensemble"]:.4f}')

    # ----------------------------------------------------------------- across-seed summary
    allseeds = [str(ANCHOR_SEED)] + [str(s) for s in NEW_SEEDS]

    def agg(getter):
        v = np.array([getter(seeds[s]) for s in allseeds], dtype=float)
        return dict(values={s: float(getter(seeds[s])) for s in allseeds},
                    mean=float(v.mean()), sd=float(v.std(ddof=1)),
                    min=float(v.min()), max=float(v.max()), range=float(v.max() - v.min()))

    summary = dict(
        anchor_macro_auc_test=agg(lambda d: d['macro_auc_test_anchor']),
        ensemble_macro_auc_test=agg(lambda d: d['macro_auc_test_ensemble']),
        mondrian_slope_per_decade=agg(lambda d: d['set_size_mondrian']['slope_per_decade']),
        global_slope_per_decade=agg(lambda d: d['set_size_global']['slope_per_decade']),
        mondrian_spearman_rho=agg(lambda d: d['set_size_mondrian']['spearman_rho_vs_age']),
        global_spearman_rho=agg(lambda d: d['set_size_global']['spearman_rho_vs_age']),
        exact_match_spearman_rho=agg(lambda d: d['exact_match']['spearman_rho_vs_age']),
    )
    for name in ORDER:
        summary[f'member_macro_auc_test_{name}'] = agg(
            lambda d, n=name: d['member_macro_auc_test'][n])
    for band in BAND_NAMES:
        summary[f'mondrian_set_size_{band}'] = agg(lambda d, b=band: d['set_size_mondrian']['band_means'][b])
        summary[f'global_set_size_{band}'] = agg(lambda d, b=band: d['set_size_global']['band_means'][b])

    # the claim the review actually needs: does the age gradient survive every seed?
    summary['age_trend_direction_consistent'] = dict(
        global_set_size_positive_in_all_seeds=bool(all(seeds[s]['set_size_global']['spearman_rho_vs_age'] > 0
                                                       for s in allseeds)),
        mondrian_set_size_positive_in_all_seeds=bool(all(seeds[s]['set_size_mondrian']['spearman_rho_vs_age'] > 0
                                                         for s in allseeds)),
        exact_match_negative_in_all_seeds=bool(all(seeds[s]['exact_match']['spearman_rho_vs_age'] < 0
                                                   for s in allseeds)),
        global_set_size_monotone_in_all_seeds=bool(all(seeds[s]['set_size_global']['monotone'] for s in allseeds)),
        all_spearman_p_below_0_001=bool(all(seeds[s]['set_size_global']['spearman_p'] < 1e-3 for s in allseeds)),
    )

    # the headline this run exists to produce: how much the lower bound understated things
    nb13_sd = nb13['summary']['ensemble_macro_auc_test']['sd']
    nb14_sd = summary['ensemble_macro_auc_test']['sd']
    summary['vs_nb13'] = dict(
        nb13_ensemble_sd_anchor_only=nb13_sd,
        nb14_ensemble_sd_all_three=nb14_sd,
        inflation_factor=float(nb14_sd / nb13_sd) if nb13_sd > 0 else None,
        note=('NB13 reseeded one member of three, so its ensemble SD was a lower bound. '
              'This run reseeds all three, so its ensemble SD is the estimate. The anchor '
              'column is identical in both runs by construction — NB14 reuses NB13 '
              'checkpoints — which is what makes the two ensemble columns comparable.'),
    )

    out = dict(
        meta=dict(anchor_seed=ANCHOR_SEED, new_seeds=list(NEW_SEEDS), alpha=ALPHA,
                  reseeded_components=RESEEDED,
                  reused_from_nb13=['InceptionTime1D (depth 4) — NB03/NB05 anchor'],
                  recipe='NB02 — AdamW, CosineAnnealingLR(T_max=10), 10 fixed epochs, '
                         'no early stopping, best-on-VAL checkpoint; this is the recipe the '
                         'seed-42 checkpoints for these two members were trained under',
                  hyperparameters=dict(max_epochs=MAX_EPOCHS, batch=BATCH, lr=LR,
                                       weight_decay=WD, scheduler='CosineAnnealingLR',
                                       cosine_t_max=COSINE_TMAX, early_stopping=False),
                  band_midpoints=BAND_MID, device=str(DEVICE),
                  torch=torch.__version__, numpy=np.__version__,
                  threads=torch.get_num_threads(),
                  guard_tolerance=GUARD_TOL,
                  supersedes='outputs/dataset_validation/13_multiseed_replication.json '
                             '(ensemble column only; the anchor column is unchanged)',
                  wall_clock_seconds=round(time.time() - t0, 1)),
        seeds=seeds,
        summary=summary,
    )
    dest = OUT_VAL / '14_full_ensemble_multiseed.json'
    dest.write_text(json.dumps(out, indent=2))
    log(f'wrote {dest}')

    a = summary['anchor_macro_auc_test']
    e = summary['ensemble_macro_auc_test']
    g = summary['global_slope_per_decade']
    log(f'ANCHOR   macro-AUC {a["mean"]:.4f} +/- {a["sd"]:.4f}  (range {a["min"]:.4f}-{a["max"]:.4f})')
    log(f'ENSEMBLE macro-AUC {e["mean"]:.4f} +/- {e["sd"]:.4f}  (range {e["min"]:.4f}-{e["max"]:.4f})')
    for name in ORDER:
        m = summary[f'member_macro_auc_test_{name}']
        log(f'  {name:<18s} {m["mean"]:.4f} +/- {m["sd"]:.4f}')
    log(f'GLOBAL set-size slope/decade {g["mean"]:.4f} +/- {g["sd"]:.4f}')
    log(f'ensemble SD: NB13 {nb13_sd:.6f} (lower bound) -> NB14 {nb14_sd:.6f} (estimate)')
    log(f'trend consistent across seeds: {summary["age_trend_direction_consistent"]}')
    log(f'total wall clock {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
