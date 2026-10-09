"""Review Item 5 — seed-variance replication of the primary pipeline.

The previous review round correctly discounted every reported number as single-seed
(seed 42 throughout). This script retrains the pipeline's anchor architecture,
InceptionTime1D (depth 4, the NB03/NB05 model), at three further seeds under the
hyperparameters recorded in the manuscript's reproducibility section, then pushes each
seed all the way through the downstream pipeline that the age-stratified claims rest on:

    ensemble re-weighting -> per-class temperature scaling -> global + Mondrian
    conformal thresholds -> prediction-set size by age band -> trend statistic.

The two heavier ensemble members (DualBranchECGNet, ResNet1D101) are held at their
released checkpoints; only the anchor is reseeded. That isolates seed variance in the
component we can afford to retrain on CPU while still reporting its effect on the
*pipeline* numbers rather than on an isolated single model.

Outputs
-------
outputs/dataset_validation/13_multiseed_replication.json   all per-seed metrics
outputs/13_seed{S}_InceptionTime1D_best.pth                per-seed best-VAL checkpoint

Run:  .venv/bin/python analysis/13_multiseed_replication.py
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
ANCHOR_SEED = 42                 # the released checkpoint
NEW_SEEDS = (1, 7, 13)           # retrained here
SUPER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LABEL_COLS = ['is_NORM', 'is_MI', 'is_STTC', 'is_CD', 'is_HYP']
NC = 5
ALPHA = 0.10
BAND_NAMES = ['<40', '40-65', '65-80', '80+']
BAND_MID = {'<40': 32.5, '40-65': 52.5, '65-80': 72.5, '80+': 85.0}
DEVICE = torch.device('cpu')

# training hyperparameters — manuscript section 10, verbatim
MAX_EPOCHS, BATCH, LR, WD = 10, 64, 1e-3, 1e-4
PLATEAU_FACTOR, PLATEAU_PATIENCE, ES_PATIENCE = 0.5, 2, 4

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'outputs'
OUT_VAL = OUT / 'dataset_validation'
OUT_VAL.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get('NB13_CACHE', '/tmp/nb13_cache'))
CACHE.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(int(os.environ.get('NB13_THREADS', '4')))


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
df_train = meta[meta['split'] == 'TRAIN']
df_cal = meta[meta['split'] == 'CONFORMAL_CALIBRATION'].reset_index(drop=True)
df_test = meta[meta['split'] == 'TEST'].reset_index(drop=True)
age_mean, age_std = df_train['age'].mean(), df_train['age'].std()


def demographics(df: pd.DataFrame) -> np.ndarray:
    a = ((df['age'].fillna(age_mean) - age_mean) / (age_std + 1e-6)).values
    s = (df['sex'] == 1).values.astype(np.float32)
    return np.column_stack([a, s]).astype(np.float32)


demo_cal, demo_test = demographics(df_cal), demographics(df_test)
assert np.array_equal(y_test, df_test[LABEL_COLS].values.astype(np.float32))
assert np.array_equal(y_cal, df_cal[LABEL_COLS].values.astype(np.float32))

bands_cal = {'<40': age_cal < 40, '40-65': (age_cal >= 40) & (age_cal < 65),
             '65-80': (age_cal >= 65) & (age_cal < 80), '80+': age_cal >= 80}
bands_test = {'<40': age_test < 40, '40-65': (age_test >= 40) & (age_test < 65),
              '65-80': (age_test >= 65) & (age_test < 80), '80+': age_test >= 80}


# ----------------------------------------------------------------------------- models (verbatim from NB05)
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


CKPT = {'DualBranchECGNet': OUT / 'ensemble_DualBranchECGNet_best.pth',
        'ResNet1D101': OUT / 'ensemble_ResNet1D101_best.pth',
        'InceptionTime1D': OUT / '05_optimized_InceptionTime1D_best.pth'}


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


def downstream(p_incep_cal, p_incep_test, fixed):
    """Full pipeline downstream of the anchor model, for one seed's anchor probabilities."""
    members = dict(fixed)
    members['InceptionTime1D'] = dict(p_cal=p_incep_cal, p_test=p_incep_test)
    order = ['DualBranchECGNet', 'InceptionTime1D', 'ResNet1D101']

    cal_aucs = np.array([macro_auc(y_cal, members[n]['p_cal']) for n in order])
    w = cal_aucs / cal_aucs.sum()
    p_ens_cal = sum(w[i] * members[n]['p_cal'] for i, n in enumerate(order))
    p_ens_test = sum(w[i] * members[n]['p_test'] for i, n in enumerate(order))

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
        ensemble_weights={n: float(w[i]) for i, n in enumerate(order)},
        temperature={c: float(T[k]) for k, c in enumerate(SUPER)},
        macro_auc_test_ensemble=macro_auc(y_test, p_ens_test),
        macro_auc_test_anchor=macro_auc(y_test, p_incep_test),
        macro_auc_cal_anchor=macro_auc(y_cal, p_incep_cal),
        global_thr={c: float(g_thr[k]) for k, c in enumerate(SUPER)},
        set_size_global=trend(ss_global),
        set_size_mondrian=trend(ss_mondrian),
        exact_match=trend(exact),
    )


# ----------------------------------------------------------------------------- training
def train_anchor(seed: int, Xtr, ytr, Xva, yva):
    """Retrain InceptionTime1D under the manuscript's recorded hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)

    pos = ytr.sum(0)
    pos_weight = torch.tensor(((len(ytr) - pos) / np.clip(pos, 1, None)), dtype=torch.float32, device=DEVICE)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                       batch_size=BATCH, shuffle=True, drop_last=True, generator=g)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                       batch_size=128, shuffle=False)

    model = InceptionTime1D(NC).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=PLATEAU_FACTOR,
                                                      patience=PLATEAU_PATIENCE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @torch.no_grad()
    def val_auc():
        model.eval()
        ps = [torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy() for xb, _ in va_dl]
        return macro_auc(yva, np.vstack(ps))

    best, best_ep, wait, best_state, history = -1.0, -1, 0, None, []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item()
        v = val_auc()
        sched.step(v)
        history.append(dict(epoch=ep, loss=tot / max(1, len(tr_dl)), val_macro_auc=v))
        log(f'  seed {seed} epoch {ep:02d}/{MAX_EPOCHS} loss {tot / max(1, len(tr_dl)):.4f} val {v:.4f}')
        if v > best:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                log(f'  seed {seed} early stop at epoch {ep} (best epoch {best_ep}, val {best:.4f})')
                break

    model.load_state_dict(best_state)
    ckpt = OUT / f'13_seed{seed}_InceptionTime1D_best.pth'
    torch.save(best_state, ckpt)
    return model, dict(best_val_macro_auc=best, best_epoch=best_ep,
                       epochs_run=len(history), history=history, checkpoint=ckpt.name)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    X_cal, X_test = load_signals('X_cal_clean_100'), load_signals('X_test_clean_100')

    # fixed ensemble members, cached — these are never reseeded
    cache_f = CACHE / 'fixed_members.npz'
    if cache_f.exists():
        z = np.load(cache_f)
        fixed = {n: {k: z[f'{n}_{k}'] for k in ('p_cal', 'p_test')}
                 for n in ('DualBranchECGNet', 'ResNet1D101')}
        log(f'fixed member probabilities loaded from {cache_f}')
    else:
        fixed, store = {}, {}
        for name, cls, needs_demo in [('DualBranchECGNet', DualBranchECGNet, True),
                                      ('ResNet1D101', ResNet1D101, False)]:
            m = cls(NC)
            m.load_state_dict(torch.load(CKPT[name], map_location=DEVICE, weights_only=True))
            _, pc = infer(m, X_cal, demo_cal if needs_demo else None)
            _, pt = infer(m, X_test, demo_test if needs_demo else None)
            fixed[name] = dict(p_cal=pc, p_test=pt)
            store[f'{name}_p_cal'], store[f'{name}_p_test'] = pc, pt
            log(f'  {name:<18s} CAL {macro_auc(y_cal, pc):.4f}  TEST {macro_auc(y_test, pt):.4f}')
        np.savez_compressed(cache_f, **store)

    seeds: dict[str, dict] = {}

    # --- seed 42: the released checkpoint, re-scored so it is measured identically
    log(f'seed {ANCHOR_SEED} (released checkpoint) — scoring')
    m = InceptionTime1D(NC)
    m.load_state_dict(torch.load(CKPT['InceptionTime1D'], map_location=DEVICE, weights_only=True))
    _, pc = infer(m, X_cal)
    _, pt = infer(m, X_test)
    seeds[str(ANCHOR_SEED)] = dict(source='released checkpoint 05_optimized_InceptionTime1D_best.pth',
                                   retrained=False, **downstream(pc, pt, fixed))
    log(f'  seed {ANCHOR_SEED} anchor TEST {seeds[str(ANCHOR_SEED)]["macro_auc_test_anchor"]:.4f} '
        f'ensemble TEST {seeds[str(ANCHOR_SEED)]["macro_auc_test_ensemble"]:.4f}')

    # --- new seeds: retrain the anchor, then push through the same pipeline
    Xtr, Xva = load_signals('X_train_clean_100'), load_signals('X_val_clean_100')
    log(f'train {Xtr.shape}  val {Xva.shape}  cal {X_cal.shape}  test {X_test.shape}')
    for s in NEW_SEEDS:
        log(f'seed {s} — training anchor InceptionTime1D')
        model, tinfo = train_anchor(s, Xtr, y_train, Xva, y_val)
        _, pc = infer(model, X_cal)
        _, pt = infer(model, X_test)
        seeds[str(s)] = dict(source='retrained here', retrained=True, training=tinfo,
                             **downstream(pc, pt, fixed))
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

    out = dict(
        meta=dict(anchor_seed=ANCHOR_SEED, new_seeds=list(NEW_SEEDS), alpha=ALPHA,
                  reseeded_component='InceptionTime1D (depth 4) — NB03/NB05 anchor',
                  fixed_components=['DualBranchECGNet', 'ResNet1D101'],
                  hyperparameters=dict(max_epochs=MAX_EPOCHS, batch=BATCH, lr=LR, weight_decay=WD,
                                       plateau_factor=PLATEAU_FACTOR, plateau_patience=PLATEAU_PATIENCE,
                                       early_stop_patience=ES_PATIENCE),
                  band_midpoints=BAND_MID, device=str(DEVICE),
                  torch=torch.__version__, numpy=np.__version__,
                  wall_clock_seconds=round(time.time() - t0, 1)),
        seeds=seeds,
        summary=summary,
    )
    dest = OUT_VAL / '13_multiseed_replication.json'
    dest.write_text(json.dumps(out, indent=2))
    log(f'wrote {dest}')

    a = summary['anchor_macro_auc_test']
    e = summary['ensemble_macro_auc_test']
    g = summary['global_slope_per_decade']
    log(f'ANCHOR  macro-AUC {a["mean"]:.4f} +/- {a["sd"]:.4f}  (range {a["min"]:.4f}-{a["max"]:.4f})')
    log(f'ENSEMBLE macro-AUC {e["mean"]:.4f} +/- {e["sd"]:.4f}  (range {e["min"]:.4f}-{e["max"]:.4f})')
    log(f'GLOBAL set-size slope/decade {g["mean"]:.4f} +/- {g["sd"]:.4f}')
    log(f'trend consistent across seeds: {summary["age_trend_direction_consistent"]}')


if __name__ == '__main__':
    main()
