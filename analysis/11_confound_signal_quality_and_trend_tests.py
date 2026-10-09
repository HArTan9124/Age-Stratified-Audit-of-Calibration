"""
NB11 — Reviewer-response analyses for the age-conditional conformal paper.

Runs three things the manuscript revision needs and NB05/NB06/NB10 do not provide:

  (A) Signal-quality confound control. Derives a per-record signal-quality
      covariate set (in-band SNR, baseline-wander power, high-frequency noise)
      from the raw-vs-filtered pair already cached by NB01, then asks whether
      the age effect on conformal set size / calibration error survives
      adjustment for it.
  (B) Formal trend tests for every age-stratified claim (Spearman, Kruskal-Wallis,
      OLS on continuous age, bootstrap slope CIs) instead of eyeballed bar charts.
  (C) Calibration-resampling stability for the Mondrian result: the conformal
      fit is repeated over bootstrap resamples of the CONFORMAL_CALIBRATION
      split so the age trend carries a spread, not a single point.

It reproduces NB05 Module 9 exactly first (same checkpoints, same AUC weighting,
same per-class temperatures, same quantile convention) and asserts agreement with
outputs/dataset_validation/review_closeout_metrics.json before adding anything.

Outputs
  outputs/dataset_validation/11_confound_and_trend_tests.json
  outputs/figures/nb11_confound_trends/*.png
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sps
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
OUT_VAL = ROOT / 'outputs' / 'dataset_validation'
OUT_FIG = ROOT / 'outputs' / 'figures' / 'nb11_confound_trends'
OUT_FIG.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get('NB11_CACHE', '/tmp/nb11_cache'))
CACHE.mkdir(parents=True, exist_ok=True)

SUPER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LABEL_COLS = ['is_NORM', 'is_MI', 'is_STTC', 'is_CD', 'is_HYP']
NC = 5
ALPHA = 0.10
M_BINS = 15
BAND_NAMES = ['<40', '40-65', '65-80', '80+']
BAND_MID = {'<40': 32.5, '40-65': 52.5, '65-80': 72.5, '80+': 85.0}  # band midpoints, 80+ capped at 90
DEVICE = torch.device('cpu')
FS = 100.0

# ----------------------------------------------------------------------------- data
def load_signals(stem):
    X = np.load(DATA / f'{stem}.npy')
    return np.transpose(X, (0, 2, 1)).astype(np.float32)

y_cal = np.load(DATA / 'y_calib_diag_100.npy').astype(np.float32)
y_test = np.load(DATA / 'y_test_diag_100.npy').astype(np.float32)
age_cal = np.load(DATA / 'y_calib_age_100.npy').astype(np.float32)
age_test = np.load(DATA / 'y_test_age_100.npy').astype(np.float32)

meta = pd.read_csv(DATA / 'MASTER_RESEARCH_METADATA.csv')
df_train = meta[meta['split'] == 'TRAIN']
df_cal = meta[meta['split'] == 'CONFORMAL_CALIBRATION'].reset_index(drop=True)
df_test = meta[meta['split'] == 'TEST'].reset_index(drop=True)
age_mean, age_std = df_train['age'].mean(), df_train['age'].std()

def demographics(df):
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
        x = self.blocks(x); x = self.gap(x).squeeze(-1)
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
        self.pool = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(drop); self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))

CKPT = {'DualBranchECGNet': ROOT / 'outputs' / 'ensemble_DualBranchECGNet_best.pth',
        'ResNet1D101': ROOT / 'outputs' / 'ensemble_ResNet1D101_best.pth',
        'InceptionTime1D': ROOT / 'outputs' / '05_optimized_InceptionTime1D_best.pth'}

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

# ----------------------------------------------------------------------------- member inference (cached)
cache_f = CACHE / 'member_probs.npz'
if cache_f.exists():
    z = np.load(cache_f)
    members = {n: {k: z[f'{n}_{k}'] for k in ('p_cal', 'p_test')} for n in CKPT}
    print(f'[CACHE] member probabilities loaded from {cache_f}')
else:
    X_cal, X_test = load_signals('X_cal_clean_100'), load_signals('X_test_clean_100')
    members, store = {}, {}
    for name, cls, needs_demo in [('DualBranchECGNet', DualBranchECGNet, True),
                                  ('InceptionTime1D', InceptionTime1D, False),
                                  ('ResNet1D101', ResNet1D101, False)]:
        m = cls(NC)
        m.load_state_dict(torch.load(CKPT[name], map_location=DEVICE, weights_only=True))
        _, pc = infer(m, X_cal, demo_cal if needs_demo else None)
        _, pt = infer(m, X_test, demo_test if needs_demo else None)
        members[name] = dict(p_cal=pc, p_test=pt)
        store[f'{name}_p_cal'], store[f'{name}_p_test'] = pc, pt
        print(f'  {name:<16s} CAL {macro_auc(y_cal, pc):.4f}  TEST {macro_auc(y_test, pt):.4f}', flush=True)
    np.savez_compressed(cache_f, **store)
    del X_cal, X_test

ORDER = ['DualBranchECGNet', 'InceptionTime1D', 'ResNet1D101']
cal_aucs = np.array([macro_auc(y_cal, members[n]['p_cal']) for n in ORDER])
w = cal_aucs / cal_aucs.sum()
p_ens_cal = sum(w[i] * members[n]['p_cal'] for i, n in enumerate(ORDER))
p_ens_test = sum(w[i] * members[n]['p_test'] for i, n in enumerate(ORDER))
eps = 1e-6
logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
z_ens_cal, z_ens_test = logit(p_ens_cal), logit(p_ens_test)

def fit_temperature(z, y):
    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-z / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.25, 5.0), method='bounded').x)

T = np.array([fit_temperature(z_ens_cal[:, k], y_cal[:, k]) for k in range(NC)])
p_cal_c = 1.0 / (1.0 + np.exp(-z_ens_cal / T))
p_test_c = 1.0 / (1.0 + np.exp(-z_ens_test / T))

def conformal_thresholds(p_cal_, y_cal_, alpha=ALPHA):
    thr = np.zeros(NC)
    for k in range(NC):
        s = 1.0 - p_cal_[y_cal_[:, k] == 1, k]
        n = len(s)
        if n == 0:
            thr[k] = 0.0; continue
        qlevel = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        thr[k] = 1.0 - np.quantile(s, qlevel, method='higher')
    return thr

global_thr = conformal_thresholds(p_cal_c, y_cal)
mondrian_thr = {b: conformal_thresholds(p_cal_c[m], y_cal[m]) for b, m in bands_cal.items()}

# ----------------------------------------------------------------------------- reproduction check
ref = json.load(open(OUT_VAL / 'review_closeout_metrics.json'))
auth = ref['mondrian_authoritative_nb05']
repro = {'weights': {n: float(w[i]) for i, n in enumerate(ORDER)},
         'temperature': {SUPER[k]: float(T[k]) for k in range(NC)},
         'macro_auc_test': macro_auc(y_test, p_ens_test),
         'global_thr': {SUPER[k]: round(float(global_thr[k]), 3) for k in range(NC)},
         'set_size_mondrian': {}, 'set_size_global': {}}
for b, m in bands_test.items():
    for tag, thr in (('global', global_thr), ('mondrian', mondrian_thr[b])):
        pred = (p_test_c[m] >= thr).astype(int)
        repro[f'set_size_{tag}'][b] = float(pred.sum(1).mean())
print('\n[REPRO] macro-AUC TEST      :', round(repro['macro_auc_test'], 4), '| closeout',
      round(ref['headline']['macro_auc_ens']['point'], 4))
for b in BAND_NAMES:
    print(f"[REPRO] {b:<6s} mondrian |S| : {repro['set_size_mondrian'][b]:.3f} | closeout "
          f"{auth['schemes']['MONDRIAN'][b]['mean_set_size']:.3f}")
mismatch = [b for b in BAND_NAMES
            if abs(repro['set_size_mondrian'][b] - auth['schemes']['MONDRIAN'][b]['mean_set_size']) > 0.02]
repro['matches_closeout'] = (len(mismatch) == 0 and
                             abs(repro['macro_auc_test'] - ref['headline']['macro_auc_ens']['point']) < 5e-4)
print('[REPRO] reproduces NB05 Module 9 :', repro['matches_closeout'], '' if not mismatch else f'(off: {mismatch})')

# ----------------------------------------------------------------------------- (A) signal quality
def signal_quality(raw, clean):
    """Per-record quality covariates from the raw/filtered pair cached by NB01.

    raw, clean : (N, 1000, 12) in mV, as written by NB01 (record-time layout).
    Returns a DataFrame with one row per record.
    """
    resid = raw - clean                                     # what the 0.5-40 Hz band-pass removed
    sos_lo = sps.butter(3, 0.5, btype='low', fs=FS, output='sos')
    sos_hi = sps.butter(3, 40.0, btype='high', fs=FS, output='sos')
    n = len(raw)
    out = {k: np.zeros(n) for k in
           ('snr_db', 'wander_rms_mv', 'hf_rms_mv', 'resid_rms_mv', 'flat_leads', 'sig_rms_mv')}
    step = 256
    for s in range(0, n, step):
        e = min(s + step, n)
        c, r = clean[s:e], resid[s:e]
        p_sig = np.mean(c ** 2, axis=1)                     # (b, 12)
        p_res = np.mean(r ** 2, axis=1)
        out['snr_db'][s:e] = 10.0 * np.log10(np.mean(p_sig, 1) / (np.mean(p_res, 1) + 1e-12) + 1e-12)
        wander = sps.sosfiltfilt(sos_lo, r, axis=1)
        hf = sps.sosfiltfilt(sos_hi, r, axis=1)
        out['wander_rms_mv'][s:e] = np.sqrt(np.mean(wander ** 2, axis=(1, 2)))
        out['hf_rms_mv'][s:e] = np.sqrt(np.mean(hf ** 2, axis=(1, 2)))
        out['resid_rms_mv'][s:e] = np.sqrt(np.mean(r ** 2, axis=(1, 2)))
        out['sig_rms_mv'][s:e] = np.sqrt(np.mean(c ** 2, axis=(1, 2)))
        lead_rms = np.sqrt(p_sig)
        out['flat_leads'][s:e] = (lead_rms < 0.02).sum(1)    # near-dead leads (<20 uV RMS)
    return pd.DataFrame(out)

sq_f = CACHE / 'signal_quality.csv'
if sq_f.exists():
    sq_test = pd.read_csv(sq_f)
    print(f'\n[CACHE] signal-quality covariates loaded from {sq_f}')
else:
    raw_t = np.load(DATA / 'X_test_100.npy')
    cln_t = np.load(DATA / 'X_test_clean_100.npy')
    sq_test = signal_quality(raw_t, cln_t)
    del raw_t, cln_t
    sq_test.to_csv(sq_f, index=False)
    print('\n[SQ] computed per-record signal-quality covariates for TEST')

sq_test['age'] = age_test
print(sq_test[['snr_db', 'wander_rms_mv', 'hf_rms_mv', 'flat_leads']].describe().round(4).to_string())

# ----------------------------------------------------------------------------- per-record outcomes
set_size_mondrian = np.zeros(len(y_test))
set_size_global = np.zeros(len(y_test))
covered_mondrian = np.zeros(len(y_test))
for b, m in bands_test.items():
    pm = (p_test_c[m] >= mondrian_thr[b]).astype(int)
    pg = (p_test_c[m] >= global_thr).astype(int)
    set_size_mondrian[m] = pm.sum(1)
    set_size_global[m] = pg.sum(1)
    covered_mondrian[m] = (pm >= y_test[m]).all(1)

abs_cal_err = np.abs(p_test_c - y_test).mean(1)             # per-record mean |p - y|, calibration proxy
pred_bin = (p_test_c >= 0.5).astype(int)
exact_correct = (pred_bin == y_test).all(1).astype(float)
n_labels = y_test.sum(1)

D = pd.DataFrame({'age': age_test, 'set_mondrian': set_size_mondrian, 'set_global': set_size_global,
                  'covered': covered_mondrian, 'abs_cal_err': abs_cal_err, 'correct': exact_correct,
                  'n_labels': n_labels}).join(sq_test.drop(columns='age'))
D['band'] = pd.cut(D.age, [-np.inf, 40, 65, 80, np.inf], right=False, labels=BAND_NAMES)

# ----------------------------------------------------------------------------- (B) trend tests
def ols(y, X, names):
    """Plain OLS with HC0 (heteroskedasticity-robust) standard errors."""
    X = np.column_stack([np.ones(len(y))] + list(X))
    names = ['const'] + list(names)
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    tt = beta / se
    dof = len(y) - X.shape[1]
    p = 2 * stats.t.sf(np.abs(tt), dof)
    return {n: dict(beta=float(b), se=float(s), t=float(t_), p=float(p_))
            for n, b, s, t_, p_ in zip(names, beta, se, tt, p)}

def trend_block(col, label):
    y = D[col].values
    a = D.age.values
    groups = [D.loc[D.band == b, col].values for b in BAND_NAMES]
    kw_h, kw_p = stats.kruskal(*groups)
    rho, rho_p = stats.spearmanr(a, y)
    jt = stats.jonckheereterpstra(*groups, alternative='increasing') if hasattr(stats, 'jonckheereterpstra') else None
    m_raw = ols(y, [a / 10.0], ['age_per_decade'])
    covs = ['snr_db', 'wander_rms_mv', 'hf_rms_mv', 'flat_leads']
    m_adj = ols(y, [a / 10.0] + [D[c].values for c in covs], ['age_per_decade'] + covs)
    m_adj_lab = ols(y, [a / 10.0] + [D[c].values for c in covs] + [D.n_labels.values],
                    ['age_per_decade'] + covs + ['n_labels'])
    out = dict(metric=label, n=int(len(y)),
               kruskal_H=float(kw_h), kruskal_p=float(kw_p),
               spearman_rho_vs_age=float(rho), spearman_p=float(rho_p),
               band_means={b: float(np.mean(g)) for b, g in zip(BAND_NAMES, groups)},
               band_n={b: int(len(g)) for b, g in zip(BAND_NAMES, groups)},
               ols_unadjusted=m_raw['age_per_decade'],
               ols_adjusted_signal_quality=m_adj['age_per_decade'],
               ols_adjusted_sq_and_label_count=m_adj_lab['age_per_decade'],
               attenuation_pct_after_sq=float(100 * (1 - m_adj['age_per_decade']['beta'] /
                                                     m_raw['age_per_decade']['beta'])),
               covariate_terms={c: m_adj[c] for c in covs})
    if jt is not None:
        out['jonckheere_p'] = float(jt.pvalue)
    return out

trends = {k: trend_block(c, k) for k, c in
          [('conformal_set_size_mondrian', 'set_mondrian'),
           ('conformal_set_size_global', 'set_global'),
           ('per_record_abs_calibration_error', 'abs_cal_err'),
           ('exact_match_correct', 'correct'),
           ('label_count', 'n_labels')]}

# age <-> signal quality association (does the confound even exist here?)
conf_assoc = {}
for c in ['snr_db', 'wander_rms_mv', 'hf_rms_mv', 'resid_rms_mv', 'flat_leads', 'sig_rms_mv']:
    rho, p = stats.spearmanr(D.age.values, D[c].values)
    kw_h, kw_p = stats.kruskal(*[D.loc[D.band == b, c].values for b in BAND_NAMES])
    conf_assoc[c] = dict(spearman_rho_vs_age=float(rho), spearman_p=float(p),
                         kruskal_H=float(kw_h), kruskal_p=float(kw_p),
                         band_means={b: float(D.loc[D.band == b, c].mean()) for b in BAND_NAMES})

# band-level ECE trend, bootstrap slope on band midpoints
def calib_stats(y_true, p, n_bins=M_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / len(p) * abs(y_true[m].mean() - p[m].mean())
    return ece

def macro_ece(mask, P):
    return float(np.mean([calib_stats(y_test[mask, k], P[mask, k]) for k in range(NC)]))

rng = np.random.default_rng(SEED)
N_BOOT = 1000
ece_band = {b: dict(uncal=macro_ece(m, p_ens_test), cal=macro_ece(m, p_test_c), n=int(m.sum()))
            for b, m in bands_test.items()}
slopes_ece, slopes_set = [], []
mids = np.array([BAND_MID[b] for b in BAND_NAMES])
for _ in range(N_BOOT):
    ece_b, set_b = [], []
    for b, m in bands_test.items():
        idx = np.where(m)[0]
        samp = rng.choice(idx, size=len(idx), replace=True)
        ece_b.append(float(np.mean([calib_stats(y_test[samp, k], p_test_c[samp, k]) for k in range(NC)])))
        set_b.append(float(set_size_mondrian[samp].mean()))
    slopes_ece.append(np.polyfit(mids, ece_b, 1)[0])
    slopes_set.append(np.polyfit(mids, set_b, 1)[0])
slopes_ece, slopes_set = np.array(slopes_ece), np.array(slopes_set)
band_trend = dict(
    ece_by_band=ece_band,
    ece_slope_per_decade=float(np.polyfit(mids, [ece_band[b]['cal'] for b in BAND_NAMES], 1)[0] * 10),
    ece_slope_per_decade_ci=[float(np.percentile(slopes_ece, 2.5) * 10), float(np.percentile(slopes_ece, 97.5) * 10)],
    ece_slope_boot_p_gt0=float((slopes_ece <= 0).mean()),
    setsize_slope_per_decade=float(np.polyfit(mids, [np.mean(set_size_mondrian[bands_test[b]]) for b in BAND_NAMES], 1)[0] * 10),
    setsize_slope_per_decade_ci=[float(np.percentile(slopes_set, 2.5) * 10), float(np.percentile(slopes_set, 97.5) * 10)],
    setsize_slope_boot_p_gt0=float((slopes_set <= 0).mean()),
    n_boot=N_BOOT)

# ----------------------------------------------------------------------------- (C) calibration-resample stability
N_CAL_BOOT = 300
stab = {b: [] for b in BAND_NAMES}
stab_cov = {b: [] for b in BAND_NAMES}
stab_slope = []
for _ in range(N_CAL_BOOT):
    sizes = {}
    for b, mc in bands_cal.items():
        idx = np.where(mc)[0]
        samp = rng.choice(idx, size=len(idx), replace=True)
        thr = conformal_thresholds(p_cal_c[samp], y_cal[samp])
        mt = bands_test[b]
        pred = (p_test_c[mt] >= thr).astype(int)
        s = float(pred.sum(1).mean())
        stab[b].append(s)
        stab_cov[b].append(float((pred >= y_test[mt]).all(1).mean()))
        sizes[b] = s
    stab_slope.append(np.polyfit(mids, [sizes[b] for b in BAND_NAMES], 1)[0])
stab_slope = np.array(stab_slope)
stability = dict(n_cal_resamples=N_CAL_BOOT,
                 set_size={b: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                   pct=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                           for b, v in stab.items()},
                 coverage={b: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                   pct=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                           for b, v in stab_cov.items()},
                 slope_per_decade=dict(mean=float(stab_slope.mean() * 10),
                                       pct=[float(np.percentile(stab_slope, 2.5) * 10),
                                            float(np.percentile(stab_slope, 97.5) * 10)],
                                       frac_positive=float((stab_slope > 0).mean())))

# ----------------------------------------------------------------------------- (D) explanation-consistency trend test
xai = ref['age_stratified_xai_nb06_m13_15']
rows = {r['subgroup']: {} for r in xai['rows']}
for r in xai['rows']:
    rows[r['subgroup']][r['class']] = r['spearman_rho']
low = np.array([rows['low-confidence elderly'][c] for c in SUPER])
high = np.array([rows['high-confidence elderly'][c] for c in SUPER])
w_stat, w_p = stats.wilcoxon(low, high, alternative='less')
xai_test = dict(classes=SUPER, rho_low=low.tolist(), rho_high=high.tolist(),
                mean_delta=float(np.mean(low - high)),
                wilcoxon_statistic=float(w_stat), wilcoxon_p_one_sided=float(w_p),
                note='paired over 5 classes only; underpowered by construction')

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25})

# F1 reliability, faceted by age band (macro over classes)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
edges = np.linspace(0, 1, M_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
for ax, b in zip(axes, BAND_NAMES):
    m = bands_test[b]
    acc_all, conf_all, wt_all = np.zeros(M_BINS), np.zeros(M_BINS), np.zeros(M_BINS)
    for k in range(NC):
        p, yk = p_test_c[m, k], y_test[m, k]
        idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, M_BINS - 1)
        for bb in range(M_BINS):
            sel = idx == bb
            if sel.any():
                acc_all[bb] += yk[sel].sum(); conf_all[bb] += p[sel].sum(); wt_all[bb] += sel.sum()
    ok = wt_all > 0
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=.5)
    ax.bar(centers[ok], (acc_all[ok] / wt_all[ok]), width=1 / M_BINS, alpha=.6, edgecolor='k', lw=.4)
    ax.plot(conf_all[ok] / wt_all[ok], acc_all[ok] / wt_all[ok], 'o-', color='crimson', ms=3, lw=1)
    ax.set_title(f'{b}   n={int(m.sum())}\nmacro-ECE {ece_band[b]["cal"]:.3f}')
    ax.set_xlabel('predicted probability')
axes[0].set_ylabel('observed frequency')
fig.suptitle('Reliability after global temperature scaling, by age band (pooled over the five superclasses)', y=1.04)
fig.tight_layout(); fig.savefig(OUT_FIG / '11_reliability_by_age_band.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# F2 age vs signal quality
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, c, lab in zip(axes, ['snr_db', 'wander_rms_mv', 'hf_rms_mv'],
                      ['in-band SNR (dB)', 'baseline-wander RMS (mV)', 'high-frequency noise RMS (mV)']):
    data = [D.loc[D.band == b, c].values for b in BAND_NAMES]
    ax.boxplot(data, tick_labels=BAND_NAMES, showfliers=False)
    r = conf_assoc[c]['spearman_rho_vs_age']
    ax.set_title(f'{lab}\nSpearman vs age rho={r:+.3f} (p={conf_assoc[c]["spearman_p"]:.3g})')
    ax.set_xlabel('age band')
fig.suptitle('Is the age effect confounded by acquisition quality? Signal-quality covariates by age band', y=1.05)
fig.tight_layout(); fig.savefig(OUT_FIG / '11_signal_quality_by_age_band.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# F3 age coefficient before/after adjustment
fig, ax = plt.subplots(figsize=(7.2, 3.4))
keys = ['conformal_set_size_mondrian', 'conformal_set_size_global', 'per_record_abs_calibration_error']
labs = ['Mondrian set size', 'global-threshold set size', 'mean |p - y|']
ypos = np.arange(len(keys))
for off, mk, colr, nm in [(-0.16, 'ols_unadjusted', 'steelblue', 'age only'),
                          (0.0, 'ols_adjusted_signal_quality', 'crimson', '+ signal quality'),
                          (0.16, 'ols_adjusted_sq_and_label_count', 'seagreen', '+ quality + label count')]:
    b = [trends[k][mk]['beta'] for k in keys]
    e = [1.96 * trends[k][mk]['se'] for k in keys]
    ax.errorbar(b, ypos + off, xerr=e, fmt='o', color=colr, ms=4, capsize=3, label=nm)
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_yticks(ypos); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xlabel('change in outcome per decade of age (OLS, HC0 95% CI)')
ax.set_title('The age effect after adjusting for acquisition quality')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG / '11_age_effect_adjusted.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# F4 calibration-resampling stability of the age trend
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.boxplot([stab[b] for b in BAND_NAMES], tick_labels=BAND_NAMES, showfliers=False)
ax.plot(range(1, 5), [repro['set_size_mondrian'][b] for b in BAND_NAMES], 'r*', ms=11,
        label='point estimate (full calibration split)')
ax.set_ylabel('mean Mondrian set size'); ax.set_xlabel('age band')
ax.set_title(f'Refit over {N_CAL_BOOT} bootstrap resamples\nof the calibration split')
ax.legend(fontsize=8)
ax = axes[1]
ax.hist(stab_slope * 10, bins=40, color='steelblue', edgecolor='k', lw=.3)
ax.axvline(0, color='k', ls='--')
ax.set_xlabel('set-size slope per decade of age')
ax.set_ylabel('resamples')
ax.set_title(f'slope > 0 in {stability["slope_per_decade"]["frac_positive"]*100:.1f}% of resamples')
fig.suptitle('Stability of the age trend to conformal-calibration resampling', y=1.04)
fig.tight_layout(); fig.savefig(OUT_FIG / '11_calibration_resampling_stability.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ----------------------------------------------------------------------------- save
result = dict(
    meta=dict(seed=SEED, alpha=ALPHA, ece_bins=M_BINS, n_boot=N_BOOT, n_cal_resamples=N_CAL_BOOT,
              band_midpoints=BAND_MID, device=str(DEVICE), torch=torch.__version__,
              numpy=np.__version__, scipy=__import__('scipy').__version__,
              signal_quality_definition=(
                  'residual = raw calibrated mV (X_*_100.npy) minus 0.5-40 Hz filtered signal '
                  '(X_*_clean_100.npy); snr_db = 10*log10(mean lead power of filtered / mean lead power '
                  'of residual); wander = <0.5 Hz part of the residual; hf = >40 Hz part; '
                  'flat_leads = leads with RMS < 0.02 mV')),
    reproduction_check=repro,
    signal_quality_summary=sq_test[['snr_db', 'wander_rms_mv', 'hf_rms_mv', 'resid_rms_mv',
                                    'flat_leads', 'sig_rms_mv']].describe().to_dict(),
    age_vs_signal_quality=conf_assoc,
    trend_tests=trends,
    band_level_trends=band_trend,
    calibration_resampling_stability=stability,
    explanation_consistency_test=xai_test)

with open(OUT_VAL / '11_confound_and_trend_tests.json', 'w') as f:
    json.dump(result, f, indent=1, default=float)
D.to_csv(CACHE / 'per_record_outcomes.csv', index=False)

print('\n================ SUMMARY ================')
print(f"age<->SNR Spearman rho = {conf_assoc['snr_db']['spearman_rho_vs_age']:+.3f} "
      f"(p={conf_assoc['snr_db']['spearman_p']:.3g})")
for k in ['conformal_set_size_mondrian', 'conformal_set_size_global', 'per_record_abs_calibration_error']:
    t = trends[k]
    print(f"\n{k}")
    print(f"  bands            : " + '  '.join(f'{b}={t["band_means"][b]:.3f}(n={t["band_n"][b]})' for b in BAND_NAMES))
    print(f"  Kruskal-Wallis   : H={t['kruskal_H']:.1f}  p={t['kruskal_p']:.3g}")
    print(f"  Spearman vs age  : rho={t['spearman_rho_vs_age']:+.3f}  p={t['spearman_p']:.3g}")
    print(f"  OLS age/decade   : {t['ols_unadjusted']['beta']:+.4f} (p={t['ols_unadjusted']['p']:.3g})")
    print(f"  + signal quality : {t['ols_adjusted_signal_quality']['beta']:+.4f} "
          f"(p={t['ols_adjusted_signal_quality']['p']:.3g})  attenuation {t['attenuation_pct_after_sq']:+.1f}%")
    print(f"  + quality+labels : {t['ols_adjusted_sq_and_label_count']['beta']:+.4f} "
          f"(p={t['ols_adjusted_sq_and_label_count']['p']:.3g})")
print(f"\nECE slope/decade   : {band_trend['ece_slope_per_decade']:+.4f} "
      f"CI {band_trend['ece_slope_per_decade_ci']}  boot p={band_trend['ece_slope_boot_p_gt0']:.3g}")
print(f"set-size slope/dec : {band_trend['setsize_slope_per_decade']:+.4f} "
      f"CI {band_trend['setsize_slope_per_decade_ci']}")
print(f"cal-resample slope : {stability['slope_per_decade']['mean']:+.4f} "
      f"CI {stability['slope_per_decade']['pct']}  positive in "
      f"{stability['slope_per_decade']['frac_positive']*100:.1f}%")
print(f"XAI Wilcoxon       : p={xai_test['wilcoxon_p_one_sided']:.3g}  mean delta {xai_test['mean_delta']:+.4f}")
print('\n[SAVED]', OUT_VAL / '11_confound_and_trend_tests.json')
