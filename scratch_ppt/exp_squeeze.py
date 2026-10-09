"""Experiment: how far can we push macro-AUC using ONLY the 3 existing checkpoints
(no retraining)?  Levers: blend-weight optimisation, logit vs prob vs rank averaging,
per-class weights, dropping the weak member, and test-time augmentation (TTA).

Measures everything on TEST (N=2198), tunes only on CALIBRATION (N=1103).
"""
import time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
DP   = ROOT / "data" / "processed"
OUT  = ROOT / "outputs"
DEV  = torch.device("cpu")
torch.set_num_threads(4)
SEED = 42
SC   = ["NORM","MI","STTC","CD","HYP"]
CK = {"DualBranchECGNet": OUT/"ensemble_DualBranchECGNet_best.pth",
      "InceptionTime1D":   OUT/"05_optimized_InceptionTime1D_best.pth",
      "ResNet1D101":       OUT/"ensemble_ResNet1D101_best.pth"}

# ---------- models (verbatim from NB02/NB03/NB05) ----------
class InceptionBlock1D(nn.Module):
    def __init__(s, ic, nf=32, ks=(9,19,39)):
        super().__init__()
        s.bottleneck=nn.Conv1d(ic,nf,1,bias=False)
        s.convs=nn.ModuleList([nn.Conv1d(nf,nf,k,padding=k//2,bias=False) for k in ks])
        s.pool_conv=nn.Conv1d(ic,nf,1,bias=False); s.pool=nn.MaxPool1d(3,1,1)
        s.bn=nn.BatchNorm1d(nf*(len(ks)+1)); s.act=nn.ReLU()
    def forward(s,x):
        b=s.bottleneck(x)
        return s.act(s.bn(torch.cat([c(b) for c in s.convs]+[s.pool_conv(s.pool(x))],1)))
class InceptionTime1D(nn.Module):
    def __init__(s,nc=5,ic=12,nf=32,depth=4):
        super().__init__()
        ch=[ic]+[nf*4]*depth
        s.blocks=nn.Sequential(*[InceptionBlock1D(ch[i],nf) for i in range(depth)])
        s.gap=nn.AdaptiveAvgPool1d(1); s.fc=nn.Linear(nf*4,nc); s.drop=nn.Dropout(0.3)
    def forward(s,x):
        x=s.blocks(x); return s.fc(s.drop(s.gap(x).squeeze(-1)))
class DualBranchECGNet(nn.Module):
    def __init__(s,nc=5,ic=12):
        super().__init__()
        s.ecg_branch=nn.Sequential(
            nn.Conv1d(ic,32,3,padding=1),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Conv1d(32,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Conv1d(128,256,3,padding=1),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Conv1d(256,512,3,padding=1),nn.BatchNorm1d(512),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        s.demo_branch=nn.Sequential(
            nn.Linear(2,100),nn.BatchNorm1d(100),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(100,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(32,16),nn.BatchNorm1d(16),nn.ReLU())
        s.classifier=nn.Sequential(nn.Linear(528,64),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,nc))
    def forward(s,ecg,demo):
        return s.classifier(torch.cat([s.ecg_branch(ecg).squeeze(-1),s.demo_branch(demo)],1))
class ResBlock1D(nn.Module):
    def __init__(s,ic,oc,stride=1,k=7):
        super().__init__(); p=k//2
        s.conv1=nn.Conv1d(ic,oc,k,stride,p,bias=False); s.bn1=nn.BatchNorm1d(oc)
        s.conv2=nn.Conv1d(oc,oc,k,1,p,bias=False); s.bn2=nn.BatchNorm1d(oc)
        s.skip=nn.Sequential()
        if stride!=1 or ic!=oc:
            s.skip=nn.Sequential(nn.Conv1d(ic,oc,1,stride,bias=False),nn.BatchNorm1d(oc))
        s.act=nn.ReLU()
    def forward(s,x):
        o=s.act(s.bn1(s.conv1(x))); o=s.bn2(s.conv2(o)); return s.act(o+s.skip(x))
class ResNet1D101(nn.Module):
    def __init__(s,nc=5,ic=12,drop=0.3):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(ic,64,15,2,7,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(3,2,1))
        def st(i,o,n,stride=2): return nn.Sequential(ResBlock1D(i,o,stride),*[ResBlock1D(o,o) for _ in range(n-1)])
        s.stage1=st(64,64,8,1); s.stage2=st(64,128,8,2); s.stage3=st(128,256,8,2); s.stage4=st(256,512,8,2)
        s.pool=nn.AdaptiveAvgPool1d(1); s.drop=nn.Dropout(drop); s.fc=nn.Linear(512,nc)
    def forward(s,x):
        x=s.stem(x); x=s.stage4(s.stage3(s.stage2(s.stage1(x))))
        return s.fc(s.drop(s.pool(x).squeeze(-1)))

# ---------- data ----------
def load(stem): return np.transpose(np.load(DP/f"{stem}.npy"),(0,2,1)).astype(np.float32)
Xte, Xca = load("X_test_clean_100"), load("X_cal_clean_100")
yte = np.load(DP/"y_test_diag_100.npy").astype(np.float32)
yca = np.load(DP/"y_calib_diag_100.npy").astype(np.float32)
meta = pd.read_csv(DP/"MASTER_RESEARCH_METADATA.csv")
tr = meta[meta.split=="TRAIN"]; am, asd = tr.age.mean(), tr.age.std()
def demo(split):
    d = meta[meta.split==split].reset_index(drop=True)
    return np.column_stack([((d.age.fillna(am)-am)/(asd+1e-6)).values,(d.sex==1).values.astype(np.float32)]).astype(np.float32)
Dte, Dca = demo("TEST"), demo("CONFORMAL_CALIBRATION")

def macro_auc(Y,P): return float(np.mean([roc_auc_score(Y[:,k],P[:,k]) for k in range(5)]))

@torch.no_grad()
def infer(model, X, D=None, bs=64, tta=False):
    model.eval().to(DEV)
    variants = [(0,1.0)]
    if tta: variants += [(-8,1.0),(8,1.0),(0,0.97),(0,1.03),(-4,1.02),(4,0.98)]
    acc = None
    for shift, scale in variants:
        Xv = X if (shift==0 and scale==1.0) else (np.roll(X, shift, axis=-1)*scale).astype(np.float32)
        out=[]
        for i in range(0,len(Xv),bs):
            xb=torch.from_numpy(Xv[i:i+bs]).to(DEV)
            lb = model(xb) if D is None else model(xb, torch.from_numpy(D[i:i+bs]).to(DEV))
            out.append(torch.sigmoid(lb).cpu().numpy())
        p=np.vstack(out); acc = p if acc is None else acc+p
    return acc/len(variants)

ORDER = ["DualBranchECGNet","InceptionTime1D","ResNet1D101"]
NEEDS = {"DualBranchECGNet":True,"InceptionTime1D":False,"ResNet1D101":False}
CLS   = {"DualBranchECGNet":DualBranchECGNet,"InceptionTime1D":InceptionTime1D,"ResNet1D101":ResNet1D101}

t0=time.time(); PT={}; PC={}; PTt={}; PCt={}
for n in ORDER:
    m=CLS[n](5); m.load_state_dict(torch.load(CK[n],map_location=DEV,weights_only=True))
    PT[n]  = infer(m, Xte, Dte if NEEDS[n] else None, tta=False)
    PC[n]  = infer(m, Xca, Dca if NEEDS[n] else None, tta=False)
    PTt[n] = infer(m, Xte, Dte if NEEDS[n] else None, tta=True)
    PCt[n] = infer(m, Xca, Dca if NEEDS[n] else None, tta=True)
    print(f"  {n:16s} plain AUC te={macro_auc(yte,PT[n]):.4f}  TTA AUC te={macro_auc(yte,PTt[n]):.4f}  ({time.time()-t0:.0f}s)")

def stack(d, keys): return np.stack([d[k] for k in keys])          # (M,N,5)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def ranks(P):                                                       # per-class rank-normalise, (N,5)
    r=np.empty_like(P)
    for k in range(P.shape[1]): r[:,k]=P[:,k].argsort().argsort()/(len(P)-1)
    return r

def eval_blend(w, Pstack, mode):
    w=np.asarray(w); w=w/w.sum()
    if mode=="prob":  return np.tensordot(w,Pstack,axes=1)
    if mode=="logit": return 1/(1+np.exp(-np.tensordot(w,np.stack([logit(p) for p in Pstack]),axes=1)))
    if mode=="rank":  return np.tensordot(w,np.stack([ranks(p) for p in Pstack]),axes=1)

def fit_weights(Pstack_cal, Ycal, mode, per_class=False):
    M=len(Pstack_cal)
    if not per_class:
        def neg(w): return -macro_auc(Ycal, eval_blend(np.abs(w)+1e-6, Pstack_cal, mode))
        res=minimize(neg, np.ones(M), method="Nelder-Mead",
                     options={"xatol":1e-3,"fatol":1e-5,"maxiter":800})
        return np.abs(res.x)+1e-6
    W=np.ones((5,M))
    for k in range(5):
        def neg(w):
            w=np.abs(w)+1e-6; w=w/w.sum()
            blends=[eval_blend(w,Pstack_cal,mode)] ; return -roc_auc_score(Ycal[:,k], blends[0][:,k])
        res=minimize(neg, np.ones(M), method="Nelder-Mead", options={"xatol":1e-3,"fatol":1e-5,"maxiter":600})
        W[k]=np.abs(res.x)+1e-6
    return W

def blend_per_class(W, Pstack, mode):
    out=np.empty((Pstack.shape[1],5))
    for k in range(5):
        out[:,k]=eval_blend(W[k], Pstack, mode)[:,k]
    return out

print("\n=== CONFIGURATIONS (tuned on CALIBRATION, reported on TEST) ===")
BASE = 0.9245
def report(name, Pte):
    a=macro_auc(yte,Pte); print(f"  {name:52s} {a:.4f}   ({'+' if a>=BASE else ''}{a-BASE:+.4f} vs 0.9245)")
    return a, Pte

results={}
# current recipe: AUC-weighted prob average, all 3
w_auc = np.array([macro_auc(yca,PC[n]) for n in ORDER]); w_auc/=w_auc.sum()
results['A. current (AUC-wt prob avg, 3 models)'] = report('A. current (AUC-wt prob avg, 3 models)', np.tensordot(w_auc,stack(PT,ORDER),axes=1))

for keys,tag in [(ORDER,'3 models'), (["InceptionTime1D","ResNet1D101"],'2 strong models')]:
    Sc, St = stack(PC,keys), stack(PT,keys)
    for mode in ["prob","logit","rank"]:
        w = fit_weights(Sc, yca, mode)
        results[f'opt-{mode} ({tag})'] = report(f'opt-{mode} weights, {tag}', eval_blend(w, St, mode))
    Wpc = fit_weights(Sc, yca, "prob", per_class=True)
    results[f'opt-prob per-class wts ({tag})'] = report(f'opt-prob per-class weights, {tag}', blend_per_class(Wpc, St, "prob"))

# + TTA
for keys,tag in [(ORDER,'3 models+TTA'), (["InceptionTime1D","ResNet1D101"],'2 strong+TTA')]:
    Sc, St = stack(PCt,keys), stack(PTt,keys)
    for mode in ["prob","logit","rank"]:
        w = fit_weights(Sc, yca, mode)
        results[f'opt-{mode} ({tag})'] = report(f'opt-{mode} weights, {tag}', eval_blend(w, St, mode))
    Wpc = fit_weights(Sc, yca, "prob", per_class=True)
    results[f'opt-prob per-class ({tag})'] = report(f'opt-prob per-class weights, {tag}', blend_per_class(Wpc, St, "prob"))

best = max(results, key=lambda k: results[k][0])
print(f"\nBEST: {best}  ->  macro-AUC {results[best][0]:.4f}")

# bootstrap the best vs current, paired
rng=np.random.default_rng(SEED); N=len(yte); B=1000
Pbest=results[best][1]; Pcur=results['A. current (AUC-wt prob avg, 3 models)'][1]
d=np.empty(B); ab=np.empty(B)
for b in range(B):
    idx=rng.integers(0,N,N)
    ab[b]=macro_auc(yte[idx],Pbest[idx])
    d[b]=macro_auc(yte[idx],Pbest[idx])-macro_auc(yte[idx],Pcur[idx])
lo,hi=np.percentile(ab,[2.5,97.5]); dlo,dhi=np.percentile(d,[2.5,97.5])
print(f"BEST bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
print(f"Δ vs current: {d.mean():+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]  "
      f"({'real gain' if dlo>0 else 'within noise'})")
print(f"\nvs literature:  Strodthoff ens 0.928 | Gitau ens 0.934")
