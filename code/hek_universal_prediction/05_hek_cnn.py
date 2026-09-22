#!/usr/bin/env python
"""
Step 5 (CNN): HEK293 PWM-initialized multi-branch CNN, architecture and protocol
IDENTICAL to the EpiMap comparison (posthoc_analysis/05_cnn_comparison.py), so the
HEK number is directly comparable to the 116-line universal CNN distribution.

Difference from posthoc 05: it reads the HEK promoter one-hot + HPA target built by
05a_build_cnn_inputs.py instead of a per-BSSID EpiMap array. Everything downstream
(conv0 PWM init, 10-branch model, 90/10 split + fold-0 val, Adam/MSE/L2, early stop,
Spearman) is unchanged.

Run on GPU (see 05_hek_cnn.sh):
  python 05_hek_cnn.py --motif_set universal      # 10 promoter branches
  python 05_hek_cnn.py --motif_set control        # single-branch, JASPAR control PWMs

Outputs: results/cnn/hek_{motif_set}_{train,val,test}_S.npy , _model.pth
"""
import os, argparse, numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
import torch, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--motif_set", choices=["universal", "control"], default="universal")
ap.add_argument("--reg_scale", type=float, default=1.0,
                help="multiplier on per-branch L2 (REG1/2/3). 1.0=EpiMap-comparable baseline; "
                     "4.0=tuned config (see results/cnn/SWEEP_SUMMARY.txt).")
ap.add_argument("--dropout", type=float, default=0.15)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--patience", type=int, default=10); ap.add_argument("--epochs", type=int, default=100)   # published HEK293 model: --dropout 0.3 --reg_scale 2.0 --lr 5e-5 --patience 15 --epochs 200
args = ap.parse_args()

UNIVERSAL_MEME = "/stg3/data1/sam/enhancer_prediction/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme"
CONTROL_MEME   = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/results/control_motifs.meme"
ONEHOT = "results/hek_one_hot.npy"     # (18356, 10, 400, 4) bool
YFILE  = "results/hek_y.npy"           # (18356,) log2(HPA TPM+1), NaN if missing
OUT_DIR = "results/cnn"
REF = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/results/cnn"  # optional

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── hyperparameters (same as posthoc 05 / CNN_CR.py) ──
PRO_ONLY = True
NUM_BRANCHES = 10
SEQ_LEN = 400
UNIV_K = 8
CONV0_OUT_LEN = SEQ_LEN - UNIV_K + 1
BATCH, EPOCHS, LR, PATIENCE = 64, args.epochs, args.lr, args.patience
CONV1_FILTER = CONV2_FILTER = CONV3_FILTER = 64
MAXPOOL0_SIZE, MAXPOOL1_SIZE, MAXPOOL2_SIZE, MAXPOOL3_SIZE = 5, 2, 3, 4
REG1, REG2, REG3 = 0.1 * args.reg_scale, 0.2 * args.reg_scale, 0.3 * args.reg_scale
DROPOUT1, DROPOUT2 = args.dropout, args.dropout


def parse_meme(path):
    motifs, names, cur, cur_name, reading = [], [], [], None, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                if reading and cur:
                    motifs.append(np.asarray(cur, np.float32)); names.append(cur_name); cur = []
                reading = False; continue
            if s.startswith("MOTIF"):
                if reading and cur:
                    motifs.append(np.asarray(cur, np.float32)); names.append(cur_name); cur = []
                parts = s.split(); cur_name = parts[1] if len(parts) >= 2 else f"m{len(motifs)}"
                reading = False
            elif s.startswith("letter-probability matrix"):
                reading = True; cur = []
            elif reading:
                parts = s.split()
                if len(parts) >= 4:
                    try: cur.append([float(p) for p in parts[:4]])
                    except ValueError: reading = False
    if reading and cur:
        motifs.append(np.asarray(cur, np.float32)); names.append(cur_name)
    return motifs, names


class UniversalMotifConv(nn.Module):
    def __init__(self, motifs):
        super().__init__()
        n = len(motifs)
        kernel = np.zeros((n, 4, UNIV_K), dtype=np.float32)
        for i, m in enumerate(motifs):
            kernel[i, :, :] = (m - 0.25).T
        self.conv = nn.Conv1d(4, n, kernel_size=UNIV_K, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(kernel))
    def forward(self, x): return self.conv(x)


class ControlMotifConv(nn.Module):
    def __init__(self, motifs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pool = nn.AdaptiveMaxPool1d(CONV0_OUT_LEN)
        for m in motifs:
            conv = nn.Conv1d(4, 1, kernel_size=m.shape[0], bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy((m - 0.25).T).unsqueeze(0))
            self.convs.append(conv)
    def forward(self, x):
        return torch.cat([self.pool(c(x)) for c in self.convs], dim=1)


class BiasLayer(nn.Module):
    def __init__(self, n):
        super().__init__(); self.weight = Parameter(torch.ones(n))
    def forward(self, x): return x * self.weight


class CNNBranch(nn.Module):
    def __init__(self, motifs, num_motifs, motif_set):
        super().__init__()
        self.conv0 = UniversalMotifConv(motifs) if motif_set == "universal" else ControlMotifConv(motifs)
        self.maxpool0 = nn.MaxPool1d(MAXPOOL0_SIZE)
        self.conv1 = nn.Conv1d(num_motifs, CONV1_FILTER, 8, padding="same")
        self.maxpool1 = nn.MaxPool1d(MAXPOOL1_SIZE)
        self.conv2 = nn.Conv1d(CONV1_FILTER, CONV2_FILTER, 8, padding="same")
        self.maxpool2 = nn.MaxPool1d(MAXPOOL2_SIZE)
        self.conv3 = nn.Conv1d(CONV2_FILTER, CONV3_FILTER, 8, padding="same")
        self.maxpool3 = nn.MaxPool1d(MAXPOOL3_SIZE)
        self.flatten = nn.Flatten()
        for c in (self.conv1, self.conv2, self.conv3):
            init.xavier_normal_(c.weight)
        with torch.no_grad():
            d = torch.zeros(1, 4, SEQ_LEN)
            d = self.maxpool0(F.relu(self.conv0(d)))
            d = self.maxpool1(self.conv1(d)); d = self.maxpool2(self.conv2(d)); d = self.maxpool3(self.conv3(d))
            flat_size = self.flatten(d).shape[1]
        self.dense1 = nn.Linear(flat_size, num_motifs)
        self.dropout1 = nn.Dropout(DROPOUT1)
        self.dense2 = nn.Linear(num_motifs, num_motifs)
        self.dropout2 = nn.Dropout(DROPOUT2)
        self.contextual_weight = nn.Linear(num_motifs, num_motifs)
        self.bias_layer = BiasLayer(1)
    def forward(self, x):
        is_nonzero = torch.any(x[:, 0, :] != 0, dim=1).unsqueeze(-1).float()
        x = x.permute(0, 2, 1)
        x = self.conv0(x)
        mean_act = torch.mean(x, dim=2)
        x = self.maxpool0(F.relu(x))
        x = self.maxpool1(self.conv1(x)); x = self.maxpool2(self.conv2(x)); x = self.maxpool3(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.dense1(x)))
        x = self.dropout2(F.relu(self.dense2(x)))
        ctx = self.contextual_weight(x) * is_nonzero
        combined = ctx * mean_act
        return self.bias_layer(combined.sum(dim=1, keepdim=True)), ctx, combined


class MultiBranchModel(nn.Module):
    def __init__(self, motifs, num_motifs, motif_set, n_branches=NUM_BRANCHES):
        super().__init__()
        self.branches = nn.ModuleList([CNNBranch(motifs, num_motifs, motif_set) for _ in range(n_branches)])
    def forward(self, x):
        x = x.float()
        out = torch.zeros(x.size(0), len(self.branches), device=x.device)
        for i, br in enumerate(self.branches):
            bo, _, _ = br(x[:, i, :, :])
            out[:, i] = bo.squeeze()
        return out.sum(dim=1, keepdim=True), None, None


class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience = patience; self.best = np.inf; self.count = 0; self.best_epoch = -1
    def __call__(self, v, epoch):
        if v < self.best: self.best = v; self.count = 0; self.best_epoch = epoch
        else: self.count += 1
        return self.count >= self.patience


def l2_reg(params, s): return s * sum(torch.sum(p ** 2) for p in params)

def train_model(model, tr_loader, val_loader, device):
    opt = optim.Adam(model.parameters(), lr=LR); crit = nn.MSELoss(); es = EarlyStopping()
    best_state = None
    for epoch in range(EPOCHS):
        model.train(); total = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _, _ = model(xb)
            loss = crit(pred, yb)
            for br in model.branches:
                loss = loss + l2_reg(br.dense1.parameters(), REG1)
                loss = loss + l2_reg(br.dense2.parameters(), REG2)
                loss = loss + l2_reg(br.contextual_weight.parameters(), REG3)
            loss.backward(); opt.step()
            with torch.no_grad():
                for br in model.branches: br.conv1.weight.clamp_(min=0)
            total += loss.item()
        model.eval(); vloss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, _, _ = model(xb); vloss += F.mse_loss(out, yb).item()
        vloss /= max(1, len(val_loader))
        stop = es(vloss, epoch)
        if es.best_epoch == epoch:   # new best-val epoch -> snapshot weights
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1:03d} | train={total/len(tr_loader):.4f} | val={vloss:.4f}", flush=True)
        if stop: print(f"Early stopping. Restoring best-val epoch {es.best_epoch+1}."); break
    if best_state is not None:
        model.load_state_dict(best_state)   # restore BEST-val weights, not the final over-fit model
    return model

def predict_all(model, X, device, batch=256):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out, _, _ = model(X[i:i+batch].to(device)); preds.append(out.cpu().numpy().ravel())
    return np.concatenate(preds)

def corr_S(y, yp): return float(spearmanr(y.ravel(), yp.ravel())[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motif_set = args.motif_set
    n_branches = NUM_BRANCHES if motif_set == "universal" else 1
    meme = UNIVERSAL_MEME if motif_set == "universal" else CONTROL_MEME
    motifs, names = parse_meme(meme)
    N_MOTIFS = len(motifs)
    print(f"[HEK|{motif_set}] {N_MOTIFS} motifs, sizes {sorted(set(m.shape[0] for m in motifs))}, "
          f"n_branches={n_branches}, device={device}", flush=True)

    X = np.load(ONEHOT)                    # (18356,10,400,4) bool
    y = np.load(YFILE)                     # (18356,) float32, NaN if no HPA
    keep = np.isfinite(y)
    X, y = X[keep], y[keep].astype(np.float32)
    N = X.shape[0]
    rng = np.random.permutation(N)
    X, y = X[rng], y[rng]
    X = X[:, :n_branches, :, :]
    print(f"[HEK|{motif_set}] modeled genes={N}  X={X.shape}", flush=True)

    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.1, random_state=SEED)
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(iter(kf.split(Xtr_f)))
    Xtr, Xval = Xtr_f[tr_idx], Xtr_f[val_idx]
    ytr, yval = ytr_f[tr_idx], ytr_f[val_idx]

    Xtr_t = torch.tensor(Xtr).float(); ytr_t = torch.tensor(ytr).float().unsqueeze(1)
    Xval_t = torch.tensor(Xval).float(); yval_t = torch.tensor(yval).float().unsqueeze(1)
    Xte_t = torch.tensor(Xte).float()

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xval_t, yval_t), batch_size=len(Xval_t), shuffle=False)

    model = MultiBranchModel(motifs, N_MOTIFS, motif_set, n_branches=n_branches).to(device)
    model = train_model(model, tr_loader, val_loader, device)

    tr_S = corr_S(ytr, predict_all(model, Xtr_t, device))
    val_S = corr_S(yval, predict_all(model, Xval_t, device))
    te_S = corr_S(yte, predict_all(model, Xte_t, device))
    print(f"[HEK|{motif_set}] train={tr_S:.4f} val={val_S:.4f} test={te_S:.4f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    pre = f"{OUT_DIR}/hek_{motif_set}"
    np.save(f"{pre}_train_S.npy", np.array([tr_S]))
    np.save(f"{pre}_val_S.npy", np.array([val_S]))
    np.save(f"{pre}_test_S.npy", np.array([te_S]))
    torch.save(model.state_dict(), f"{pre}_model.pth")

    # optional compare to EpiMap 116-line universal CNN test distribution
    tests = []
    if os.path.isdir(REF):
        for f in os.listdir(REF):
            if f.endswith(f"_{motif_set}_test_S.npy") and not f.startswith("hek_"):
                tests.append(float(np.load(f"{REF}/{f}")[0]))
    cmp = ""
    if tests:
        med = float(np.median(tests)); pct = float(np.mean(np.array(tests) < te_S) * 100)
        cmp = (f"EpiMap 116-line CNN {motif_set} test: median={med:.4f} "
               f"[{min(tests):.3f},{max(tests):.3f}] (n={len(tests)})\n"
               f"HEK test={te_S:.4f} -> {pct:.0f}th percentile")
    open(f"{pre}_SUMMARY.txt", "w").write(
        f"HEK293 CNN-CR ({motif_set}) — PWM-init multi-branch, matches posthoc 05\n"
        f"genes={N} branches={n_branches} motifs={N_MOTIFS}\n"
        f"train={tr_S:.4f} val={val_S:.4f} test={te_S:.4f}\n{cmp}\n")
    print(f"[HEK|{motif_set}] saved -> {pre}_*.npy/_model.pth\n{cmp}", flush=True)


if __name__ == "__main__":
    main()
