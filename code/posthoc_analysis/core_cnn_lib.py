"""The 43-motif (motif-initialised) CNN-CR of Fig. 2e / Fig. 4e, as an importable module: the model classes, the training loop and the
hyperparameters are the verbatim block of the script used for the paper (posthoc_analysis/05_cnn_comparison.py on the development machine, which read the one-hot arrays). Notebook 05 trains it start to end from the window FASTA.
"""
import os
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

EIGHTBP_SETS = {"universal", "universal_denovo"}   # motif sets whose members are all 8 bp: one shared Conv1d, ten promoter branches

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Hyperparameters (same as CNN_CR.py) ───────────────────────────────────────
PRO_ONLY      = True
NUM_BRANCHES  = 10          # promoter windows only
SEQ_LEN       = 400         # bp per window
UNIV_K        = 8           # all universal regulators are 8 bp
CONV0_OUT_LEN = SEQ_LEN - UNIV_K + 1   # 393  — target length after the motif conv
BATCH         = 64
EPOCHS        = 100
LR            = 1e-4
PATIENCE      = 10
CONV1_FILTER  = 64
CONV2_FILTER  = 64
CONV3_FILTER  = 64
MAXPOOL0_SIZE = 5
MAXPOOL1_SIZE = 2
MAXPOOL2_SIZE = 3
MAXPOOL3_SIZE = 4
REG1, REG2, REG3       = 0.1, 0.2, 0.3
DROPOUT1, DROPOUT2     = 0.15, 0.15
GENES                  = 18356

# n_branches is now set per motif_set inside run_one()


# ── MEME parser ───────────────────────────────────────────────────────────────

def parse_meme(path):
    """Return list of (L, 4) float32 PWM arrays and their names."""
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


# ── First conv layer — two implementations ────────────────────────────────────

class UniversalMotifConv(nn.Module):
    """
    Single Conv1d for uniform 8-bp universal regulator PWMs.
    Output: (B, N_motifs, SEQ_LEN - 8 + 1) = (B, N, 393)
    """
    def __init__(self, motifs):
        super().__init__()
        n = len(motifs)
        for i, m in enumerate(motifs):
            if m.shape[0] != UNIV_K:
                raise ValueError(f"Universal motif {i} has length {m.shape[0]}, expected {UNIV_K}.")
        kernel = np.zeros((n, 4, UNIV_K), dtype=np.float32)
        for i, m in enumerate(motifs):
            kernel[i, :, :] = (m - 0.25).T
        self.conv = nn.Conv1d(4, n, kernel_size=UNIV_K, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(kernel))

    def forward(self, x):
        return self.conv(x)   # (B, N, 393)


class ControlMotifConv(nn.Module):
    """
    One Conv1d per JASPAR motif at its native kernel size, followed by
    AdaptiveMaxPool1d to normalise every output to CONV0_OUT_LEN (393).
    """
    def __init__(self, motifs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pool  = nn.AdaptiveMaxPool1d(CONV0_OUT_LEN)
        for m in motifs:
            k    = m.shape[0]
            conv = nn.Conv1d(4, 1, kernel_size=k, bias=False)
            w    = torch.from_numpy((m - 0.25).T).unsqueeze(0)   # (1, 4, k)
            with torch.no_grad():
                conv.weight.copy_(w)
            self.convs.append(conv)

    def forward(self, x):
        acts = [self.pool(c(x)) for c in self.convs]   # each (B, 1, 393)
        return torch.cat(acts, dim=1)                   # (B, N_motifs, 393)


# ── Model (same downstream as CNN_CR.py) ─────────────────────────────────────

class BiasLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = Parameter(torch.ones(n))
    def forward(self, x):
        return x * self.weight


class CNNBranch(nn.Module):
    def __init__(self, motifs, num_motifs, motif_set):
        super().__init__()

        if motif_set in EIGHTBP_SETS:
            self.conv0 = UniversalMotifConv(motifs)
        else:
            self.conv0 = ControlMotifConv(motifs)

        self.maxpool0 = nn.MaxPool1d(MAXPOOL0_SIZE)
        self.conv1    = nn.Conv1d(num_motifs,   CONV1_FILTER, 8, padding="same")
        self.maxpool1 = nn.MaxPool1d(MAXPOOL1_SIZE)
        self.conv2    = nn.Conv1d(CONV1_FILTER, CONV2_FILTER, 8, padding="same")
        self.maxpool2 = nn.MaxPool1d(MAXPOOL2_SIZE)
        self.conv3    = nn.Conv1d(CONV2_FILTER, CONV3_FILTER, 8, padding="same")
        self.maxpool3 = nn.MaxPool1d(MAXPOOL3_SIZE)
        self.flatten  = nn.Flatten()

        for c in (self.conv1, self.conv2, self.conv3):
            init.xavier_normal_(c.weight)

        with torch.no_grad():
            dummy = torch.zeros(1, 4, SEQ_LEN)
            d = self.conv0(dummy)
            d = F.relu(d)
            d = self.maxpool0(d)
            d = self.conv1(d); d = self.maxpool1(d)
            d = self.conv2(d); d = self.maxpool2(d)
            d = self.conv3(d); d = self.maxpool3(d)
            flat_size = self.flatten(d).shape[1]

        self.dense1            = nn.Linear(flat_size, num_motifs)
        self.dropout1          = nn.Dropout(DROPOUT1)
        self.dense2            = nn.Linear(num_motifs, num_motifs)
        self.dropout2          = nn.Dropout(DROPOUT2)
        self.contextual_weight = nn.Linear(num_motifs, num_motifs)
        self.bias_layer        = BiasLayer(1)

    def forward(self, x):
        is_nonzero = torch.any(x[:, 0, :] != 0, dim=1).unsqueeze(-1).float()
        x = x.permute(0, 2, 1)

        x        = self.conv0(x)
        mean_act = torch.mean(x, dim=2)

        x = F.relu(x)
        x = self.maxpool0(x)
        x = self.conv1(x);  x = self.maxpool1(x)
        x = self.conv2(x);  x = self.maxpool2(x)
        x = self.conv3(x);  x = self.maxpool3(x)
        x = self.flatten(x)

        x   = F.relu(self.dense1(x)); x = self.dropout1(x)
        x   = F.relu(self.dense2(x)); x = self.dropout2(x)
        ctx = self.contextual_weight(x) * is_nonzero
        combined  = ctx * mean_act
        return self.bias_layer(combined.sum(dim=1, keepdim=True)), ctx, combined


class MultiBranchModel(nn.Module):
    def __init__(self, motifs, num_motifs, motif_set, n_branches=NUM_BRANCHES):
        super().__init__()
        self.branches = nn.ModuleList(
            [CNNBranch(motifs, num_motifs, motif_set) for _ in range(n_branches)]
        )

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        branch_outputs    = torch.zeros(batch_size, len(self.branches), device=x.device)
        contextual_outputs = []
        combined_outputs   = []
        for i, branch in enumerate(self.branches):
            branch_output, contextual_output, combined_output = branch(x[:, i, :, :])
            branch_outputs[:, i] = branch_output.squeeze()
            contextual_outputs.append(contextual_output.squeeze())
            combined_outputs.append(combined_output.squeeze())
        final_output = branch_outputs.sum(dim=1, keepdim=True)
        return final_output, torch.stack(contextual_outputs, dim=1), torch.stack(combined_outputs, dim=1)


# ── Training helpers ──────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience = patience; self.best = np.inf; self.count = 0
    def __call__(self, val_loss):
        if val_loss < self.best: self.best = val_loss; self.count = 0
        else: self.count += 1
        return self.count >= self.patience


def l2_reg(params, s):
    return s * sum(torch.sum(p ** 2) for p in params)


def train_model(model, tr_loader, val_loader, device):
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    es   = EarlyStopping()
    for epoch in range(EPOCHS):
        model.train()
        total = 0
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
                for br in model.branches:
                    br.conv1.weight.clamp_(min=0)
            total += loss.item()
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs, _, _ = model(xb)
                total_val_loss += F.mse_loss(outputs, yb).item()
        val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1:03d} | train={total/len(tr_loader):.4f} | val={val_loss:.4f}")
        if es(val_loss):
            print("Early stopping."); break
    return model


def predict_all(model, X_tensor, device, batch=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch):
            out, _, _ = model(X_tensor[i:i+batch].to(device))
            preds.append(out.cpu().numpy().ravel())
    return np.concatenate(preds)


def corr_S(y, yp):
    return float(spearmanr(y.ravel(), yp.ravel())[0])


