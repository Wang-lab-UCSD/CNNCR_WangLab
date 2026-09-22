"""Motif extraction from a trained CNN-CR model (the procedure of extract_CNN_CR_motifs_pytorch.py, as a function).

For one branch of one trained model:
  1. run every gene's window of that branch through the first convolution (256 filters x 8 bp) and through the
     contextual-weight module;
  2. for each filter, take the 1,000 positions with the largest |activation| over all genes and windows and
     build the PWM from the 8-mers under them (position frequencies);
  3. give the filter a weight: the median of (contextual weight x mean activation) over the genes where that
     product exceeds half of its maximum (0 if the maximum is not positive).
Branches 0-9 are the promoter branches, 10-19 the enhancer branches; a full model therefore yields 2,560 promoter
and 2,560 enhancer PWMs. Memory: the activations of one branch over 18,356 genes are 7.4 GB (float32).
"""
import numpy as np, torch

NUM_ACTIVATIONS, MOTIF_LENGTH = 1000, 8

def branch_forward(branch, X_branch, device, chunk=1024):
    """X_branch: (genes, 400, 4) bool/float -> conv0 activations (genes, 256, 393) float32, combined weight (genes, 256)."""
    branch.eval(); acts, comb = [], []
    with torch.no_grad():
        for i in range(0, len(X_branch), chunk):
            x = torch.tensor(np.asarray(X_branch[i:i+chunk])).float().to(device)
            a = branch.conv0(x.permute(0, 2, 1))                      # first-layer activations, what the hook captured
            _, _, c = branch(x)[:3]                                    # branch(x) -> (output, contextual weight, combined, ...)
            acts.append(a.cpu().numpy().astype(np.float32)); comb.append(c.cpu().numpy())
    return np.concatenate(acts), np.concatenate(comb)

def filter_weights(combined):
    """median of the combined weight over the genes where it exceeds half its maximum; 0 if that maximum is <= 0."""
    thr = 0.5 * np.max(combined, axis=0); w = np.zeros(combined.shape[1])
    for m in range(combined.shape[1]):
        if thr[m] <= 0: continue
        w[m] = np.median(combined[combined[:, m] > thr[m], m])
    return w

def filter_pwms(acts, X_branch):
    """PWM of every filter from the 8-mers under its 1,000 strongest |activations|."""
    acts_t = np.transpose(acts, (1, 0, 2))                             # (256, genes, 393)
    pwms, keep = [], []
    for m in range(acts_t.shape[0]):
        flat = np.abs(acts_t[m]).flatten()
        idx = np.argpartition(flat, -NUM_ACTIVATIONS)[-NUM_ACTIVATIONS:]
        g, p = np.unravel_index(idx, acts_t[m].shape)
        starts = p[:, None] + np.arange(MOTIF_LENGTH)
        sub = np.asarray(X_branch)[g[:, None, None], starts[:, :, None], :]            # (1000, 8, 1, 4)
        pwm = np.squeeze(np.sum(sub, axis=0), axis=1) / len(g)
        rs = pwm.sum(axis=1)
        if np.any(rs < 0.99): keep.append(False); pwms.append(pwm); continue          # too many N/absent positions: dropped
        if not np.allclose(rs, 1, atol=1e-6):
            fix = ~np.isclose(rs, 1, atol=1e-6); pwm[fix, -1] += 1 - rs[fix]
        keep.append(True); pwms.append(pwm)
    return pwms, np.array(keep)

def extract_branch(model, br, X, device):
    """Returns (list of (8,4) PWMs, weights) of the 256 filters of branch br; filters whose PWM rows do not sum to 1 are dropped."""
    Xb = X[:, br]
    acts, comb = branch_forward(model.branches[br], Xb, device)
    if not np.any(comb != 0): return [], np.array([])                  # no gene has a window in this branch
    w = filter_weights(comb); pwms, keep = filter_pwms(acts, Xb); del acts
    return [p for p, k in zip(pwms, keep) if k], w[keep]

def extract_model(model, X, device, branches=None, verbose=True):
    """All branches -> dict with 'pro' and 'enh': (PWMs, weights) pooled over the promoter (0-9) / enhancer (10-19) branches."""
    n = len(model.branches); branches = range(n) if branches is None else branches
    out = {"pro": ([], []), "enh": ([], [])}
    for br in branches:
        pw, w = extract_branch(model, br, X, device); key = "pro" if br < 10 else "enh"
        out[key][0].extend(pw); out[key][1].extend(w)
        if verbose: print(f"  branch {br:2d} ({key}): {len(pw)} PWMs", flush=True)
    return {k: (v[0], np.array(v[1])) for k, v in out.items()}

def write_meme(path, pwms):
    with open(path, "w") as f:
        f.write("MEME version 5.1.1\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        for i, m in enumerate(pwms):
            f.write(f"MOTIF {i} MEME-{i}\nletter-probability matrix: alength=4 w={m.shape[0]} nsites=1 E=0.0e+00\n")
            for row in m: f.write("\t".join(str(x) for x in row) + "\n")
            f.write("\n")

def read_meme(path):
    """MEME file -> list of (w, 4) arrays."""
    out = []
    for b in open(path).read().split("MOTIF ")[1:]:
        rows = [l.split() for l in b.split("\n")[2:] if l.strip() and not l.startswith(("URL", "letter"))]
        out.append(np.array([[float(x) for x in r] for r in rows if len(r) == 4]))
    return out
