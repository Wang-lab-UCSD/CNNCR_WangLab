#!/usr/bin/env python
"""
08c — PREVIEW: Fig 5 e/f/g statistics recomputed on ALL target genes within the
within-TF top-10% ChIP binding of each cell line (prof request 2026-08-17),
instead of the displayed top-20.

Loading + reader construction copied VERBATIM from 08_reader_switch_v2.py
(QN_SHARED=1 canonical run, 2026-08-14). Nothing in
results/reader_switching_qnshared/ is touched; all outputs go to
results/reader_switching_qnshared_alltargets/.

Three target-set definitions per reader record (native line):
  top20     — canonical rank-average top-20 (sanity check: must reproduce
              reader_switch_v2_stats.csv exactly)
  allbound  — EVERY gene bound by the reader's within-TF top-10% ChIP peaks in
              its native line (∩ shared 8,236-gene axis), self-gene excluded
  allresp   — allbound ∩ responsive (|qnorm z| ≥ per-line P10), self excluded
              (= the canonical "strong" set that top-20 is drawn from)
  t10x      — allbound ∩ the reader's own TOP DECILE of |qnorm KO effect| over the
              shared axis, self excluded (user definition 2026-08-19: "perturb-seq is
              only trustable in the top 10%"). Proposed Fig 5 e/f/g target set; the
              proposed Fig 5d top-20 = rank-average selection WITHIN this set
              (written to top20_from_t10x.csv with overlap vs the canonical top-20).

Env: CNN_CR_pytorch (LD_LIBRARY_PATH+=$CONDA_PREFIX/lib), ~3 min local.
"""
import os, csv, pickle, collections, re, json
import numpy as np
# anndata is imported lazily inside the raw-screen fallback below: a released run reads the
# precomputed quantile-normalised vectors instead and never needs it.
from scipy.stats import wilcoxon

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")
SEQ_ROOT = os.environ.get("SEQ_ROOT") or ENHANCER_ROOT + "/sequences"   # the one-hot training sequences

ROOT   = ENHANCER_ROOT
CHIP   = os.path.join(ROOT, "multi_agent/chipseq_enrichment")
OUTDIR = os.path.join(CHIP, "results/reader_switching")
QN_SHARED = True                                     # canonical: shared-gene quantile normalization
WDIR   = os.path.join(CHIP, "results/reader_switching_qnshared_alltargets")
os.makedirs(WDIR, exist_ok=True)

K562_H5AD = os.path.join(ROOT, "perturb_seq/K562_gwps_normalized_bulk_01.h5ad")
ORION_NPZ = SEQ_ROOT + "/HEK293T_orion_perturbseq/processed/orion_hek293t_replogle_matched.npz"
TSS_BED   = os.path.join(ROOT, "IGV_tracks/CAGE_TSS_hg38.bed")
FEAS      = os.path.join(OUTDIR, "hek_k562_switch_feasibility.csv")

PROM_WIN = 10000
PVAL     = 1e-4
EFF_KD   = -0.10
TOPN     = 20
STRONG_PCTL = 10
MIN_STRONG = 20
ACT_FRAC = 0.70

def norm(s): return re.sub(r'[-_.]', '', str(s)).upper()

# ------------------------------------------------------------------ TSS
print("Loading CAGE TSS ...", flush=True)
tss = {}
for l in open(TSS_BED):
    c = l.rstrip("\n").split("\t")
    if len(c) >= 4 and c[3].strip() and c[3].strip() not in tss:
        tss[c[3].strip()] = (c[0], int(c[1]))
chrom = collections.defaultdict(list)
for g, (ch, pos) in tss.items(): chrom[ch].append((pos, g))
for ch in chrom: chrom[ch].sort()
chrom_pos  = {ch: np.array([p for p, _ in v]) for ch, v in chrom.items()}
chrom_gene = {ch: [g for _, g in v] for ch, v in chrom.items()}
print(f"  {len(tss)} TSS genes")

# ------------------------------------------------------------------ per-line binding
ENCODE4 = ENHANCER_ROOT + "/encode4_tf_chipseq"
HALF_WIN = 150
SIG_PCTL = 90
def build_intensity(cell, peaksets):
    want = {norm(t) for t in pickle.load(open(peaksets, "rb"))}
    inten = {}; tf_sigs = collections.defaultdict(list)
    cdir = os.path.join(ENCODE4, cell)
    if not os.path.isdir(cdir): return inten, {}
    for d in os.listdir(cdir):
        rn = norm(d.replace("_human", ""))
        if rn not in want: continue
        p = os.path.join(cdir, d)
        if not os.path.isdir(p): continue
        beds = [f for f in os.listdir(p) if f.endswith(".bed")]
        if not beds: continue
        with open(os.path.join(p, beds[0])) as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) < 10: continue
                try:
                    chrom, start, sig, off = c[0], int(c[1]), float(c[6]), int(c[9])
                except (ValueError, IndexError): continue
                summit = start + (off if off >= 0 else 0)
                key = f"{chrom}:{max(0, summit - HALF_WIN)}-{summit + HALF_WIN}"
                if sig > inten.get(key, -1): inten[key] = sig
                tf_sigs[rn].append(sig)
    tf_thr = {rn: float(np.percentile(s, SIG_PCTL)) for rn, s in tf_sigs.items() if s}
    LO_PCTL = float(os.environ.get("CHIP_LO_PCTL", 10))   # 2026-10-01: CHIP_LO_PCTL=0 keeps every peak (no bottom-decile cut)
    tf_lo  = {rn: float(np.percentile(s, LO_PCTL)) for rn, s in tf_sigs.items() if s}   # bottom-10% cut by default
    return inten, tf_thr, tf_lo

def build_line(fimo, peaksets, want_motifs, inten_map):
    tf_peaks = pickle.load(open(peaksets, "rb"))
    tf_peaks = {norm(t): pk for t, pk in tf_peaks.items()}
    peak_id = {}
    for pks in tf_peaks.values():
        for k in pks:
            if k not in peak_id: peak_id[k] = len(peak_id)
    NP_ = len(peak_id)
    peak_inten = np.zeros(NP_)
    for k, pid in peak_id.items(): peak_inten[pid] = inten_map.get(k, 0.0)
    tf_peakids = {t: np.fromiter((peak_id[k] for k in pks), dtype=np.int64) for t, pks in tf_peaks.items()}
    peak_genes = [None] * NP_
    for k, pid in peak_id.items():
        ch, se = k.split(":"); s, e = se.split("-"); center = (int(s) + int(e)) // 2
        if ch not in chrom_pos: peak_genes[pid] = (); continue
        pos = chrom_pos[ch]
        lo = np.searchsorted(pos, center - PROM_WIN); hi = np.searchsorted(pos, center + PROM_WIN)
        peak_genes[pid] = tuple(chrom_gene[ch][gi] for gi in range(lo, hi))
    want = set(want_motifs)
    motif_peaks = collections.defaultdict(set)
    with open(fimo) as fh:
        next(fh)
        for l in fh:
            if not l.startswith("universal_"): continue
            f = l.rstrip("\n").split("\t")
            if f[0] not in want: continue
            try:
                if float(f[7]) > PVAL: continue
            except (ValueError, IndexError): continue
            pid = peak_id.get(f[2].split("::")[0])
            if pid is not None: motif_peaks[f[0]].add(pid)
    return dict(tf_peakids=tf_peakids, peak_genes=peak_genes, motif_peaks=motif_peaks, peak_inten=peak_inten)

# ------------------------------------------------------------------ Perturb-seq loaders
# Two ways in, and the released one is the first.
#
#   QN  (used whenever perturbseq_qn/ is present, which is what the Zenodo archive ships)
#       perturbseq_qn/perturbseq_qn_{K562,HEK293T}.npz hold the quantile-normalised knockout-
#       effect vector of every perturbed gene on the two screens' shared gene axis, written by
#       bin/zenodo_build/export_perturbseq_qn.py from the raw screens below. Everything
#       downstream reads only k_delta / o_delta on that axis, so the raw screens are not needed
#       to reproduce any Fig. 5d-g panel.
#   RAW (fallback, development machine only)
#       the original screens: Replogle's normalised bulk h5ad and the Orion matrix processed to
#       the same form. Needed only to rebuild the QN vectors, and NOT distributed.
QN_DIR = os.path.join(ROOT, "perturbseq_qn")
USE_QN = all(os.path.exists(os.path.join(QN_DIR, f"perturbseq_qn_{s}.npz"))
             for s in ("K562", "HEK293T"))

if USE_QN:
    print("Loading quantile-normalised Perturb-seq vectors (the raw screens are not needed) ...", flush=True)
    _QK = np.load(os.path.join(QN_DIR, "perturbseq_qn_K562.npz"), allow_pickle=True)
    _QH = np.load(os.path.join(QN_DIR, "perturbseq_qn_HEK293T.npz"), allow_pickle=True)
    assert list(_QK["genes"]) == list(_QH["genes"]), "the two QN files are on different gene axes"
    # delta_qn is ALREADY on the shared axis, in sorted order, so k_col/o_col are the identity
    # map over it. The chain builds kidx = [k_col[g] for g in sorted(shared)] and then indexes
    # k_delta(tf)[kidx], which therefore returns exactly the raw path's vector.
    _shl = [str(g) for g in _QK["genes"]]
    shared = set(_shl)
    k_ensg = o_ensg = list(_shl)
    k_col = {g: i for i, g in enumerate(_shl)}
    o_col = dict(k_col)
    _ktf = [str(t) for t in _QK["tfs"]]
    _otf = [str(t) for t in _QH["tfs"]]
    _krow = {t: i for i, t in enumerate(_ktf)}
    _orow = {t: i for i, t in enumerate(_otf)}
    _KD, _HD = _QK["delta_qn"], _QH["delta_qn"]
    k_perturbed, o_perturbed = set(_ktf), set(_otf)
    k_sym = {str(a): str(b) for a, b in zip(_QK["sym_ensg"], _QK["sym_name"])}
    o_sym = {str(a): str(b) for a, b in zip(_QH["sym_ensg"], _QH["sym_name"])}
    _keff = {t: bool(e) for t, e in zip(_ktf, _QK["eff_kd"])}
    o_effkd_mask = {t: bool(e) for t, e in zip(_otf, _QH["eff_kd"])}
    # not cached: the matrices are already resident, and 09x walks several hundred control TFs
    def k_delta(tf): return np.asarray(_KD[_krow[tf]], float)
    def o_delta(tf): return np.asarray(_HD[_orow[tf]], float)
    def k_effkd(tf): return _keff.get(tf, True)
    _kraw = {}          # the raw path's cache; kept so 09x/09z5 can still call _kraw.clear()
    _kcache, _ocache = {}, {}
    THR = {"K": float(_QK["thr"]), "H": float(_QH["thr"])}
    print(f"  {len(_shl):,} shared genes, {len(_ktf):,} K562 and {len(_otf):,} HEK293T perturbations")
    print(f"  responsive-target threshold (P{STRONG_PCTL}): "
          f"K562 >= {THR['K']:.3f}  HEK >= {THR['H']:.3f}")

else:
    import anndata as ad
    print("Loading K562 Perturb-seq ...", flush=True)
    A = ad.read_h5ad(K562_H5AD)
    Xk = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X); Xk = Xk.astype(float)
    Xk[np.isinf(Xk)] = np.nan
    k_ensg = [str(g) for g in A.var.index]
    k_col = {g: i for i, g in enumerate(k_ensg)}
    k_sym = {str(gid): str(sym) for gid, sym in zip(A.var.index, A.var["gene_name"])}
    k_tf_rows = collections.defaultdict(list); k_tf_self = {}
    for i, name in enumerate(A.obs.index):
        p = str(name).split("_"); sym = norm(p[1]); k_tf_rows[sym].append(i)
        if len(p) >= 4 and p[-1] in k_col: k_tf_self[sym] = p[-1]
    _kraw = {}
    def k_delta_raw(tf):
        if tf not in _kraw: _kraw[tf] = np.nanmean(Xk[k_tf_rows[tf], :], axis=0)
        return _kraw[tf]
    _kcache = {}
    def k_delta(tf):
        if tf not in _kcache: _kcache[tf] = qnormK(k_delta_raw(tf))
        return _kcache[tf]
    def k_effkd(tf):
        e = k_tf_self.get(tf)
        if e is None: return True
        return k_delta_raw(tf)[k_col[e]] < EFF_KD
    k_perturbed = set(k_tf_rows)

    print("Loading Orion HEK293T Perturb-seq ...", flush=True)
    O = np.load(ORION_NPZ, allow_pickle=True)
    Xo = O["X"]
    o_ensg = [str(x) for x in O["ensembl_id"]]
    o_sym  = {str(en): str(sy) for en, sy in zip(O["ensembl_id"], O["gene_name"])}
    o_col  = {g: i for i, g in enumerate(o_ensg)}
    o_pert = [norm(x) for x in O["perturbations"]]
    o_row  = {p: i for i, p in enumerate(o_pert)}
    o_effkd_mask = {norm(p): bool(e) for p, e in zip(O["perturbations"], O["effective_kd"])}
    def o_delta_raw(tf):
        return Xo[o_row[tf], :]
    _ocache = {}
    def o_delta(tf):
        if tf not in _ocache: _ocache[tf] = qnormH(o_delta_raw(tf))
        return _ocache[tf]
    o_perturbed = set(o_pert)

    shared = set(k_col) & set(o_col)
    print(f"  {len(shared)} genes measured in both K562 and HEK293T")

    _shl = sorted(shared)
    _QCOLS = {id(Xk): np.array([k_col[g] for g in _shl]),
              id(Xo): np.array([o_col[g] for g in _shl])} if QN_SHARED else {}
    if QN_SHARED: print("QN_SHARED=1: quantile normalization built on the shared genes only")
    _rng = np.random.default_rng(0)
    def _qsample(Xmat, nrow=800):
        r = _rng.choice(Xmat.shape[0], size=min(nrow, Xmat.shape[0]), replace=False)
        cols = _QCOLS.get(id(Xmat))
        v = (Xmat[np.ix_(r, cols)] if cols is not None else Xmat[r, :]).ravel()
        return v[np.isfinite(v)]
    _Pg = np.linspace(0, 1, 20001)
    _qK = np.quantile(_qsample(Xk), _Pg); _qH = np.quantile(_qsample(Xo), _Pg)
    _refg = 0.5 * (_qK + _qH)
    def qnormK(vec):
        out = np.asarray(vec, float).copy(); m = np.isfinite(out); out[m] = np.interp(out[m], _qK, _refg); return out
    def qnormH(vec):
        out = np.asarray(vec, float).copy(); m = np.isfinite(out); out[m] = np.interp(out[m], _qH, _refg); return out

    def _line_thr(Xmat, qfun):
        return float(np.percentile(np.abs(qfun(_qsample(Xmat))), STRONG_PCTL))
    THR = {"K": _line_thr(Xk, qnormK), "H": _line_thr(Xo, qnormH)}
    print(f"responsive-target threshold (P{STRONG_PCTL}): K562 >= {THR['K']:.3f}  HEK >= {THR['H']:.3f}")

# ------------------------------------------------------------------ switch set
print("Reading feasibility / switch set ...", flush=True)
switch = []
disp_name = {}
for r in csv.DictReader(open(FEAS)):
    if r["switch"] != "DIFF": continue
    kA = [t for t in r["K562_readers_chip_ko"].split(";") if t.strip()]
    kB = [t for t in r["HEK293_readers_chip_effKD"].split(";") if t.strip()]
    for t in kA + kB: disp_name.setdefault(norm(t), t)
    switch.append(dict(motif=r["motif"], kA=kA, kB=kB))
motif_ids = [s["motif"] for s in switch]
print(f"  {len(switch)} DIFF-switch motifs: {motif_ids}")

print("Indexing K562 binding + intensity ...", flush=True)
K_PKS = os.path.join(CHIP, "data/K562/tf_peak_sets.pkl")
H_PKS = os.path.join(CHIP, "data/HEK293/tf_peak_sets.pkl")
K_INT, K_THR, K_LO = build_intensity("K562", K_PKS)
K = build_line(os.path.join(CHIP, "results/fimo/K562/fimo_peaks/fimo.tsv"), K_PKS, motif_ids, K_INT)
print("Indexing HEK293 binding + intensity ...", flush=True)
H_INT, H_THR, H_LO = build_intensity("HEK293", H_PKS)
H = build_line(os.path.join(CHIP, "results/fimo/HEK293/fimo_peaks/fimo.tsv"), H_PKS, motif_ids, H_INT)
BIND_THR = {"K": K_THR, "H": H_THR}
BIND_LO  = {"K": K_LO, "H": H_LO}

def targets(line, motif, tf):
    mp = line["motif_peaks"].get(motif)
    tp = line["tf_peakids"].get(norm(tf))
    if not mp or tp is None or len(tp) == 0: return {}
    co = np.intersect1d(np.fromiter(mp, dtype=np.int64), tp, assume_unique=False)
    gi = {}
    for pid in co:
        inten = line["peak_inten"][pid]
        for g in line["peak_genes"][pid]:
            if inten > gi.get(g, -1.0): gi[g] = inten
    return gi

# ------------------------------------------------------------------ reader records
def make_reader(rn, native, motif):
    if native == "K":
        if rn not in k_perturbed or not k_effkd(rn): return None
        line = K; nat_delta = k_delta(rn); nat_cols = k_col
    else:
        if rn not in o_perturbed or not o_effkd_mask.get(rn, False): return None
        line = H; nat_delta = o_delta(rn); nat_cols = o_col
    full = {g: v for g, v in targets(line, motif, rn).items() if g in shared}
    bthr = BIND_THR[native].get(rn, 0.0)
    tgt = {g: v for g, v in full.items() if v >= bthr}
    lothr = BIND_LO[native].get(rn, 0.0)
    bound90 = {g: v for g, v in full.items() if v >= lothr}   # ChIP top-90% (bottom decile removed)
    cand = [(abs(nat_delta[nat_cols[g]]), g) for g in tgt if np.isfinite(nat_delta[nat_cols[g]])]
    strong = [(a, g) for a, g in cand if a >= THR[native]]
    if len(strong) < MIN_STRONG:
        dropped.append((disp_name.get(rn, rn), "K562" if native == "K" else "HEK293T", motif,
                        len(cand), len(strong)))
        return None
    strong.sort(reverse=True)
    erank = {g: i for i, (_, g) in enumerate(strong)}
    crank = {g: i for i, g in enumerate(sorted(erank, key=lambda g: -tgt[g]))}
    top = sorted(erank, key=lambda g: (erank[g] + crank[g], erank[g]))[:TOPN]
    top.sort(key=lambda g: -tgt[g])
    zK = k_delta(rn) if rn in k_perturbed else None
    zH = o_delta(rn) if rn in o_perturbed else None
    # t10x: the reader's own top decile of |qnorm native effect| over the shared axis
    absv = [(abs(nat_delta[nat_cols[g]]), g) for g in shared if np.isfinite(nat_delta[nat_cols[g]])]
    thr10 = float(np.percentile([a for a, _ in absv], 90))
    t10set = {g for a, g in absv if a >= thr10}
    # NEW vs 08: keep the FULL strong-bound universe too (self-gene excluded downstream)
    return dict(norm=rn, disp=disp_name.get(rn, rn), native=native, motif=motif,
                targets=set(g for _, g in strong), top=top, n_strong=len(strong),
                bound=set(tgt), bound_int=dict(tgt), t10set=t10set, t10thr=thr10,
                bound90=set(bound90), bound90_int=dict(bound90),
                nat_score=strong[0][0], zK=zK, zH=zH)

def zval(rec, gene, line):
    if line == "K":
        return rec["zK"][k_col[gene]] if (rec["zK"] is not None and gene in k_col) else np.nan
    return rec["zH"][o_col[gene]] if (rec["zH"] is not None and gene in o_col) else np.nan

dropped = []
motif_readers = {}
for s in switch:
    m = s["motif"]
    recs = []
    for t in s["kA"]:
        r = make_reader(norm(t), "K", m)
        if r: recs.append(r)
    for t in s["kB"]:
        r = make_reader(norm(t), "H", m)
        if r: recs.append(r)
    seen = set(); uniq = []
    for r in recs:
        key = (r["norm"], r["native"])
        if key in seen: continue
        seen.add(key); uniq.append(r)
    motif_readers[m] = uniq
    nk = sum(r["native"] == "K" for r in uniq); nh = len(uniq) - nk
    print(f"  {m}: {len(uniq)} readers with data  (K562 {nk}, HEK {nh})")

if dropped:
    print(f"\nExcluded {len(dropped)} reader(s):")
    for disp, ln, m, ntg, nst in sorted(dropped):
        print(f"    {disp:8} {ln:8} {m:12} {ntg:4} bound, only {nst} strong")

# ------------------------------------------------------------------ display scale (EXACTLY as 08: pooled over top-20 cells)
poolK, poolH = [], []
for m, recs in motif_readers.items():
    genes = list(dict.fromkeys(g for r in recs for g in r["top"]))
    for r in recs:
        for g in genes:
            vk = zval(r, g, "K"); vh = zval(r, g, "H")
            if np.isfinite(vk): poolK.append(abs(vk))
            if np.isfinite(vh): poolH.append(abs(vh))
_sall = np.nanpercentile(poolK + poolH, 97)
sK = sH = _sall
print(f"display scale (canonical definition): /{_sall:.3f}")

def reg_scaled(rec, gene, line):
    z = zval(rec, gene, line)
    if not np.isfinite(z): return np.nan
    v = -z / (sK if line == "K" else sH)
    return float(np.clip(v, -1, 1))

def not_self(rec, g):
    return norm(k_sym.get(g, o_sym.get(g, g))) != rec["norm"]

def geneset(rec, which):
    if which == "top20":    return list(rec["top"])                              # canonical, self as in 08
    if which == "allbound": return [g for g in rec["bound"]   if not_self(rec, g)]
    if which == "allresp":  return [g for g in rec["targets"] if not_self(rec, g)]
    if which == "t10x":     return [g for g in rec["bound"] if g in rec["t10set"] and not_self(rec, g)]
    if which == "c90k10":   return [g for g in rec["bound90"] if g in rec["t10set"] and not_self(rec, g)]
    if which == "b90t50":   # ChIP top-90% -> top-50 by |native KO effect| (user 2026-08-19, pure effect rank)
        ln = "K" if rec["native"] == "K" else "H"
        cand = [(abs(zval(rec, g, ln)), g) for g in rec["bound90"]
                if not_self(rec, g) and np.isfinite(zval(rec, g, ln))]
        cand.sort(reverse=True)
        return [g for _, g in cand[:50]]
    raise ValueError(which)

def mron(rec, ln, which):
    vals = [abs(reg_scaled(rec, g, ln)) for g in geneset(rec, which)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")

# ------------------------------------------------------------------ representative pairs (same rule as 08)
def best_reader(recs, native):
    cand = [r for r in recs if r["native"] == native]
    if not cand: return None
    return max(cand, key=lambda r: np.mean(sorted(
        [abs(zval(r, g, native == "K" and "K" or "H")) for g in r["top"]])[-TOPN:]))

combined = []
for m, recs in motif_readers.items():
    A_ = best_reader(recs, "K"); B_ = best_reader(recs, "H")
    if A_ is None or B_ is None: continue
    combined.append((m, A_, B_))
print(f"\n{len(combined)} motifs with a K562 & HEK representative reader")

DEFS = ["top20", "allbound", "allresp", "t10x", "c90k10", "b90t50"]

# per-motif stats for the representative pairs, all three definitions
motif_rows = []
for m, A_, B_ in combined:
    row = dict(motif=m, A=A_["disp"], B=B_["disp"],
               A_n_bound=len(geneset(A_, "allbound")), A_n_resp=len(geneset(A_, "allresp")),
               B_n_bound=len(geneset(B_, "allbound")), B_n_resp=len(geneset(B_, "allresp")),
               A_n_t10x=len(geneset(A_, "t10x")), B_n_t10x=len(geneset(B_, "t10x")),
               A_n_c90k10=len(geneset(A_, "c90k10")), B_n_c90k10=len(geneset(B_, "c90k10")),
               A_n_b90t50=len(geneset(A_, "b90t50")), B_n_b90t50=len(geneset(B_, "b90t50")))
    for w in DEFS:
        Ao, Af = mron(A_, "K", w), mron(A_, "H", w)
        Bo, Bf = mron(B_, "H", w), mron(B_, "K", w)
        row.update({f"A_own_{w}": Ao, f"A_off_{w}": Af, f"B_own_{w}": Bo, f"B_off_{w}": Bf,
                    f"on_diag_{w}": (Ao + Bo) / 2, f"off_diag_{w}": (Af + Bf) / 2,
                    f"switch_score_{w}": (Ao + Bo) / 2 - (Af + Bf) / 2})
    motif_rows.append(row)

with open(os.path.join(WDIR, "alltargets_stats_per_motif.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(motif_rows[0].keys())); w.writeheader()
    for r in motif_rows: w.writerow(r)

# per-reader-record stats (all readers with data, the 5f/pooled view)
reader_rows = []
for m, recs in motif_readers.items():
    for r in recs:
        own_ln = "K" if r["native"] == "K" else "H"
        off_ln = "H" if r["native"] == "K" else "K"
        row = dict(motif=m, reader=r["disp"], native="K562" if r["native"] == "K" else "HEK293T",
                   n_bound=len(geneset(r, "allbound")), n_resp=len(geneset(r, "allresp")),
                   n_t10x=len(geneset(r, "t10x")), n_c90k10=len(geneset(r, "c90k10")),
                   n_b90t50=len(geneset(r, "b90t50")))
        for w in DEFS:
            row[f"own_{w}"] = mron(r, own_ln, w); row[f"off_{w}"] = mron(r, off_ln, w)
        reader_rows.append(row)
with open(os.path.join(WDIR, "alltargets_stats_per_reader.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(reader_rows[0].keys())); w.writeheader()
    for r in reader_rows: w.writerow(r)

# direction counts on ALL responsive targets (aggregated per reader x native line, as 08 does)
agg = {}
for m, recs in motif_readers.items():
    for r in recs:
        key = (r["norm"], r["native"])
        d = agg.setdefault(key, dict(disp=r["disp"], native=r["native"], targets=set()))
        d["targets"] |= r["targets"]
dir_rows = []
for (rn, nat), d in agg.items():
    delta = k_delta(rn) if nat == "K" else o_delta(rn)
    cols  = k_col if nat == "K" else o_col
    cand = [g for g in d["targets"]
            if g in cols and np.isfinite(delta[cols[g]]) and abs(delta[cols[g]]) >= THR[nat]
            and norm(k_sym.get(g, o_sym.get(g, g))) != rn]
    if len(cand) < MIN_STRONG: continue
    regs = np.array([-delta[cols[g]] for g in cand])
    n_act = int(np.sum(regs > 0)); n_rep = int(np.sum(regs < 0))
    frac_act = n_act / len(cand)
    cls = "activator" if frac_act >= ACT_FRAC else ("repressor" if frac_act <= (1 - ACT_FRAC) else "both")
    dir_rows.append(dict(reader=d["disp"], native="K562" if nat == "K" else "HEK293T",
                         n_targets=len(cand), n_activated=n_act, n_repressed=n_rep,
                         frac_activated=round(frac_act, 3), classification=cls))
dir_rows.sort(key=lambda r: (r["native"], r["classification"], r["reader"]))
with open(os.path.join(WDIR, "direction_counts_alltargets.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dir_rows[0].keys())); w.writeheader()
    for r in dir_rows: w.writerow(r)

# direction counts + proposed-top-20 for the intersection definitions (t10x, c90k10)
XDEFS = [("t10x", "bound_int"), ("c90k10", "bound90_int"), ("b90t50", "bound90_int")]
dir_by_def, t20_by_def = {}, {}
for dname, intkey in XDEFS:
    agg_t = {}
    for m, recs in motif_readers.items():
        for r in recs:
            key = (r["norm"], r["native"])
            d = agg_t.setdefault(key, dict(disp=r["disp"], native=r["native"], targets=set()))
            d["targets"] |= set(geneset(r, dname))
    dir_rows_t = []
    for (rn, nat), d in agg_t.items():
        delta = k_delta(rn) if nat == "K" else o_delta(rn)
        cols  = k_col if nat == "K" else o_col
        cand = [g for g in d["targets"] if g in cols and np.isfinite(delta[cols[g]])]
        if len(cand) < MIN_STRONG: continue
        regs = np.array([-delta[cols[g]] for g in cand])
        n_act = int(np.sum(regs > 0)); n_rep = int(np.sum(regs < 0))
        frac_act = n_act / len(cand)
        cls = "activator" if frac_act >= ACT_FRAC else ("repressor" if frac_act <= (1 - ACT_FRAC) else "both")
        dir_rows_t.append(dict(reader=d["disp"], native="K562" if nat == "K" else "HEK293T",
                               n_targets=len(cand), n_activated=n_act, n_repressed=n_rep,
                               frac_activated=round(frac_act, 3), classification=cls))
    dir_rows_t.sort(key=lambda r: (r["native"], r["classification"], r["reader"]))
    with open(os.path.join(WDIR, f"direction_counts_{dname}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dir_rows_t[0].keys())); w.writeheader()
        for r in dir_rows_t: w.writerow(r)
    dir_by_def[dname] = dir_rows_t
    t20_rows = []
    for m, recs in motif_readers.items():
        for r in recs:
            gset = geneset(r, dname)
            if not gset: continue
            nat_ln = "K" if r["native"] == "K" else "H"
            eff = {g: abs(zval(r, g, nat_ln)) for g in gset}
            erank = {g: i for i, g in enumerate(sorted(eff, key=lambda g: -eff[g]))}
            crank = {g: i for i, g in enumerate(sorted(eff, key=lambda g: -r[intkey].get(g, 0.0)))}
            top20 = sorted(erank, key=lambda g: (erank[g] + crank[g], erank[g]))[:TOPN]
            ovl = len(set(top20) & set(r["top"]))
            t20_rows.append(dict(motif=m, reader=r["disp"], native="K562" if r["native"] == "K" else "HEK293T",
                                 n_set=len(gset), n_top=len(top20), overlap_with_canonical_top20=ovl,
                                 genes=";".join(top20)))
    with open(os.path.join(WDIR, f"top20_from_{dname}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t20_rows[0].keys())); w.writeheader()
        for r in t20_rows: w.writerow(r)
    t20_by_def[dname] = t20_rows
dir_rows_t = dir_by_def["t10x"]; t20_rows = t20_by_def["t10x"]   # keep old names for the summary below

# ------------------------------------------------------------------ summary
print("\n" + "=" * 100)
for w in DEFS:
    on = np.array([r[f"on_diag_{w}"] for r in motif_rows]); off = np.array([r[f"off_diag_{w}"] for r in motif_rows])
    npos = int(np.sum(on - off > 0))
    try: _, p_m = wilcoxon(on, off, alternative="greater")
    except ValueError: p_m = float("nan")
    own14 = []; oth14 = []
    for r in motif_rows:
        own14 += [r[f"A_own_{w}"], r[f"B_own_{w}"]]; oth14 += [r[f"A_off_{w}"], r[f"B_off_{w}"]]
    own14 = np.array(own14); oth14 = np.array(oth14)
    npos14 = int(np.sum(own14 - oth14 > 0))
    try: _, p_r = wilcoxon(own14, oth14, alternative="greater")
    except ValueError: p_r = float("nan")
    ownall = np.array([r[f"own_{w}"] for r in reader_rows]); othall = np.array([r[f"off_{w}"] for r in reader_rows])
    fin = np.isfinite(ownall) & np.isfinite(othall)
    nposP = int(np.sum(ownall[fin] - othall[fin] > 0))
    try: _, p_p = wilcoxon(ownall[fin], othall[fin], alternative="greater")
    except ValueError: p_p = float("nan")
    print(f"[{w:8s}] PANEL G per-motif (n={len(on)}): on {on.mean():.3f} vs off {off.mean():.3f}, "
          f"{npos}/{len(on)} positive, Wilcoxon p={p_m:.2e}")
    print(f"[{w:8s}] PANEL F per representative reader (n={len(own14)}): own {own14.mean():.3f} vs other "
          f"{oth14.mean():.3f}, {npos14}/{len(own14)} own>other, Wilcoxon p={p_r:.2e}")
    print(f"[{w:8s}] POOLED all reader records (n={int(fin.sum())}): own {ownall[fin].mean():.3f} vs other "
          f"{othall[fin].mean():.3f}, {nposP}/{int(fin.sum())} own>other, Wilcoxon p={p_p:.2e}")
    print("-" * 100)

cats = ["activator", "both", "repressor"]
for ln in ["K562", "HEK293T"]:
    cts = [sum(1 for r in dir_rows if r["native"] == ln and r["classification"] == c) for c in cats]
    print(f"PANEL E direction on ALL responsive targets, {ln}: act/mixed/rep = {cts}  (n={sum(cts)})")
for dname, _ in XDEFS:
    for ln in ["K562", "HEK293T"]:
        cts = [sum(1 for r in dir_by_def[dname] if r["native"] == ln and r["classification"] == c) for c in cats]
        print(f"PANEL E direction on {dname} targets, {ln}: act/mixed/rep = {cts}  (n={sum(cts)})")
for dname, _ in XDEFS:
    print(f"\nProposed Fig 5d top-20 (rank-average within {dname}) vs canonical top-20 overlap:")
    for r in t20_by_def[dname]:
        print(f"  {r['reader']:8s} {r['native']:8s} {r['motif']:13s} n_set={r['n_set']:4d} "
              f"top={r['n_top']:2d} overlap={r['overlap_with_canonical_top20']:2d}/20")

print("\nPer-motif detail (representative pairs):")
for r in motif_rows:
    print(f"  {r['motif']:13s} {r['A']:>7s}(nb={r['A_n_bound']:5d},nr={r['A_n_resp']:5d}) / "
          f"{r['B']:>7s}(nb={r['B_n_bound']:5d},nr={r['B_n_resp']:5d})  "
          f"score top20 {r['switch_score_top20']:+.3f}  allbound {r['switch_score_allbound']:+.3f}  "
          f"allresp {r['switch_score_allresp']:+.3f}")
print(f"\nOutputs -> {WDIR}/")
print("DONE.")
