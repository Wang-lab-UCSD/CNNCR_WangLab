#!/usr/bin/env python
"""
STEP 1 — ChIP peak filtering: for each of the 5 cell lines, keep each TF's TOP-10%
signalValue peaks (within-TF percentile) that overlap a TSS+/-10kb promoter, at FULL
peak width. Emits the stats + within-TF rank curves that feed the peak-filtering figure.

  out/peak_filter_stats.csv     per cell line: TFs, peaks at each filter stage, widths
  out/rank_curves.npz           within-TF intensity ranking (median + example TFs)
No peak coordinates are written here; step 3 rebuilds the same top-10% set with signal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib.config import CELLS, SIG_PCTL, OUT
from lib.peaks import read_narrowpeak, tf_beds, top_intensity, Promoters

GRID = np.linspace(0, 100, 501)
MIN_PEAKS_CURVE = 100

def main():
    prom = Promoters()
    stats, curves = [], {}
    for cell in CELLS:
        n_tf = raw = tss = top = 0
        widths, per_tf_top, cell_curves = [], [], []
        for tf, bed in tf_beds(cell):
            recs = read_narrowpeak(bed)
            if not recs:
                continue
            n_tf += 1; raw += len(recs)
            near = [r for r in recs if prom.overlaps(r[0], r[1], r[2])]
            tss += len(near)
            if not near:
                per_tf_top.append(0); continue
            keep = top_intensity(near)
            top += len(keep); per_tf_top.append(len(keep))
            widths.extend(r[2] - r[1] for r in keep)
            if len(near) >= MIN_PEAKS_CURVE:                 # within-TF rank curve
                s = np.sort([r[3] for r in near])[::-1]
                L = np.log10(s); lo, hi = L.min(), L.max()
                if hi > lo:
                    cell_curves.append(np.interp(GRID, np.linspace(0, 100, len(s)),
                                                 (L - lo) / (hi - lo)))
        pt = np.array(per_tf_top); W = np.array(widths)
        stats.append(dict(cell=cell, n_tf=n_tf, raw=raw, tss=tss, top=top,
                          pct_tss=100 * tss / max(raw, 1), pct_top=100 * top / max(tss, 1),
                          tf_ge100=int((pt >= 100).sum()), tf_ge20=int((pt >= 20).sum()),
                          width_med=float(np.median(W)) if len(W) else np.nan,
                          width_p90=float(np.percentile(W, 90)) if len(W) else np.nan))
        C = np.vstack(cell_curves)
        curves[f"{cell}_med"] = np.median(C, 0)
        curves[f"{cell}_ex"] = C[:18]
        curves[f"{cell}_n"] = np.array([C.shape[0]])
        print(f"{cell:<9} TFs {n_tf} | raw {raw:,} -> TSS {tss:,} "
              f"({100*tss/max(raw,1):.0f}%) -> top-{100-SIG_PCTL}% {top:,} | "
              f"width med {np.median(W):.0f}", flush=True)

    curves["grid"] = GRID
    np.savez(f"{OUT}/rank_curves.npz", **curves)
    pd.DataFrame(stats).to_csv(f"{OUT}/peak_filter_stats.csv", index=False)
    print(f"\n[out] {OUT}/peak_filter_stats.csv, rank_curves.npz")

if __name__ == "__main__":
    main()
