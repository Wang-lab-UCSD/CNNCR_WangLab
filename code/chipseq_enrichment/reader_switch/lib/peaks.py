"""
Shared peak helpers: read a TF's narrowPeak, apply the within-TF top-intensity cutoff,
and test TSS-proximity. Used by both the filtering figure and the reader-binding step so
the definition lives in exactly one place.
"""
import os, bisect, numpy as np
from lib.config import ENCODE4, PROMOTER, SIG_PCTL

# narrowPeak columns: 0 chr 1 start 2 end ... 6 signalValue 9 summit-offset
def read_narrowpeak(path):
    """-> list of (chrom, start, end, signalValue), full peak width."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 10:
                continue
            try:
                out.append((c[0], int(c[1]), int(c[2]), float(c[6])))
            except ValueError:
                pass
    return out

def tf_beds(cell):
    """Yield (tf_name, narrowPeak_path) for every ChIP'd TF in a cell line."""
    cdir = f"{ENCODE4}/{cell}"
    for d in sorted(os.listdir(cdir)):
        p = f"{cdir}/{d}"
        if not os.path.isdir(p):
            continue
        beds = [f for f in os.listdir(p) if f.endswith(".bed")]
        if beds:
            yield d.replace("_human", ""), f"{p}/{beds[0]}"

def top_intensity(recs):
    """Keep peaks with signalValue >= the SIG_PCTL-th percentile of THIS TF's own peaks."""
    if not recs:
        return recs
    thr = np.percentile([r[3] for r in recs], SIG_PCTL)
    return [r for r in recs if r[3] >= thr]

class Promoters:
    """TSS +/-10kb overlap test (the promoter bed defines the windows)."""
    def __init__(self):
        self.reg, self.starts = {}, {}
        d = {}
        with open(PROMOTER) as fh:
            for line in fh:
                c = line.split("\t")
                if len(c) < 3:
                    continue
                d.setdefault(c[0], []).append((int(c[1]), int(c[2])))
        for ch, v in d.items():
            v.sort(); self.reg[ch] = v; self.starts[ch] = [x[0] for x in v]

    def overlaps(self, chrom, s, e):
        if chrom not in self.reg:
            return False
        reg, st = self.reg[chrom], self.starts[chrom]
        i = bisect.bisect_right(st, e) - 1
        for j in range(max(0, i - 1), min(len(reg), i + 2)):
            ps, pe = reg[j]
            if s < pe and e > ps:
                return True
        return False
