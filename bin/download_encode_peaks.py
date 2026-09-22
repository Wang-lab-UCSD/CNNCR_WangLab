#!/usr/bin/env python
"""Download the ENCODE TF ChIP-seq peak files used in Fig. 4d and Fig. 5 into the layout the scripts expect.

    python bin/download_encode_peaks.py --dest $ENHANCER_ROOT/encode4_tf_chipseq [--cells K562 HepG2 A549 GM12878 MCF.7 HEK293] [--jobs 8]

Reads data/encode/encode_tf_chipseq_peaks.tsv (one row per file: cell line, TF, ENCODE file accession, md5, URL) and writes
{dest}/{cell line}/{TF}_human/{ENCFF accession}.bed (narrowPeak, GRCh38, IDR thresholded peaks; the .gz is verified against the
ENCODE md5 and decompressed). Files already present are skipped. About 1,350 files, 1 GB; 20-40 min with 8 parallel downloads.
The ENCODE portal is the source of these files (https://www.encodeproject.org); please cite ENCODE when using them."""
import os, sys, csv, gzip, hashlib, shutil, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--dest", required=True); ap.add_argument("--cells", nargs="*", default=None); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
rows = [r for r in csv.DictReader(open(os.path.join(here, "data", "encode", "encode_tf_chipseq_peaks.tsv")), delimiter="\t") if a.cells is None or r["cell_line"] in a.cells]
print(f"{len(rows)} peak files for {sorted({r['cell_line'] for r in rows})} -> {a.dest}", flush=True)
def fetch(r):
    out = os.path.join(a.dest, r["cell_line"], r["tf_dir"], r["file_accession"] + ".bed")
    if os.path.exists(out) and os.path.getsize(out) > 0: return "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True); gz = out + ".gz"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(r["url"], gz)
            if hashlib.md5(open(gz, "rb").read()).hexdigest() != r["md5sum"]: raise IOError("md5 mismatch")
            with gzip.open(gz, "rb") as fi, open(out, "wb") as fo: shutil.copyfileobj(fi, fo)
            os.remove(gz); return "ok"
        except Exception as e:
            err = e
    return f"FAILED {r['file_accession']}: {err}"
with ThreadPoolExecutor(a.jobs) as ex:
    res = list(ex.map(fetch, rows))
print({k: sum(1 for x in res if x.startswith(k)) for k in ("ok", "skip", "FAILED")}); [print(x) for x in res if x.startswith("FAILED")]
