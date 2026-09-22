"""phyloP100way (hg38) restricted to the union of the promoter windows (TSS +/- 10 kb peak windows, slots 0-9) of the 116 biosamples.
Writes a bigWig readable by the conservation scripts exactly like the full UCSC file (pyBigWig bw.stats over sites inside windows)."""
import os, sys, numpy as np, pandas as pd, pyBigWig, time
E="/stg3/data1/sam/enhancer_prediction"; SRC=f"{E}/multi_agent/conservation_sequence/hg38.phyloP100way.bw"
OUT=sys.argv[1] if len(sys.argv)>1 else f"{E}/github/_build/fig3/phyloP100way_hg38_promoter_windows.bw"
B=[b.strip() for b in open(f"{E}/Epimap_BSSID_nondup.txt") if b.strip()]
t=time.time(); iv={}
for i,b in enumerate(B):
    df=pd.read_csv(f"{E}/data_generated/{b}.coord.exp.df.final.csv",sep=",",lineterminator="\n",usecols=["chr","start","end"])
    df=df.iloc[[k for k in range(len(df)) if k%20<10]].dropna()               # promoter slots only
    for c,g in df.groupby("chr"): iv.setdefault(c,[]).append(np.c_[g["start"].values.astype(np.int64),g["end"].values.astype(np.int64)])
    if i%10==0: print(i,b,"%.0f s"%(time.time()-t),flush=True)
src=pyBigWig.open(SRC); chroms=src.chroms(); order=[c for c in chroms if c in iv]
def merge(a):
    a=a[np.argsort(a[:,0])]; out=[]; s,e=a[0]
    for x,y in a[1:]:
        if x<=e: e=max(e,y)
        else: out.append((s,e)); s,e=x,y
    out.append((s,e)); return out
merged={c:merge(np.concatenate(iv[c])) for c in order}; tot=sum(e-s for c in merged for s,e in merged[c]); print("union: %d intervals, %.1f Mb"%(sum(len(v) for v in merged.values()),tot/1e6),flush=True)
bw=pyBigWig.open(OUT,"w"); bw.addHeader([(str(c),int(chroms[c])) for c in sorted(order,key=lambda c: list(chroms).index(c))])
for c in sorted(order,key=lambda c: list(chroms).index(c)):
    for s,e in merged[c]:
        e=min(e,chroms[c]); v=src.values(c,s,e,numpy=True)
        if np.all(np.isnan(v)): continue
        ok=~np.isnan(v); edges=np.flatnonzero(np.diff(np.r_[False,ok,False]))          # contiguous runs of defined values
        for r0,r1 in zip(edges[::2],edges[1::2]): bw.addEntries(str(c),int(s+r0),values=v[r0:r1].astype(np.float64).tolist(),span=1,step=1)
    print(c,"done %.0f s"%(time.time()-t),flush=True)
bw.close(); src.close(); print("wrote",OUT,os.path.getsize(OUT)/1e9,"GB")
# check: same stats as the full file on 2,000 random K562 core-motif sites
chk=pyBigWig.open(OUT); src=pyBigWig.open(SRC)
bed=pd.read_csv(f"{E}/multi_agent/conservation_sequence/results/coordinates/BSS00762_universal.bed",sep="\t").sample(2000,random_state=0)
d=[abs((chk.stats(r.chrom,int(r.start),int(r.end),type="mean")[0] or 0)-(src.stats(r.chrom,int(r.start),int(r.end),type="mean")[0] or 0)) for r in bed.itertuples()]
print("max |subset - full| over 2000 K562 core sites: %.2e"%max(d)); print("SUBSET_DONE")
