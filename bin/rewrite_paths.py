#!/usr/bin/env python
"""Rewrite the two absolute data roots hard-coded in the archived scripts to your own locations.

The analysis scripts are archived verbatim from the development machine, where the project lived at
/stg3/data1/sam/enhancer_prediction and the large sequence inputs at /new-stg/home/sam2. Run once after
cloning and after downloading the Zenodo archives:

    python bin/rewrite_paths.py --enhancer-root /path/to/CNN-CR-core-regulators/work \
                                --seq-root      /path/to/CNN-CR-core-regulators/work/sequences

It also points the archived SLURM preambles at the shipped `cnncr` environment.
It edits every .py and .sh file under code/ and figures/ in place (a .orig copy is kept the first time and is
the source of every later run, so the script can be rerun with new roots at any time)
and prints the number of substitutions. The expected layout under --enhancer-root is given in PATHS.md.
"""
import argparse, os, re, shutil
ap=argparse.ArgumentParser(); ap.add_argument("--enhancer-root",required=True); ap.add_argument("--seq-root",required=True); a=ap.parse_args()
here=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); n=0; files=0
for base in ("code","figures"):
    for root,_,fs in os.walk(os.path.join(here,base)):
        for f in fs:
            if not f.endswith((".py",".sh")): continue
            p=os.path.join(root,f)
            if os.path.exists(p+".orig"): t=open(p+".orig",errors="ignore").read()   # idempotent: always rewrite from the archived original
            else:
                t=open(p,errors="ignore").read()
                if a.enhancer_root.rstrip("/") in t and "/stg3/data1/sam/enhancer_prediction/" not in t.replace(a.enhancer_root.rstrip("/"),""): continue   # generated after a previous rewrite: already uses the new root
            u=t.replace("/stg3/data1/sam/enhancer_prediction",a.enhancer_root.rstrip("/")).replace("/new-stg/home/sam2",a.seq_root.rstrip("/"))
            u=u.replace("/new-stg/home/hanbei/data/TF_ENCODE4",a.enhancer_root.rstrip("/")+"/encode4_tf_chipseq").replace("/new-stg/home/hanbei/data/hg38.fa",a.enhancer_root.rstrip("/")+"/genomes/hg38.fa").replace("/stg3/data1/sam/genome_annotation",a.enhancer_root.rstrip("/")+"/genome_annotation").replace("/stg3/data1/sam/utility",a.enhancer_root.rstrip("/")+"/genomes").replace("/stg3/data1/sam/meme/bin/","").replace("/stg3/data1/sam/Programs/bedtools2/bin/","").replace("/stg3/data1/sam/anaconda3/envs/multi_agent/bin/python","python").replace("/stg3/data1/sam/anaconda3/envs/CNN_CR_pytorch/bin/python","python").replace("/stg3/data1/sam/anaconda3/envs/enhancer/bin/python","python").replace("/stg3/data1/sam/anaconda3/bin/python","python").replace("source /stg3/data1/sam/anaconda3/etc/profile.d/conda.sh",'source "$(conda info --base)/etc/profile.d/conda.sh"')
            for _e in ("enhancer","multi_agent","CNN_CR_pytorch","parm_env"):   # the four development environments -> the one shipped environment
                u=u.replace(f"conda activate {_e}","conda activate cnncr")
            if u!=t:
                if not os.path.exists(p+".orig"): shutil.copy2(p,p+".orig")
                open(p,"w").write(u); n+=t.count("/stg3/data1/sam")+t.count("/new-stg/home/"); files+=1
import json; json.dump({"enhancer_root":a.enhancer_root.rstrip("/"),"seq_root":a.seq_root.rstrip("/")},open(os.path.join(here,".roots.json"),"w"),indent=1)   # read by the notebooks
print(f"rewrote {files} files, {n} path occurrences; roots saved to .roots.json")
