#!/usr/bin/env python
"""Fig. 1d: predicted versus measured expression of the held-out K562 genes (CNN-CR, median of the ten models).

    python make_fig1d_k562.py --pred_dir <dir with BSS00762256MOTIF_pytorch_test_{y,pred}_all.npy> [--bssid BSS00762] [--out fig1d_k562]

--pred_dir is either data/k562_sample (the released K562 predictions, the paper's panel) or the work folder written by notebook 01. Writes <out>.svg/.pdf/.png at final size (48 x 48 mm).
"""
import os, argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy.stats import gaussian_kde, spearmanr
ap=argparse.ArgumentParser(); ap.add_argument("--pred_dir",required=True); ap.add_argument("--bssid",default="BSS00762"); ap.add_argument("--label",default="K562"); ap.add_argument("--out",default="fig1d_k562"); a=ap.parse_args()
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","Helvetica","DejaVu Sans"],"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
MM=1/25.4; PRO_E,PRO_F="#8FB8C9","#D6E7EE"; INK,SUB,AXC="#26313a","#66737c","#a9b4bb"; FS_A,FS_S,FS_XS=6.0,5.6,5.2
yt=np.median(np.load(f"{a.pred_dir}/{a.bssid}256MOTIF_pytorch_test_y_all.npy"),axis=0)/np.log2(10)
yp=np.median(np.load(f"{a.pred_dir}/{a.bssid}256MOTIF_pytorch_test_pred_all.npy"),axis=0)/np.log2(10)
rho=spearmanr(yt,yp)[0]; z=gaussian_kde(np.vstack([yt,yp]))(np.vstack([yt,yp])); i=z.argsort(); yt,yp,z=yt[i],yp[i],z[i]
base=matplotlib.colormaps["viridis"].resampled(256); cm=ListedColormap(base(np.linspace(0,1,256))[:-1,:]); cm.set_over("yellow")
fig=plt.figure(figsize=(48*MM,48*MM)); gs=fig.add_gridspec(2,2,width_ratios=[1,0.17],height_ratios=[0.17,1],wspace=0.05,hspace=0.05,left=0.2,right=0.98,bottom=0.18,top=0.98)
axd=fig.add_subplot(gs[1,0]); adx=fig.add_subplot(gs[0,0],sharex=axd); ady=fig.add_subplot(gs[1,1],sharey=axd)
axd.scatter(yt,yp,c=z,s=0.5,cmap=cm,norm=Normalize(vmin=0,vmax=0.9,clip=True),rasterized=True)
axd.plot([0,3],[0,3],color=SUB,lw=0.7,ls=(0,(3,2)),zorder=4)
axd.text(0.05,2.93,f"{a.label}\nSpearman ρ = {rho:.3f}",fontsize=FS_XS,ha="left",va="top",color=INK,linespacing=1.35)
axd.set_xlim(-0.06,3.06); axd.set_ylim(-0.06,3.06); axd.set_xticks([0,1,2,3]); axd.set_yticks([0,1,2,3]); axd.set_box_aspect(1)
line=np.linspace(0,3,400); kx,ky=gaussian_kde(yt),gaussian_kde(yp)
adx.plot(line,kx(line),color=PRO_E,lw=0.7); adx.fill_between(line,kx(line),color=PRO_F); ady.plot(ky(line),line,color=PRO_E,lw=0.7); ady.fill_betweenx(line,ky(line),color=PRO_F); adx.axis("off"); ady.axis("off")
for sp in ("top","right"): axd.spines[sp].set_visible(False)
for sp in ("left","bottom"): axd.spines[sp].set_color(AXC); axd.spines[sp].set_linewidth(0.7)
axd.tick_params(labelsize=FS_S,colors=SUB,width=0.7,length=2.0,pad=1.4)
for t in axd.get_xticklabels()+axd.get_yticklabels(): t.set_color(INK)
axd.set_xlabel("actual expression (log10 FPKM)",fontsize=FS_A,color=INK,labelpad=1.5); axd.set_ylabel("predicted expression (log10 FPKM)",fontsize=FS_A,color=INK,labelpad=1.5)
for ext in ("svg","pdf","png"): fig.savefig(f"{a.out}.{ext}",dpi=600,transparent=True)
print(f"[d] {a.label} Spearman rho = {rho:.4f} (n = {len(yt)} held-out genes) -> {a.out}.svg/.pdf/.png")
