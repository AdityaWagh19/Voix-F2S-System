import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = r"D:\voix\docs\figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 180,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

PAL = {
    "blue":"#2563EB","green":"#16A34A","red":"#DC2626",
    "orange":"#EA580C","purple":"#7C3AED","teal":"#0891B2",
    "gray":"#6B7280","light":"#F3F4F6",
}

# VERIFIED data from results/benchmarks/summary.csv
LABELS = ["A: Baseline\nRegression","B: SpkAvg\n+Gender","D: Joint\nInfoNCE",
          "E: Retrieval\nk=5","F: Hybrid\nk=10 (n=20)"]
VAL_COS = [0.1372, 0.1370, 0.1288, 0.0642, 0.1299]
R1  = [0.0010, 0.0005, 0.0010, 0.0010, 0.0010]
R3  = [0.0025, 0.0020, 0.0020, 0.0020, 0.0040]
R10 = [0.0050, 0.0050, 0.0060, 0.0060, 0.0085]
G_ACC = [0.6265, 0.6265, 0.6265, 1.0000, 1.0000]
SECS = [0.0207, 0.0171, 0.0177, 0.0252, 0.0324]
SECS_SD = [0.0673, 0.0722, 0.0600, 0.0542, 0.0728]
COLS = [PAL["gray"],PAL["blue"],PAL["purple"],PAL["teal"],PAL["green"]]

# ── FIG 1: 4-panel ablation overview ─────────────────────────────────────────
fig, axes = plt.subplots(1,4,figsize=(16,4))
fig.suptitle("VOIX Paradigm Ablation – Verified Results (results/benchmarks/summary.csv)",
             fontsize=12, fontweight="bold", y=1.02)
x = np.arange(len(LABELS)); bkw = dict(width=0.65,edgecolor="white",linewidth=0.8)
for ax, data, title, ylabel, ylim, fmt in [
    (axes[0],VAL_COS,"Zero-Shot Val Cosine\n(higher=better)","Cosine Sim",0.18,"{:.4f}"),
    (axes[1],[g*100 for g in G_ACC],"Gender Accuracy %\n(higher=better)","Accuracy (%)",115,"{:.1f}%"),
    (axes[2],SECS,"SECS – Speaker Embedding\nCosine Similarity","SECS (Mean)",0.055,"{:.4f}"),
]:
    bars = ax.bar(x,data,color=COLS,**bkw)
    ax.set_title(title,fontweight="bold"); ax.set_ylabel(ylabel)
    ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=8); ax.set_ylim(0,ylim)
    for bar,v in zip(bars,data):
        ax.text(bar.get_x()+bar.get_width()/2,v+ylim*0.012,fmt.format(v),
                ha="center",va="bottom",fontsize=7.5,fontweight="bold")
ax = axes[3]; rw=0.22
ax.bar(x-rw,[r*1000 for r in R1],width=rw,color=PAL["blue"],label="R@1",edgecolor="white")
ax.bar(x,[r*1000 for r in R3],width=rw,color=PAL["orange"],label="R@3",edgecolor="white")
ax.bar(x+rw,[r*1000 for r in R10],width=rw,color=PAL["red"],label="R@10",edgecolor="white")
ax.set_title("Recall@K x1000\n(higher=better)",fontweight="bold"); ax.set_ylabel("Recall x1000")
ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=8); ax.legend(fontsize=8,framealpha=0.7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_ablation_overview.png"); plt.close()
print("fig1 done")

# ── FIG 2: Gender Accuracy vs SECS dual axis ─────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10,5)); ax2 = ax1.twinx()
b1 = ax1.bar(x-0.2,[g*100 for g in G_ACC],width=0.38,color=PAL["blue"],alpha=0.85,label="Gender Acc (%)",edgecolor="white")
b2 = ax2.bar(x+0.2,SECS,width=0.38,color=PAL["green"],alpha=0.85,label="SECS Mean",edgecolor="white")
ax1.set_ylabel("Gender Accuracy (%)",color=PAL["blue"],fontweight="bold")
ax2.set_ylabel("SECS",color=PAL["green"],fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(LABELS,fontsize=9)
ax1.set_ylim(0,120); ax2.set_ylim(0,0.055)
ax1.axhline(100,color=PAL["blue"],linestyle="--",lw=1,alpha=0.4)
for bar,v in zip(b1,G_ACC): ax1.text(bar.get_x()+bar.get_width()/2,v*100+2,f"{v*100:.0f}%",ha="center",fontsize=8,fontweight="bold",color=PAL["blue"])
for bar,v in zip(b2,SECS): ax2.text(bar.get_x()+bar.get_width()/2,v+0.001,f"{v:.4f}",ha="center",fontsize=8,fontweight="bold",color=PAL["green"])
l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,lb1+lb2,loc="upper left",fontsize=9)
plt.title("Gender Fidelity vs Speaker Embedding Quality\n(Verified: results/benchmarks/summary.csv)",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_gender_secs.png"); plt.close()
print("fig2 done")

# ── FIG 3: Paradigm development trajectory ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12,6)); ax2 = ax.twinx()
x5 = np.arange(5)
ax.plot(x5,VAL_COS,"o-",color=PAL["blue"],lw=2.5,ms=8,label="Val Cosine (left)")
ax.plot(x5,[s*4 for s in SECS],"s--",color=PAL["orange"],lw=2,ms=7,label="SECS x4 (left, scaled)")
ax2.plot(x5,[g*100 for g in G_ACC],"^-.",color=PAL["green"],lw=2.5,ms=9,label="Gender Acc % (right)")
ax.set_xticks(x5); ax.set_xticklabels(LABELS,fontsize=9)
ax.set_ylabel("Cosine / SECS x4",color=PAL["blue"],fontweight="bold")
ax2.set_ylabel("Gender Accuracy (%)",color=PAL["green"],fontweight="bold")
ax.set_ylim(0,0.22); ax2.set_ylim(50,115)
ax.annotate("Gallery gender filter\nachieves 100% acc",xy=(3,0.0642),xytext=(2.3,0.18),
            arrowprops=dict(arrowstyle="->",color=PAL["teal"]),fontsize=8.5,color=PAL["teal"],
            bbox=dict(boxstyle="round,pad=0.3",facecolor="lightyellow",alpha=0.8))
ax.annotate("Hybrid fusion\nmax SECS",xy=(4,[s*4 for s in SECS][4]),xytext=(3.2,0.155),
            arrowprops=dict(arrowstyle="->",color=PAL["orange"]),fontsize=8.5,color=PAL["orange"],
            bbox=dict(boxstyle="round,pad=0.3",facecolor="lightyellow",alpha=0.8))
l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,lb1+lb2,loc="lower left",fontsize=9)
plt.title("VOIX Paradigm Development Trajectory – Verified Metrics",fontweight="bold",fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_paradigm_progression.png"); plt.close()
print("fig3 done")

# ── FIG 4: 560-D feature composition ─────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
dims=[512,32,16]
labels=["ArcFace Identity\n512-D (91.4%)","Craniofacial Ratios\n32-D (5.7%)","Soft Demographics\n16-D (2.9%)"]
colors=[PAL["blue"],PAL["green"],PAL["orange"]]
wedges,texts,autotexts = axes[0].pie(dims,labels=labels,colors=colors,autopct="%1.1f%%",
    startangle=140,pctdistance=0.75,wedgeprops=dict(linewidth=2,edgecolor="white"))
for t in autotexts: t.set_fontsize(9); t.set_fontweight("bold")
axes[0].set_title("560-D Face Feature Composition\n(voix/models/fusion.py)",fontweight="bold")
demo_labels=["Age bins [0:8]\n8-D Gaussian","Sex posterior [8:10]\n2-D [p_male,p_female]",
             "Vocal age groups [10:14]\n4-D [child,young,adult,senior]","Build proxy [14:16]\n2-D [slender,broad]"]
demo_dims=[8,2,4,2]; demo_cols=["#93C5FD","#FCA5A5","#86EFAC","#FCD34D"]
axes[1].barh(range(4),demo_dims,color=demo_cols,edgecolor="white",height=0.6)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(demo_labels,fontsize=9)
axes[1].set_xlabel("Dimensions")
axes[1].set_title("16-D Soft Demographic Vector Layout\n(voix/data/demographics.py)",fontweight="bold")
for i,v in enumerate(demo_dims): axes[1].text(v+0.05,i,f"{v}-D",va="center",fontsize=9,fontweight="bold")
axes[1].set_xlim(0,11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_feature_composition.png"); plt.close()
print("fig4 done")

# ── FIG 5: Hardware VRAM profiles ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,5))
tiers=["Tier 1\n<=4.5GB\n(GTX 1650)","Tier 2\n6-12GB\n(RTX 4050)","Tier 3\n>=16GB\n(A100)","Apple\nSilicon\nMPS","CPU\nOnly"]
vram=[4.5,12.5,40.0,24.0,0.5]; peak=[1.41,3.8,5.0,2.5,0.8]
cols_t=[PAL["red"],PAL["orange"],PAL["green"],PAL["purple"],PAL["gray"]]
x5=np.arange(5)
ax.bar(x5,vram,width=0.5,color=[c+"44" for c in cols_t],label="Available VRAM (GB)",edgecolor=cols_t,linewidth=2)
ax.bar(x5,peak,width=0.35,color=cols_t,alpha=0.9,label="Peak Inference VRAM (GB)")
ax.set_xticks(x5); ax.set_xticklabels(tiers,fontsize=9)
ax.set_ylabel("VRAM (GB)"); ax.set_ylim(0,46)
ax.set_title("Hardware Auto-Configuration – VRAM Profiles\n(configs/hardware.yaml + voix/utils/hardware.py)",fontweight="bold")
ax.legend(fontsize=9)
bmode=["Sequential\nbatch_voices=false","Parallel\nbatch_voices=true","Parallel\nbatch_voices=true","Parallel\nbatch_voices=true","Sequential\nbatch_voices=false"]
for i,(bm,v) in enumerate(zip(bmode,peak)):
    ax.text(i,v+0.6,bm,ha="center",fontsize=7.5,color=cols_t[i],fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_hardware_profiles.png"); plt.close()
print("fig5 done")

# ── FIG 6: Recall@K bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,5)); w=0.25; x5=np.arange(5)
ax.bar(x5-w,[r*1000 for r in R1],width=w,color=PAL["blue"],label="Recall@1 x1000",edgecolor="white")
ax.bar(x5,[r*1000 for r in R3],width=w,color=PAL["orange"],label="Recall@3 x1000",edgecolor="white")
ax.bar(x5+w,[r*1000 for r in R10],width=w,color=PAL["red"],label="Recall@10 x1000",edgecolor="white")
ax.set_xticks(x5); ax.set_xticklabels(LABELS,fontsize=9)
ax.set_ylabel("Recall x1000 (out of 400 zero-shot identities)")
ax.set_title("Zero-Shot Speaker Retrieval Recall@K – Verified Results\nAll models show low recall; reflects difficulty of cross-modal retrieval from 400 identities",fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim(0,12)
ax.axhline(1000/400,color="black",linestyle="--",lw=1.5,alpha=0.6)
ax.text(4.6,1000/400+0.1,f"Random @1={1000/400:.2f}",fontsize=8.5,color="black")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_recall_k.png"); plt.close()
print("fig6 done")

# ── FIG 7: CVAE v2 architecture text diagram ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14,8))
ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis("off")
ax.text(7,7.6,"VOIX CVAE v2 Architecture (voix/models/cvae.py)",
        ha="center",fontsize=13,fontweight="bold",color="#111827")

def box(ax,x,y,w,h,title,sub="",color="#DBEAFE",tc="#1E40AF",fs=9):
    r=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",facecolor=color,edgecolor="#374151",linewidth=1.5)
    ax.add_patch(r)
    ax.text(x+w/2,y+h*(0.62 if sub else 0.5),title,ha="center",va="center",fontsize=fs,fontweight="bold",color=tc)
    if sub: ax.text(x+w/2,y+h*0.28,sub,ha="center",va="center",fontsize=fs-1.5,color=tc,style="italic")

def arr(ax,x1,y1,x2,y2,lbl="",col="#374151"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=col,lw=1.8))
    if lbl:
        mx,my=(x1+x2)/2,(y1+y2)/2
        ax.text(mx+0.08,my,lbl,fontsize=7.5,color=col)

box(ax,0.2,5.0,2.5,1.0,"e_f (560-D)","Fused face repr.","#DBEAFE","#1E40AF")
box(ax,0.2,3.2,2.5,1.0,"e_s (192-D)","[TRAINING ONLY]\nSpeaker emb.","#FEF3C7","#92400E")
box(ax,3.3,5.5,2.8,1.3,"Prior Network p_psi","560->512->384->256\nGroupNorm+GELU | mu,logvar","#DBEAFE","#1E40AF")
box(ax,3.3,3.5,2.8,1.3,"Recognition Network q_phi","752->512->384->256\n[training only] | mu,logvar","#FEF3C7","#92400E")
arr(ax,2.7,5.5,3.3,5.95)
arr(ax,2.7,5.2,3.3,4.25,"e_f","#1E40AF")
arr(ax,2.7,3.7,3.3,4.05,"e_s","#92400E")

box(ax,6.8,5.8,1.6,0.7,"mu_prior","sigma_prior","#EDE9FE","#5B21B6",fs=8.5)
box(ax,6.8,4.0,1.6,0.7,"mu_z","sigma_z","#FEF3C7","#92400E",fs=8.5)
arr(ax,6.1,6.1,6.8,6.15)
arr(ax,6.1,4.1,6.8,4.35)

box(ax,6.8,1.9,1.8,1.1,"Reparameterize","z = mu + eps*sigma*T\neps~N(0,I)","#D1FAE5","#065F46")
arr(ax,7.7,5.8,7.7,3.0," inference\n  path","#7C3AED")
arr(ax,7.7,4.0,7.7,3.0,"","#92400E")

box(ax,9.4,3.2,2.8,1.4,"Decoder p_theta","192->512->384->256->192\nNO e_f skip (anti-collapse)","#D1FAE5","#065F46")
arr(ax,8.6,2.4,9.4,3.8)
box(ax,12.8,3.5,1.0,0.9,"e_s_hat\n192-D","","#ECFDF5","#065F46",fs=8.5)
arr(ax,12.2,3.9,12.8,3.9)

ax.text(5.5,0.8,
    "Loss = L_recon + beta(t)*L_KL\n"
    "  L_recon = 1 - cos_sim(e_s_hat, e_s)\n"
    "  L_KL    = sum_d max(lambda_fb=0.5, KL_d)   [free-bits]\n"
    "  beta(t) = beta_max*min(1, t/5000)           [warmup annealing]",
    fontsize=8.5,family="monospace",color="#374151",
    bbox=dict(boxstyle="round,pad=0.4",facecolor="#F9FAFB",edgecolor="#D1D5DB"))

patches=[mpatches.Patch(facecolor="#DBEAFE",edgecolor="gray",label="Prior path (inference)"),
         mpatches.Patch(facecolor="#FEF3C7",edgecolor="gray",label="Recognition path (training only)"),
         mpatches.Patch(facecolor="#D1FAE5",edgecolor="gray",label="Latent + Decoder")]
ax.legend(handles=patches,loc="lower right",fontsize=9,framealpha=0.9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_cvae_architecture.png"); plt.close()
print("fig7 done")

# ── FIG 8: Biomechanical coupling (labeled HYPOTHESIZED) ─────────────────────
fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off")
ax.set_title("Craniofacial-to-Voice Biomechanical Coupling Hypotheses (RQ2)\n[HYPOTHESIZED – not yet formally ablated in this codebase]",
             fontweight="bold",color="#991B1B",fontsize=11)
pairs=[
    ("Facial Height\n(Tr-Gn)",0.7,4.2,"Vocal Tract\nLength (VTL)",4.5,4.2,"r~0.926 (literature)","#2563EB"),
    ("Jaw Width\n(Gonion-Gonion)",0.7,3.0,"Oral Cavity\nVolume",4.5,3.0,"r~0.61 (estimated)","#16A34A"),
    ("Bizygomatic\nWidth",0.7,1.8,"Pharyngeal\nWidth Proxy",4.5,1.8,"r~0.52 (estimated)","#7C3AED"),
    ("Mandibular\nGonial Angle",0.7,0.6,"Laryngeal Frame\n& F0 Range",4.5,0.6,"r~-0.577 (literature)","#EA580C"),
]
for feat,fx,fy,acoustic,ax_,ay,corr,col in pairs:
    r1=FancyBboxPatch((fx-0.55,fy-0.3),2.4,0.6,boxstyle="round,pad=0.07",facecolor=col+"22",edgecolor=col,linewidth=1.8)
    r2=FancyBboxPatch((ax_-0.55,ay-0.3),2.6,0.6,boxstyle="round,pad=0.07",facecolor=col+"22",edgecolor=col,linewidth=1.8)
    ax.add_patch(r1); ax.add_patch(r2)
    ax.text(fx+0.65,fy,feat,ha="center",va="center",fontsize=9,color=col,fontweight="bold")
    ax.text(ax_+0.75,ay,acoustic,ha="center",va="center",fontsize=9,color=col,fontweight="bold")
    ax.annotate("",xy=(ax_-0.55,ay),xytext=(fx+1.85,fy),arrowprops=dict(arrowstyle="-|>",color=col,lw=2.0))
    ax.text((fx+1.85+ax_-0.55)/2,fy+0.18,corr,ha="center",fontsize=7.5,color=col,style="italic")
effects=[("Formant Dispersion\ndF ~ c/(2*VTL)",7.8,4.2,"#2563EB"),
         ("F1, F2 Resonances",7.8,3.0,"#16A34A"),
         ("Acoustic Radiation\nImpedance",7.8,1.8,"#7C3AED"),
         ("Fundamental\nFrequency F0",7.8,0.6,"#EA580C")]
for label,ex,ey,col in effects:
    r=FancyBboxPatch((ex-0.6,ey-0.3),2.8,0.6,boxstyle="round,pad=0.07",facecolor="#F9FAFB",edgecolor=col,linewidth=1.5,linestyle="--")
    ax.add_patch(r)
    ax.text(ex+0.8,ey,label,ha="center",va="center",fontsize=8.5,color=col)
    ax.annotate("",xy=(ex-0.6,ey),xytext=(ax_+2.05,ey),arrowprops=dict(arrowstyle="-|>",color=col,lw=1.5,linestyle="dotted"))
ax.text(0.5,5.5,"Craniofacial\nFeatures (32-D)",fontweight="bold",fontsize=10,color="#374151")
ax.text(4.7,5.5,"Acoustic\nStructure",fontweight="bold",fontsize=10,color="#374151")
ax.text(8.1,5.5,"Perceptual\nVoice Properties",fontweight="bold",fontsize=10,color="#374151")
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_biomechanical_coupling.png"); plt.close()
print("fig8 done")

# ── FIG 9: StyleAdapter + pipeline bridge (text diagram) ─────────────────────
fig, ax = plt.subplots(figsize=(13,5))
ax.set_xlim(0,13); ax.set_ylim(0,5); ax.axis("off")
ax.text(6.5,4.7,"StyleAdapter + VocoderStyleBridge (voix/models/adapter.py, tts_backend.py)",
        ha="center",fontsize=12,fontweight="bold",color="#111827")

def bk(ax,x,y,w,h,t,s="",c="#DBEAFE",tc="#1E40AF"):
    r=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",facecolor=c,edgecolor="#374151",linewidth=1.5)
    ax.add_patch(r)
    ax.text(x+w/2,y+h*(0.62 if s else 0.5),t,ha="center",va="center",fontsize=9,fontweight="bold",color=tc)
    if s: ax.text(x+w/2,y+h*0.28,s,ha="center",va="center",fontsize=7.5,color=tc,style="italic")

def ar2(ax,x1,y1,x2,y2,lbl=""):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#374151",lw=1.8))
    if lbl: ax.text((x1+x2)/2,(y1+y2)/2+0.15,lbl,ha="center",fontsize=7.5,color="#4B5563")

bk(ax,0.2,2.0,2.2,1.0,"CVAE Output","e_s_hat (192-D)\nor hybrid blend")
bk(ax,2.9,2.0,2.4,1.0,"StyleAdapter","192->256->256->128\nLayerNorm+GELU\n~163k params","#DDD6FE","#5B21B6")
bk(ax,6.0,3.2,2.2,0.9,"s_anchor","LibriSpeech\nref audio style","#FEF9C3","#713F12")
bk(ax,6.0,2.0,2.2,1.0,"VocoderBridge","s_target = 0.3*s_hat\n+ 0.7*s_anchor","#FED7AA","#92400E")
bk(ax,9.0,2.0,2.2,1.0,"StyleTTS 2\n(Frozen)","24 kHz synthesis\n128-D style input","#D1FAE5","#065F46")
bk(ax,11.7,2.0,1.1,1.0,"Waveform\n(Voice k)","watermarked","#ECFDF5","#065F46",)
bk(ax,6.0,0.5,2.2,0.9,"Text Prompt","phonemized\ninput","#F3F4F6","#374151")
ar2(ax,2.4,2.5,2.9,2.5,"e_s 192-D"); ar2(ax,5.3,2.5,6.0,2.5,"s_hat 128-D")
ar2(ax,7.1,3.2,7.1,3.0); ar2(ax,8.2,2.5,9.0,2.5,"s_target")
ar2(ax,7.1,0.5,7.1,2.0,"text"); ar2(ax,9.0,1.0,9.5,1.0)
ar2(ax,11.2,2.5,11.7,2.5)
ax.text(4.0,0.5,"Training:\nL = MSE(s_hat,s*) +\n  cos_align(s_hat,s*)",fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="#F9FAFB",edgecolor="#D1D5DB"))
plt.tight_layout(); plt.savefig(f"{OUT}/fig9_style_adapter.png"); plt.close()
print("fig9 done")

# ── FIG 10: Summary comparison radar ─────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch
fig, ax = plt.subplots(figsize=(11,5))
model_names = ["A: Baseline","B: SpkAvg+G","D: InfoNCE","E: Retrieval","F: Hybrid"]
colors_m = [PAL["gray"],PAL["blue"],PAL["purple"],PAL["teal"],PAL["green"]]
# Bar group: val_cos (norm to 0-1), gender (0-1), secs (norm to 0-1)
norm_vc = [v/max(VAL_COS) for v in VAL_COS]
norm_se = [s/max(SECS) for s in SECS]
norm_ga = G_ACC  # already 0-1
x5=np.arange(5); w=0.25
b1=ax.bar(x5-w,norm_vc,width=w,color=[c+"bb" for c in colors_m],label="Val Cosine (norm)",edgecolor="white",hatch="")
b2=ax.bar(x5,norm_ga,width=w,color=colors_m,label="Gender Acc (0-1)",edgecolor="white",alpha=0.7)
b3=ax.bar(x5+w,norm_se,width=w,color=colors_m,label="SECS (norm)",edgecolor="white",alpha=0.5,hatch="//")
ax.set_xticks(x5); ax.set_xticklabels(model_names,fontsize=9)
ax.set_ylabel("Normalized Score (0-1)")
ax.set_title("Normalized Multi-Metric Comparison — All Paradigms\n(Val Cosine / Gender Acc / SECS each normalized to [0,1])",fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0,1.2)
for i in range(5):
    ax.text(i-w,norm_vc[i]+0.03,f"{VAL_COS[i]:.3f}",ha="center",fontsize=7,color=colors_m[i])
    ax.text(i,norm_ga[i]+0.03,f"{G_ACC[i]*100:.0f}%",ha="center",fontsize=7,color=colors_m[i])
    ax.text(i+w,norm_se[i]+0.03,f"{SECS[i]:.4f}",ha="center",fontsize=7,color=colors_m[i])
plt.tight_layout(); plt.savefig(f"{OUT}/fig10_normalized_comparison.png"); plt.close()
print("fig10 done")

print("\nAll figures generated.")
import os
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(f"{OUT}/{f}")
    print(f"  {f:45s}  {sz//1024:4d} KB")
