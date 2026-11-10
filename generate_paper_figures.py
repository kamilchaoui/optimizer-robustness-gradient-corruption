"""
generate_paper_figures.py
=========================
Generate publication-quality figures for the IEEE report.
Output: figures/paper/
Run: python3 generate_paper_figures.py
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

OUT = "figures/paper"
os.makedirs(OUT, exist_ok=True)

COLORS  = {"sgd":"#1f77b4","adamw":"#2ca02c","signsgd":"#d62728","lion":"#9467bd"}
LABELS  = {"sgd":"SGD","adamw":"AdamW","signsgd":"signSGD","lion":"Lion"}
ORDER   = ["sgd","adamw","signsgd","lion"]
FS_AX   = 13   # axis labels
FS_TICK = 11   # tick labels
FS_LEG  = 10   # legend
FS_ANN  = 9    # bar annotations / burst label
LW      = 2.0
LW_BASE = 1.0
BURST_S, BURST_E = 20, 25

def shade_burst(ax):
    ax.axvspan(BURST_S+1, BURST_E, alpha=0.10, color="red")
    ax.axvline(BURST_S+1, color="red", lw=0.8, ls=":")
    ax.axvline(BURST_E,   color="red", lw=0.8, ls=":")
    ymin, ymax = ax.get_ylim()
    ax.text((BURST_S+BURST_E)/2+1, ymin+0.04*(ymax-ymin),
            "burst", ha="center", color="red", fontsize=FS_ANN)

def style(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────────────────────
df_p = pd.read_csv("results/global_persistent.csv")


# ══ Figure 2: val_acc curves p=40%, persistent mode ════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
for opt in ORDER:
    sub  = df_p[(df_p["optimizer"]==opt)&(df_p["corruption_p"]==0.4)].sort_values("epoch")
    base = df_p[(df_p["optimizer"]==opt)&(df_p["corruption_p"]==0.0)].sort_values("epoch")
    if sub.empty: continue
    ax.plot(sub["epoch"]+1,  sub["val_acc"],  color=COLORS[opt], lw=LW,     label=LABELS[opt])
    if not base.empty:
        ax.plot(base["epoch"]+1, base["val_acc"], color=COLORS[opt], lw=LW_BASE, ls="--", alpha=0.4)
shade_burst(ax)
ax.set_xlabel("Epoch", fontsize=FS_AX)
ax.set_ylabel("Validation accuracy (%)", fontsize=FS_AX)
ax.set_title("Persistent mask, $p=40\\%$ (dashed = clean baseline)", fontsize=FS_AX)
ax.legend(fontsize=FS_LEG, loc="lower right")
style(ax); fig.tight_layout()
save(fig, "fig2_persistent_p040.png")


# ══ Figure 3: recovery time barplot (random mode) ══════════════════════════════
p_vals  = [0.05, 0.10, 0.20, 0.40]
rec = {
    "sgd":     [2,  8,  34, np.nan],
    "adamw":   [27, 21, 28, np.nan],
    "signsgd": [0,  0,   0, np.nan],
    "lion":    [15, 17,  15, 33],
}
x = np.arange(len(p_vals)); w = 0.8/len(ORDER)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, opt in enumerate(ORDER):
    for j, t in enumerate(rec[opt]):
        no_drop = (t == 0)           # accuracy never fell below threshold
        dnr     = np.isnan(t)        # fell but never recovered
        height  = 0 if (dnr or no_drop) else t

        bar = ax.bar(x[j]+i*w, height, w,
                     label=LABELS[opt] if j==0 else "_",
                     color=COLORS[opt],
                     hatch="////" if no_drop else "",
                     edgecolor="white" if no_drop else COLORS[opt])

        if dnr:
            lbl = "DNR"
        elif no_drop:
            lbl = "—"      # no drop at all, nothing to recover
        else:
            lbl = str(int(t))
        ax.text(x[j]+i*w + w/2, height + 0.5,
                lbl, ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")

ax.set_xticks(x+w*(len(ORDER)-1)/2)
ax.set_xticklabels([f"{int(p*100)}%" for p in p_vals], fontsize=FS_TICK)
ax.set_xlabel("Corruption probability $p$", fontsize=FS_AX)
ax.set_ylabel("Epochs to recover after burst", fontsize=FS_AX)
ax.set_title("Recovery time — random mask\n"
             "(DNR = did not recover;  — = accuracy never dropped)", fontsize=FS_AX)
ax.legend(fontsize=FS_LEG)
style(ax); fig.tight_layout()
save(fig, "fig3_recovery_time.png")

# ══ Figure 4: layer-wise accuracy gap at p=40%, random mode ═══════════════════
stem_r  = pd.read_csv("results/stem_random.csv")
head_r  = pd.read_csv("results/head_random_jo.csv")

b_stem = stem_r[stem_r.corruption_p==0].groupby("optimizer")["val_acc"].last()
b_head = head_r[head_r.corruption_p==0].groupby("optimizer")["val_acc"].last()

OPTS_LW = ["sgd", "adamw", "lion"]
labels_lw = ["SGD", "AdamW", "Lion"]

gap = {
    "global": [50.3, 61.4, 1.3],
    "stem":   [(b_stem[o] - stem_r[(stem_r.optimizer==o)&(stem_r.corruption_p==0.4)]["val_acc"].values[-1]) for o in OPTS_LW],
    "head":   [(b_head[o] - head_r[(head_r.optimizer==o)&(head_r.corruption_p==0.4)]["val_acc"].values[-1]) for o in OPTS_LW],
}
COLORS_LW = {"global": "#d62728", "stem": "#1f77b4", "head": "#2ca02c"}

x = np.arange(len(OPTS_LW)); w = 0.25
fig, ax = plt.subplots(figsize=(5.5, 3.2))
for i, (layer, vals) in enumerate(gap.items()):
    bars = ax.bar(x + (i-1)*w, vals, w, label=layer.capitalize(),
                  color=COLORS_LW[layer], zorder=3)
    for bar, v in zip(bars, vals):
        lbl = f"{v:.0f}" if v >= 1 else f"{v:.1f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels_lw, fontsize=FS_AX)
ax.set_ylabel("Accuracy drop vs. baseline (%)", fontsize=FS_AX)
ax.set_title("Layer-wise vs. global corruption at $p=40\\%$  (random mask)", fontsize=FS_AX)
ax.legend(fontsize=FS_LEG, loc="upper right")
ax.set_ylim(0, 70); ax.tick_params(labelsize=FS_TICK)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "fig4_layerwise.png")

# ══ Figures 5 & 6: val_acc curves at p=5% and p=20%, persistent mode ══════════
df_pers = pd.read_csv("results/global_persistent.csv")

for p_val, fname, title in [
    (0.05, "fig5_persistent_p005.png", "Persistent mask, $p=5\\%$  (dashed = clean baseline)"),
    (0.20, "fig6_persistent_p020.png", "Persistent mask, $p=20\\%$  (dashed = clean baseline)"),
]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt in ORDER:
        sub  = df_pers[(df_pers.optimizer==opt)&(df_pers.corruption_p==p_val)].sort_values("epoch")
        base = df_pers[(df_pers.optimizer==opt)&(df_pers.corruption_p==0.0)].sort_values("epoch")
        if sub.empty: continue
        ax.plot(sub["epoch"]+1,  sub["val_acc"],  color=COLORS[opt], lw=LW,     label=LABELS[opt])
        if not base.empty:
            ax.plot(base["epoch"]+1, base["val_acc"], color=COLORS[opt], lw=LW_BASE, ls="--", alpha=0.4)
    shade_burst(ax)
    ax.set_xlabel("Epoch", fontsize=FS_AX)
    ax.set_ylabel("Validation accuracy (%)", fontsize=FS_AX)
    ax.set_title(title, fontsize=FS_AX)
    ax.legend(fontsize=FS_LEG, loc="lower right")
    style(ax); fig.tight_layout()
    save(fig, fname)

print("Done.")
