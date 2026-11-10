"""
plot_results.py
===============
Load results CSV and generate all figures for the report.

Usage:
    python plot_results.py                                        # default paths
    python plot_results.py --input results/results.csv --output figures/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

COLORS = {
    "sgd":     "#1f77b4",   # blue
    "adamw":   "#2ca02c",   # green
    "signsgd": "#d62728",   # red
    "lion":    "#9467bd",   # purple
}

RECOVERY_THRESHOLD = 0.98   # corrupted acc must reach >= 98% of baseline acc


# ── helpers ──────────────────────────────────────────────────────────────────

def savefig(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")
    plt.close(fig)


def get_burst_window(df):
    return int(df["corruption_start"].iloc[0]), int(df["corruption_end"].iloc[0])


def get_optimizers(df):
    return sorted(df["optimizer"].unique(), key=lambda x: list(COLORS).index(x) if x in COLORS else 99)


def get_p_values(df):
    return sorted(df["corruption_p"].unique())


# ── figure 1 : accuracy (or loss) curves per corruption level ────────────────

def plot_metric_curves(df, metric, p, out_dir):
    burst_start, burst_end = get_burst_window(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for opt in get_optimizers(df):
        color = COLORS.get(opt, "grey")

        # corrupted run
        sub = df[(df["optimizer"] == opt) & (df["corruption_p"] == p)].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(sub["epoch"] + 1, sub[metric], color=color, lw=2, label=opt)

        # clean baseline (dashed)
        base = df[(df["optimizer"] == opt) & (df["corruption_p"] == 0.0)].sort_values("epoch")
        if not base.empty:
            ax.plot(base["epoch"] + 1, base[metric], color=color, lw=1, ls="--", alpha=0.4)

    ax.axvspan(burst_start + 1, burst_end, alpha=0.10, color="red")
    ax.axvline(burst_start + 1, color="red", lw=0.8, ls=":")
    ax.axvline(burst_end,       color="red", lw=0.8, ls=":")
    ymin, ymax = ax.get_ylim()
    ax.text((burst_start + burst_end) / 2 + 1, ymin + 0.02 * (ymax - ymin),
            "burst", ha="center", color="red", fontsize=8)

    label_map = {"val_acc": "Validation Accuracy (%)", "val_loss": "Validation Loss",
                 "train_loss": "Train Loss", "train_acc": "Train Accuracy (%)"}
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(label_map.get(metric, metric), fontsize=11)
    ax.set_title(f"{metric}  |  corruption p={int(p*100)}%  (dashed = clean baseline)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    fname = f"{metric}_p{int(p*100):03d}.png"
    savefig(fig, os.path.join(out_dir, fname))


# ── figure 2 : final accuracy drop vs p (barplot) ────────────────────────────

def plot_accuracy_drop(df, out_dir):
    p_values = [p for p in get_p_values(df) if p > 0]
    optimizers = get_optimizers(df)

    x = np.arange(len(p_values))
    width = 0.8 / len(optimizers)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, opt in enumerate(optimizers):
        baseline_rows = df[(df["optimizer"] == opt) & (df["corruption_p"] == 0.0)]
        if baseline_rows.empty:
            continue
        baseline_acc = baseline_rows.sort_values("epoch")["val_acc"].iloc[-1]

        drops = []
        for p in p_values:
            rows = df[(df["optimizer"] == opt) & (df["corruption_p"] == p)]
            if rows.empty:
                drops.append(0)
            else:
                final_acc = rows.sort_values("epoch")["val_acc"].iloc[-1]
                drops.append(baseline_acc - final_acc)

        ax.bar(x + i * width, drops, width, label=opt, color=COLORS.get(opt, "grey"))

    ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels([f"{int(p*100)}%" for p in p_values])
    ax.set_xlabel("Corruption probability p", fontsize=11)
    ax.set_ylabel("Final val-acc drop vs baseline (%)", fontsize=11)
    ax.set_title("Final accuracy drop after burst (↓ = worse)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    savefig(fig, os.path.join(out_dir, "accuracy_drop.png"))


# ── figure 3 : recovery time vs p (barplot) ──────────────────────────────────

def compute_recovery_time(df):
    """
    Recovery time = first epoch AFTER the burst where
    val_acc_corrupted >= RECOVERY_THRESHOLD * val_acc_baseline (same optimizer).
    Returns NaN if never recovered within the experiment.
    """
    burst_start, burst_end = get_burst_window(df)
    records = []

    for opt in get_optimizers(df):
        baseline = df[(df["optimizer"] == opt) & (df["corruption_p"] == 0.0)].sort_values("epoch")
        if baseline.empty:
            continue
        baseline_final = baseline["val_acc"].iloc[-1]

        for p in get_p_values(df):
            if p == 0.0:
                continue
            sub = df[(df["optimizer"] == opt) & (df["corruption_p"] == p)].sort_values("epoch")
            if sub.empty:
                continue

            post_burst = sub[sub["epoch"] >= burst_end]
            threshold  = RECOVERY_THRESHOLD * baseline_final
            recovered  = post_burst[post_burst["val_acc"] >= threshold]

            if recovered.empty:
                rec_time = float("nan")
            else:
                rec_time = int(recovered["epoch"].iloc[0]) - burst_end

            # additional metrics
            at_burst_end = sub[sub["epoch"] == burst_end - 1]
            drop_at_burst = (baseline_final - at_burst_end["val_acc"].values[0]
                             if not at_burst_end.empty else float("nan"))

            records.append({
                "optimizer":         opt,
                "corruption_p":      p,
                "baseline_final_acc": round(baseline_final, 2),
                "final_val_acc":     round(sub["val_acc"].iloc[-1], 2),
                "acc_gap":           round(baseline_final - sub["val_acc"].iloc[-1], 2),
                "drop_at_burst_end": round(drop_at_burst, 2),
                "recovery_time_epochs": rec_time,
            })

    return pd.DataFrame(records)


def plot_recovery_time(metrics_df, out_dir):
    p_values   = sorted(metrics_df["corruption_p"].unique())
    optimizers = [o for o in COLORS if o in metrics_df["optimizer"].unique()]

    x     = np.arange(len(p_values))
    width = 0.8 / len(optimizers)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, opt in enumerate(optimizers):
        times = []
        for p in p_values:
            row = metrics_df[(metrics_df["optimizer"] == opt) & (metrics_df["corruption_p"] == p)]
            t = row["recovery_time_epochs"].values[0] if not row.empty else float("nan")
            times.append(t)

        bars = ax.bar(x + i * width,
                      [0 if np.isnan(t) else t for t in times],
                      width, label=opt, color=COLORS.get(opt, "grey"))

        for bar, t in zip(bars, times):
            label = "DNR" if np.isnan(t) else str(int(t))
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    label, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels([f"{int(p*100)}%" for p in p_values])
    ax.set_xlabel("Corruption probability p", fontsize=11)
    ax.set_ylabel("Epochs to recover after burst (↓ better)", fontsize=11)
    ax.set_title(f"Recovery time (threshold = {int(RECOVERY_THRESHOLD*100)}% of baseline)  —  DNR = Did Not Recover", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    savefig(fig, os.path.join(out_dir, "recovery_time.png"))


# ── figure 4 : heatmap accuracy gap ──────────────────────────────────────────

def plot_heatmap(metrics_df, out_dir):
    optimizers = [o for o in COLORS if o in metrics_df["optimizer"].unique()]
    p_values   = sorted(metrics_df["corruption_p"].unique())

    data = np.full((len(optimizers), len(p_values)), np.nan)
    for i, opt in enumerate(optimizers):
        for j, p in enumerate(p_values):
            row = metrics_df[(metrics_df["optimizer"] == opt) & (metrics_df["corruption_p"] == p)]
            if not row.empty:
                data[i, j] = row["acc_gap"].values[0]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="Reds", aspect="auto",
                   vmin=0, vmax=np.nanmax(data) * 1.1 + 1e-6)
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"{int(p*100)}%" for p in p_values])
    ax.set_yticks(range(len(optimizers)))
    ax.set_yticklabels(optimizers)
    ax.set_xlabel("Corruption probability p", fontsize=11)
    ax.set_title("Final val-acc gap vs clean baseline  (↑ red = worse)", fontsize=11)
    for i in range(len(optimizers)):
        for j in range(len(p_values)):
            v = data[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.1f}",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Accuracy gap (%)")
    savefig(fig, os.path.join(out_dir, "heatmap_acc_gap.png"))


# ── figure 5 : zoomed window around burst ────────────────────────────────────

def plot_zoom_burst(df, out_dir, zoom=10):
    burst_start, burst_end = get_burst_window(df)
    p_values = [p for p in get_p_values(df) if p > 0]

    for p in p_values:
        fig, ax = plt.subplots(figsize=(7, 4))
        e_min = max(0, burst_start - zoom)
        e_max = burst_end + zoom

        for opt in get_optimizers(df):
            sub = df[(df["optimizer"] == opt) & (df["corruption_p"] == p)
                     & (df["epoch"] >= e_min) & (df["epoch"] <= e_max)].sort_values("epoch")
            base = df[(df["optimizer"] == opt) & (df["corruption_p"] == 0.0)
                      & (df["epoch"] >= e_min) & (df["epoch"] <= e_max)].sort_values("epoch")
            if sub.empty:
                continue
            color = COLORS.get(opt, "grey")
            ax.plot(sub["epoch"] + 1,  sub["val_acc"],  color=color, lw=2,  label=opt)
            if not base.empty:
                ax.plot(base["epoch"] + 1, base["val_acc"], color=color, lw=1, ls="--", alpha=0.4)

        ax.axvspan(burst_start + 1, burst_end, alpha=0.12, color="red")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        ax.set_title(f"Zoom around burst  |  p={int(p*100)}%", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        savefig(fig, os.path.join(out_dir, f"zoom_burst_p{int(p*100):03d}.png"))


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="results/results.csv")
    p.add_argument("--output", default="figures/")
    return p.parse_args()


def main():
    args  = parse_args()
    out   = args.output

    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows  |  optimizers: {sorted(df['optimizer'].unique())}")
    print(f"  p values: {sorted(df['corruption_p'].unique())}")
    print(f"  epochs: {df['epoch'].max() + 1}")
    print()

    p_values = [p for p in get_p_values(df) if p > 0]

    print("Generating accuracy & loss curves …")
    for p in p_values:
        plot_metric_curves(df, "val_acc",   p, out)
        plot_metric_curves(df, "val_loss",  p, out)
        plot_metric_curves(df, "train_loss", p, out)

    print("Generating accuracy drop barplot …")
    plot_accuracy_drop(df, out)

    print("Computing recovery metrics …")
    metrics = compute_recovery_time(df)
    metrics_path = os.path.join(os.path.dirname(args.input) or "results", "recovery_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  Recovery metrics saved -> {metrics_path}")
    print(metrics.to_string(index=False))

    print("\nGenerating recovery time barplot …")
    plot_recovery_time(metrics, out)

    print("Generating accuracy gap heatmap …")
    plot_heatmap(metrics, out)

    print("Generating zoom plots …")
    plot_zoom_burst(df, out)

    print(f"\nAll figures saved to {out}")


if __name__ == "__main__":
    main()
