#!/usr/bin/env python3
"""
run.py  —  Reproduce all paper figures and (optionally) all experiments.

Modes
-----
python run.py                 Regenerate all paper figures from pre-computed
                              results already stored in results/.  Fast (~5s).

python run.py --smoke         Run a 2-minute smoke-test experiment, then
                              generate figures from the full pre-computed data.

python run.py --full          Re-run every experiment from scratch, then
                              regenerate figures.
                              WARNING: takes ~40 h on CPU (no GPU required).

Output
------
All figures are written to figures/paper/:
  fig2_persistent_p040.png   Persistent mask, p=40%
  fig3_recovery_time.png     Recovery-time barplot (random mask)
  fig4_layerwise.png         Layer-wise accuracy drop at p=40%
  fig5_persistent_p005.png   Persistent mask, p=5%
  fig6_persistent_p020.png   Persistent mask, p=20%
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Experiment commands (run with --full)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # Global, persistent — all 4 optimizers
    {
        "cmd": [
            sys.executable, "run_experiments_bis.py",
            "--mode", "persistent", "--layer", "global",
            "--output", "results/global_persistent.csv",
        ],
        "desc": "Global / persistent  (SGD, AdamW, signSGD, Lion)",
    },
    # Stem (conv1), random — SGD / AdamW / Lion
    {
        "cmd": [
            sys.executable, "run_experiments_bis.py",
            "--mode", "random", "--layer", "stem",
            "--optimizers", "sgd", "adamw", "lion",
            "--output", "results/stem_attack_random.csv",
        ],
        "desc": "Stem / random  (SGD, AdamW, Lion)",
    },
    # Stem (conv1), random — signSGD  (lr override required)
    {
        "cmd": [
            sys.executable, "run_experiments_bis.py",
            "--mode", "random", "--layer", "stem",
            "--optimizers", "signsgd", "--lr_override", "1e-3",
            "--output", "results/stem_random_signsgd_lr1e3.csv",
        ],
        "desc": "Stem / random  (signSGD, lr=1e-3)",
    },
    # Stem (conv1), persistent — SGD / AdamW / Lion
    {
        "cmd": [
            sys.executable, "run_experiments_bis.py",
            "--mode", "persistent", "--layer", "stem",
            "--optimizers", "sgd", "adamw", "lion",
            "--output", "results/stem_attack_persistent.csv",
        ],
        "desc": "Stem / persistent  (SGD, AdamW, Lion)",
    },
    # Stem (conv1), persistent — signSGD
    {
        "cmd": [
            sys.executable, "run_experiments_bis.py",
            "--mode", "persistent", "--layer", "stem",
            "--optimizers", "signsgd", "--lr_override", "1e-3",
            "--output", "results/stem_persistent_signsgd_lr1e3.csv",
        ],
        "desc": "Stem / persistent  (signSGD, lr=1e-3)",
    },
]

# Merge commands run after the individual experiments
MERGE_SCRIPT = """
import pandas as pd
# Merge stem random
pd.concat([
    pd.read_csv('results/stem_attack_random.csv'),
    pd.read_csv('results/stem_random_signsgd_lr1e3.csv'),
], ignore_index=True).to_csv('results/stem_random.csv', index=False)

# Merge stem persistent
pd.concat([
    pd.read_csv('results/stem_attack_persistent.csv'),
    pd.read_csv('results/stem_persistent_signsgd_lr1e3.csv'),
], ignore_index=True).to_csv('results/stem_persistent.csv', index=False)

print("Merged CSVs written to results/stem_random.csv and results/stem_persistent.csv")
"""

REQUIRED_RESULTS = [
    "results/global_persistent.csv",
    "results/stem_random.csv",
    "results/stem_persistent.csv",
    "results/head_random_jo.csv",
    "results/head_persistent_jo.csv",
    "results/recovery_metrics.csv",
]


# ---------------------------------------------------------------------------
def check_results():
    missing = [f for f in REQUIRED_RESULTS if not os.path.exists(f)]
    return missing


def run_smoke_test():
    print("\n=== Smoke test (2 min, subset=2000) ===")
    cmd = [
        sys.executable, "run_experiments_bis.py",
        "--epochs", "5", "--subset", "2000",
        "--optimizers", "sgd", "adamw",
        "--p_values", "0.0", "0.2",
        "--output", "results/smoke_test.csv",
    ]
    subprocess.run(cmd, check=True)
    print("Smoke test passed.\n")


def run_full_experiments():
    print("\n=== Full experiment suite  (~40 h on CPU) ===")
    os.makedirs("results", exist_ok=True)
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['desc']}")
        subprocess.run(exp["cmd"], check=True)
    # Merge stem files
    subprocess.run([sys.executable, "-c", MERGE_SCRIPT], check=True)
    print("\nAll experiments done.")


def generate_figures():
    print("\n=== Generating paper figures ===")
    os.makedirs("figures/paper", exist_ok=True)
    subprocess.run([sys.executable, "generate_paper_figures.py"], check=True)
    print("Figures written to figures/paper/")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2-min smoke test before figure generation")
    parser.add_argument("--full",  action="store_true",
                        help="Re-run ALL experiments from scratch (~40 h)")
    args = parser.parse_args()

    if args.full:
        run_full_experiments()
    elif args.smoke:
        run_smoke_test()
    else:
        missing = check_results()
        if missing:
            print("Missing pre-computed result files:")
            for f in missing:
                print(f"  {f}")
            print("\nRun  python run.py --full  to generate them from scratch.")
            print("Or download them from the GitHub repository.")
            sys.exit(1)

    generate_figures()
    print("\nDone. Open figures/paper/ to view the plots.")


if __name__ == "__main__":
    main()
