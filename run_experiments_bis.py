"""
run_experiments.py
==================
Runs all (optimizer × corruption_p) combinations and saves results to CSV.

Quick smoke test (~2 min CPU):
    python run_experiments.py --epochs 5 --subset 2000 --optimizers sgd adamw --p_values 0.0 0.2

Full experiment:
    python run_experiments.py

With CNN2:
    python run_experiments.py --model cnn2 --epochs 60
"""

import argparse
import os

import pandas as pd
import torch
import torchvision.models as models

from data_loader import data_loader
from models.CNN import Net as CNN
from models.CNN2 import CNN_for_CIFAR as CNN2
from train_bis import train_model


def get_model(name):
    if name == "cnn":
        return CNN()
    if name == "cnn2":
        return CNN2()
    if name == "resnet18" : # load the standard ResNet18 with no weights because we train it from scratch
        model = models.resnet18(weights = None)
        model.fc = nn.Linear(model.fc.in_features, 10) # because 10 classes in CIFAr
        return model
    raise ValueError(f"Unknown model: {name!r}. Choose 'cnn' or 'cnn2' or 'resnet18'.")


def parse_args():
    p = argparse.ArgumentParser(description="OptML – gradient sign corruption experiments")
    p.add_argument("--model",         type=str,   default="cnn",
                   choices=["cnn", "cnn2", "resnet18"], help="Model architecture (default: cnn)")
    p.add_argument("--optimizers",    type=str,   nargs="+",
                   default=["sgd", "adamw", "signsgd", "lion"],
                   help="Optimizers to test (default: all four)")
    p.add_argument("--p_values",      type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.20, 0.40],
                   help="Corruption probabilities (default: 0 5 10 20 40%%)")
    p.add_argument("--layer",   type = str,     default = "classifier",
                   choices = ["head", "classifier", "stem", "global"], help = "Which layer to corrupt (default: classifier)")
    p.add_argument("--epochs",        type=int,   default=60,
                   help="Number of training epochs (default: 60)")
    p.add_argument("--corrupt_start", type=int,   default=20,
                   help="First corrupted epoch (default: 20)")
    p.add_argument("--corrupt_end",   type=int,   default=25,
                   help="First clean epoch after burst (default: 25)")
    p.add_argument("--mode",          type=str,   default="random",
                   choices=["random", "persistent"],
                   help="Corruption mode (default: random)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--subset",        type=int,   default=None,
                   help="Use only N training samples (smoke test)")
    p.add_argument("--batchsize",     type=int,   default=128)
    p.add_argument("--output",        type=str,   default="results/results.csv",
                   help="Output CSV path (default: results/results.csv)")
    p.add_argument("--lr_override",   type=float, default=None,
                   help="Override learning rate (e.g. 1e-3 to retune signSGD)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Epochs : {args.epochs}  |  Burst : [{args.corrupt_start}, {args.corrupt_end})")
    print(f"Mode   : {args.mode}")
    print(f"Layer  : {args.layer}")
    if args.subset:
        print(f"Subset : {args.subset} training samples  (smoke-test mode)")
    print()

    train_loader, test_loader, _ = data_loader(
        batchsize=args.batchsize,
        subset_size=args.subset,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_rows = []
    run_id = 0

    total_runs = len(args.optimizers) * len(args.p_values)
    done = 0

    for opt in args.optimizers:
        for p in args.p_values:
            done += 1
            print(f"\n{'='*60}")
            print(f"Run {done}/{total_runs} — optimizer={opt}  p={p:.2f}  mode={args.mode}")
            print('='*60)

            torch.manual_seed(args.seed)
            model = get_model(args.model)

            rows = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer_name=opt,
                lr_override=args.lr_override,
                epochs=args.epochs,
                corruption_p=p,
                corruption_layer = args.layer,
                corruption_start=args.corrupt_start,
                corruption_end=args.corrupt_end,
                corruption_mode=args.mode,
                device=device,
                seed=args.seed,
                run_id=run_id,
                model_name=args.model,
            )
            all_rows.extend(rows)
            run_id += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
