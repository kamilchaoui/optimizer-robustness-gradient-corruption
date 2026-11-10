import torch
import torch.nn as nn
import torch.optim as optim

from corruption import corrupt_gradient_signs, corrupt_gradient_layerwise
from optimizers import SignSGD, Lion


# Default hyper-parameters per optimizer (tuned for CIFAR-10 / CNN)
OPTIMIZER_DEFAULTS = {
    "sgd":     dict(lr=0.05,  momentum=0.9, weight_decay=5e-4),
    "adamw":   dict(lr=1e-3,  weight_decay=1e-2),
    "signsgd": dict(lr=0.01),
    "lion":    dict(lr=1e-4,  betas=(0.9, 0.99), weight_decay=1e-2),
}


def build_optimizer(name, model, lr_override=None):
    cfg = dict(OPTIMIZER_DEFAULTS[name])  # copy to avoid mutating defaults
    if lr_override is not None:
        cfg["lr"] = lr_override
    if name == "sgd":
        return optim.SGD(model.parameters(), **cfg)
    if name == "adamw":
        return optim.AdamW(model.parameters(), **cfg)
    if name == "signsgd":
        return SignSGD(model.parameters(), **cfg)
    if name == "lion":
        return Lion(model.parameters(), **cfg)
    raise ValueError(f"Unknown optimizer: {name!r}")


def evaluate(model, loader, device):
    """Returns (accuracy_percent, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, total_loss / total


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer_name="adamw",
    lr_override=None,
    epochs=60,
    corruption_p=0.0,
    corruption_start=20,
    corruption_end=25,
    corruption_mode="random",
    device="cpu",
    seed=42,
    run_id=0,
    model_name="cnn",
    corruption_layer = "classifier"
):
    """
    Train model and return a list of per-epoch dicts (one row per epoch).

    The corruption hook is applied after loss.backward() and before
    optimizer.step(), only during epochs [corruption_start, corruption_end).
    """
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = build_optimizer(optimizer_name, model, lr_override=lr_override)
    criterion = nn.CrossEntropyLoss()

    cfg = OPTIMIZER_DEFAULTS[optimizer_name]
    lr = cfg.get("lr")
    weight_decay = cfg.get("weight_decay", 0.0)

    history = []
    corruption_state = {}   # used only in persistent mode

    for epoch in range(epochs):
        is_corrupted = corruption_p > 0 and corruption_start <= epoch < corruption_end
        if not is_corrupted:
            corruption_state = {}   # reset mask when burst ends

        # ── Training pass ─────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if is_corrupted:
                corruption_state = corrupt_gradient_layerwise(
                    model, corruption_p,
                    layer = corruption_layer,
                    mode=corruption_mode,
                    _state=corruption_state,
                )

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc  = 100 * correct / total

        # ── Validation pass ───────────────────────────────────────────────
        val_acc, val_loss = evaluate(model, test_loader, device)

        row = {
            "run_id":           run_id,
            "seed":             seed,
            "epoch":            epoch,
            "model":            model_name,
            "optimizer":        optimizer_name,
            "lr":               lr,
            "weight_decay":     weight_decay,
            "corruption_p":     corruption_p,
            "corruption_layer": corruption_layer,
            "corruption_start": corruption_start,
            "corruption_end":   corruption_end,
            "corruption_mode":  corruption_mode,
            "train_loss":       train_loss,
            "train_acc":        train_acc,
            "val_loss":         val_loss,
            "val_acc":          val_acc,
            "is_corrupted_epoch": is_corrupted,
        }
        history.append(row)

        print(
            f"[{optimizer_name:8s} p={corruption_p:.2f}] "
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.1f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.1f}%"
            + (" [CORRUPTED]" if is_corrupted else "")
        )

    return history
