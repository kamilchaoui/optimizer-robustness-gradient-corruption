import torch


def corrupt_gradient_signs(model, p, mode="random", _state=None):
    """
    Flip the sign of a fraction p of gradient coordinates.

    Args:
        model   : the nn.Module whose .grad tensors will be corrupted
        p       : fraction of coordinates to flip (0.0 = no corruption)
        mode    : "random"     – new random mask at every call (i.i.d.)
                  "persistent" – same mask throughout the burst window;
                                 pass _state={} on the first call, then
                                 forward the returned dict on every subsequent call
        _state  : dict used to persist the mask across steps (only for
                  mode="persistent"); pass None for mode="random"

    Returns:
        _state  : updated state dict (only meaningful for mode="persistent")

    Usage – random mode (simplest):
        corrupt_gradient_signs(model, p=0.2)

    Usage – persistent mode:
        state = {}
        for step in burst_steps:
            loss.backward()
            state = corrupt_gradient_signs(model, p=0.2,
                                           mode="persistent", _state=state)
            optimizer.step()
        state = {}   # reset at end of burst
    """
    if p <= 0:
        return _state

    if mode == "random":
        for param in model.parameters():
            if param.grad is not None:
                mask = torch.rand_like(param.grad) < p
                param.grad[mask] *= -1

    elif mode == "persistent":
        if _state is None:
            _state = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name not in _state:
                _state[name] = torch.rand_like(param.grad) < p
            param.grad[_state[name]] *= -1

    else:
        raise ValueError(f"Unknown corruption mode: {mode!r}. Choose 'random' or 'persistent'.")

    return _state


def corrupt_gradient_layerwise(model, p, layer="classifier", mode="random", _state=None):
    """
    Flip the sign of a fraction p of gradient coordinates,
    restricted to a specific layer of the model.

    Args:
        model   : the nn.Module whose .grad tensors will be corrupted
        p       : fraction of coordinates to flip (0.0 = no corruption)
        layer   : "head" or "classifier" – corrupt only the last Linear layer
                  "stem"                 – corrupt only the first Conv2d layer
                  "global"               – corrupt all layers (same as corrupt_gradient_signs)
        mode    : "random" or "persistent" (same semantics as corrupt_gradient_signs)
        _state  : persistent-mode state dict

    Returns:
        _state  : updated state dict
    """
    if p <= 0:
        return _state
    if _state is None:
        _state = {}

    # Identify which parameter names belong to the target layer
    target_names = _get_layer_param_names(model, layer)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if target_names is not None and name not in target_names:
            continue  # skip parameters not in the target layer

        if mode == "random":
            mask = torch.rand_like(param.grad) < p
            param.grad[mask] *= -1
        elif mode == "persistent":
            if name not in _state:
                _state[name] = torch.rand_like(param.grad) < p
            param.grad[_state[name]] *= -1
        else:
            raise ValueError(f"Unknown corruption mode: {mode!r}.")

    return _state


def _get_layer_param_names(model, layer):
    """Return the set of parameter names that belong to the requested layer."""
    if layer in ("global", None):
        return None  # None means "all parameters"

    if layer in ("head", "classifier"):
        # Find the last Linear module in the model
        last_linear_name = None
        for name, module in model.named_modules():
            if isinstance(module, __import__("torch").nn.Linear):
                last_linear_name = name
        if last_linear_name is None:
            return set()
        # Collect all parameter names that start with this module name
        prefix = last_linear_name + "."
        return {
            n for n, _ in model.named_parameters()
            if n.startswith(prefix) or n == last_linear_name
        }

    if layer == "stem":
        # Find the first Conv2d module in the model
        for name, module in model.named_modules():
            if isinstance(module, __import__("torch").nn.Conv2d):
                prefix = name + "."
                return {
                    n for n, _ in model.named_parameters()
                    if n.startswith(prefix) or n == name
                }
        return set()

    raise ValueError(f"Unknown layer: {layer!r}. Choose 'head', 'classifier', 'stem', or 'global'.")
