import torch


class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                param.add_(-lr * torch.sign(param.grad))


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                state = self.state[param]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(param)

                m = state["m"]

                update = beta1 * m + (1 - beta1) * grad
                param.add_(-lr * torch.sign(update))

                m.mul_(beta2).add_(grad, alpha=1 - beta2)