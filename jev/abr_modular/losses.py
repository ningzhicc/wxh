"""Module 2 objective, independent of the candidate-input module."""
from __future__ import annotations

import math


def decision_loss(logits, label, supervision, enabled=False, weight=.2, scale=4.3, clip=20.0, keys=None):
    import torch
    import torch.nn.functional as F
    if not all(math.isfinite(v) for v in (weight, scale, clip)) or weight < 0 or scale <= 0 or clip <= 0:
        raise ValueError("Invalid regret objective settings")
    z = logits.float()
    ce = F.cross_entropy(z[None], torch.tensor([label], device=z.device))
    regret = ce * 0
    valid = bool(supervision.get("valid"))
    if enabled and valid:
        values = supervision["regret_qoe"]
        expected = {str(b) for b in (300, 750, 1200, 1850, 2850, 4300)}
        keys = keys if keys is not None else [str(b) for b in (300, 750, 1200, 1850, 2850, 4300)]
        if set(values) != expected or set(keys) != expected or len(z) != len(keys):
            raise ValueError("Regret candidates differ from decision candidates")
        vector = torch.tensor([values[k] for k in keys], device=z.device, dtype=z.dtype)
        if not torch.isfinite(vector).all() or (vector < 0).any() or vector.min() > 1e-5:
            raise ValueError("Invalid candidate regrets")
        regret = (F.softmax(z, dim=-1) * vector.clamp(max=clip)).sum() / scale
    loss = ce + weight * regret if enabled else ce
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite decision loss")
    return loss, {"ce": float(ce.detach()), "regret": float(regret.detach()),
                  "regret_valid": int(enabled and valid)}
