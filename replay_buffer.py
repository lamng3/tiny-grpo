from dataclasses import dataclass, fields
from typing import Optional, Self

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    advantages: Optional[torch.Tensor]
    action_mask: torch.Tensor

    def to(self, device: torch.device) -> Self:
        """send elements to device (GPU/CPU)"""
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)