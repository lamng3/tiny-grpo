from typing import Optional, Literal
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Approximates the KL divergence between the current policy and a reference policy
    using a Monte Carlo method (k3 estimator).
    See: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs (torch.Tensor): Log-probabilities from the current policy.
        log_probs_ref (torch.Tensor): Log-probabilities from the reference policy.
        action_mask (Optional[torch.Tensor]): Mask to select valid positions (1=keep, 0=ignore).

    Returns:
        torch.Tensor: Average KL divergence over the valid positions.
    """
    logr = log_probs_ref.float() - log_probs.float() # log ratio
    if action_mask is not None:
        logr = logr * action_mask
    return (logr.exp() - 1) - logr # r - 1 - logr


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
    strategy: Literal["grpo", "dr.grpo"] = "grpo",
    generate_max_length: int = 1024, 
) -> torch.Tensor:
    """
    Compute the mean over `dim`, ignoring elements where `mask==0`.
    
    Args:
        tensor (torch.Tensor): Input tensor whose mean to be computed.
        mask (Optional[torch.Tensor]): Binary mask same shape as tensor (1=keep, 0=ignore).
        dim (int or tuple of ints, optional): 
            Dimension(s) along which to compute the mean.
            If 'None', mean is taken over all elements in tensor.
        
    Returns:
        torch.Tensor: The masked mean of `tensor` along `dim`.
    """
    assert strategy in ("grpo", "dr.grpo", "dapo")
    if mask is None:
        # previously a bug appear here, it is fixed now
        # torch.mean() is used instead of tensor.mean()
        # so the loss treated as a function instead of a tensor
        # which results in 'int' object has no attribute 'long'
        # then GRPOLoss forward is calculated incorrectly,
        # resulting in objective calculated wrong
        # the model loading has no issue, the bug is here :)
        return tensor.mean(axis=dim)
    if strategy == "grpo":
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    if strategy == "dr.grpo":
        # Dr.GRPO modification 1: remove length bias by using a constant normalizer
        return (tensor * mask).sum(axis=-1) / generate_max_length
    # default: use grpo version
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss over a batch of sampled outputs."""

    def __init__(
        self, 
        clip_eps_low: float,
        clip_eps_high: float, 
        kl_weight: float, 
        policy_ops: str, 
        generate_max_length: int
    ) -> None:
        """kl_weight: beta in the DeepSeekMath paper."""
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.kl_weight = kl_weight
        self.policy_ops = policy_ops # policy optimization strategy
        self.generate_max_length = generate_max_length

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Follow DeepSeekMath GRPO formula, but with a twist on action_mask."""
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        # DeepSeekMath algorithm (and PPO-style algorithms) maximizes GRPO objective.
        # However, in PyTorch optimizer, Adam minimizes the loss. 
        # Thus, we switch the sign here.
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        # Compute single loss for the whole sampled batch of outputs.
        loss = masked_mean(
            loss, action_mask, 
            dim=-1, 
            strategy=self.policy_ops,
            generate_max_length=self.generate_max_length
        ).mean()

        # Compute KL divergence over the batch into 1 scalar to feed into optimizer.
        return loss, kl.mean()