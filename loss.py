from typing import Optional
import torch
import torch.nn as nn


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor]
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
    logr = log_probs - log_probs_ref # log ratio
    if action_mask is not None:
        logr = logr * action_mask
    return (logr.exp() - 1) - logr # r - 1 - logr
