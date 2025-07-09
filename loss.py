from typing import Optional
import torch
import torch.nn as nn


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
) -> torch.Tensor:
    """
    Compute the mean over `dim`, ignoring elements where `mask==0`.
    
    Args:
        tensor (torch.Tensor): Input tensor whose mean to be computed
        mask (Optional[torch.Tensor]): Binary mask same shape as tensor (1=keep, 0=ignore)
        dim (int or tuple of ints, optional): 
            Dimension(s) along which to compute the mean.
            If 'None', mean is taken over all elements in tensor.
        
    Returns:
        torch.Tensor: The masked mean of `tensor` along `dim`.
    """
    if mask is None:
        return torch.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
