# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Exact Optimal Baseline (EOB) implementation.

This module implements the theoretically optimal baseline for policy gradient variance reduction:

    b* = E[||∇_θ log π(τ)||² · R(τ)] / E[||∇_θ log π(τ)||²]

where ||∇_θ log π(τ)||² is computed EXACTLY via backward passes (not approximated).

Key components:
1. compute_gradient_norms_for_batch: Computes exact gradient norms in actor worker
2. compute_exact_optimal_baseline_advantage: Computes advantages using exact gradient norms
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def compute_gradient_norms_for_batch(
    model: nn.Module,
    forward_fn: Callable,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    multi_modal_inputs: dict | None = None,
) -> torch.Tensor:
    """
    Compute exact gradient norms squared ||∇_θ log π(τ)||² for a batch of trajectories.

    This function computes the gradient norm for each trajectory by:
    1. Forward pass to get log probabilities
    2. Sum log probs over sequence: log π(τ) = Σ_t log π(a_t|s_t)
    3. Backward pass to compute ∇_θ log π(τ)
    4. Compute ||∇||² = Σ_param ||∇_param||²

    IMPORTANT: This requires one forward + backward pass per sample, so it's N times
    more expensive than a standard forward pass (where N is batch size).

    Args:
        model: The policy model (actor). Must be the actual nn.Module, not wrapped.
        forward_fn: A function that takes (model, input_ids, attention_mask, position_ids, ...)
                    and returns logits of shape [batch, seq_len, vocab_size].
        input_ids: Token IDs [shape: (bs, seq_len)]
        attention_mask: Attention mask [shape: (bs, seq_len)]
        position_ids: Position IDs [shape: (bs, seq_len)]
        responses: Response token IDs [shape: (bs, response_len)]
        response_mask: Binary mask for valid response tokens [shape: (bs, response_len)]
        temperature: Temperature for log probability computation
        multi_modal_inputs: Optional multi-modal inputs (for VLMs)

    Returns:
        gradient_norms_squared: ||∇_θ log π(τ)||² for each trajectory [shape: (bs,)]

    Example:
        >>> # In actor worker:
        >>> grad_norms_sq = compute_gradient_norms_for_batch(
        ...     model=self.actor_module,
        ...     forward_fn=my_forward_fn,
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     position_ids=position_ids,
        ...     responses=responses,
        ...     response_mask=response_mask,
        ...     temperature=1.0,
        ... )
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    response_len = responses.shape[1]

    # Store original training mode and set to train for gradient computation
    was_training = model.training
    model.train()

    gradient_norms_squared = torch.zeros(batch_size, device=device)

    # Process each sample individually
    for i in range(batch_size):
        # Zero all gradients before processing this sample
        model.zero_grad()

        # Extract single sample (keep batch dimension for model compatibility)
        single_input_ids = input_ids[i : i + 1]  # [1, seq_len]
        single_attention_mask = attention_mask[i : i + 1]  # [1, seq_len]
        single_position_ids = position_ids[i : i + 1]  # [1, seq_len]
        single_response = responses[i : i + 1]  # [1, response_len]
        single_response_mask = response_mask[i : i + 1]  # [1, response_len]

        # Prepare multi-modal inputs if present
        single_mm_inputs = None
        if multi_modal_inputs is not None:
            single_mm_inputs = {}
            for key, value in multi_modal_inputs.items():
                if isinstance(value, torch.Tensor):
                    single_mm_inputs[key] = value[i : i + 1]
                elif isinstance(value, list):
                    single_mm_inputs[key] = [value[i]]
                else:
                    single_mm_inputs[key] = value

        # Forward pass WITH gradient tracking
        # Note: We need the computation graph, so no torch.no_grad() here
        if single_mm_inputs is not None:
            logits = forward_fn(
                model,
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                position_ids=single_position_ids,
                **single_mm_inputs,
            )
        else:
            logits = forward_fn(
                model,
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                position_ids=single_position_ids,
            )

        # logits shape: [1, seq_len, vocab_size]
        # Extract response logits (last response_len tokens)
        response_logits = logits[:, -response_len:, :]  # [1, response_len, vocab_size]

        # Apply temperature
        response_logits = response_logits / temperature

        # Compute log probabilities for the actual tokens
        log_probs_all = torch.log_softmax(response_logits, dim=-1)  # [1, response_len, vocab_size]

        # Gather log probs for the actual response tokens
        # single_response: [1, response_len]
        log_probs = torch.gather(
            log_probs_all, dim=-1, index=single_response.unsqueeze(-1)
        ).squeeze(-1)  # [1, response_len]

        # Compute log π(τ) = Σ_t log π(a_t|s_t) for valid tokens only
        log_prob_trajectory = (log_probs * single_response_mask).sum()  # scalar

        # Backward pass to compute ∇_θ log π(τ)
        log_prob_trajectory.backward()

        # Compute ||∇_θ log π(τ)||² = Σ_param ||∇_param||²
        # OPTIMIZED: Keep computation on GPU, avoid repeated .item() calls
        grad_norm_sq = torch.tensor(0.0, device=device)
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm_sq = grad_norm_sq + param.grad.pow(2).sum()

        gradient_norms_squared[i] = grad_norm_sq

        # Clear gradients for next sample
        model.zero_grad()

    # Restore original training mode
    model.train(was_training)

    return gradient_norms_squared


def compute_exact_optimal_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gradient_norms_squared: torch.Tensor,
    epsilon: float = 1e-8,
    min_grad_norm_threshold: float = 1e-10,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using EXACT Optimal Baseline (EOB).

    This implements the theoretically optimal baseline formula:

        b* = Σ[||∇_θ log π(τᵢ)||² · R(τᵢ)] / Σ[||∇_θ log π(τᵢ)||²]

    where ||∇_θ log π(τᵢ)||² is pre-computed exactly via backward passes.

    Theory:
        For each prompt group, the optimal baseline that minimizes the variance
        of the policy gradient estimator is the weighted average of rewards,
        where weights are the gradient norms squared.

        This is provably optimal because trajectories with larger gradients
        contribute more to the variance of the gradient estimator.

    Key differences from approximation methods:
        - GRPO: Uses uniform weighting (W = 1)
        - OPO: Uses length as proxy (W ≈ length)
        - OTB/SOB: Uses logit-gradient proxy (W ≈ Σ(1 - 2π + Σπ²))
        - EOB: Uses EXACT gradient norm (W = ||∇log π||²)

    Args:
        token_level_rewards: Rewards at each token [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        gradient_norms_squared: Pre-computed ||∇_θ log π(τ)||² [shape: (bs,)]
        epsilon: Small constant for numerical stability (default: 1e-8)
        min_grad_norm_threshold: Minimum gradient norm threshold. If all gradients
            in a group are below this, falls back to uniform weighting (default: 1e-10)
        **kwargs: Additional arguments for compatibility (unused)

    Returns:
        advantages: EOB advantage estimates [shape: (bs, response_length)]
        returns: Total rewards broadcasted to token level [shape: (bs, response_length)]

    Note:
        gradient_norms_squared must be computed in the actor worker using
        compute_gradient_norms_for_batch() before calling this function.
    """
    # Suppress unused kwargs warning
    _ = kwargs

    with torch.no_grad():
        batch_size, _ = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute total reward per sequence: R(τ) = Σ_t r_t
        rewards = (token_level_rewards * response_mask).sum(dim=-1)  # [bs]

        # Group trajectories by prompt index
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines (one per sequence)
        baselines = torch.zeros(batch_size, device=device)

        # Compute optimal baseline for each prompt group
        for prompt_id, trajectory_indices in prompt_groups.items():
            n_trajectories = len(trajectory_indices)

            if n_trajectories == 1:
                # Single trajectory: baseline = 0 (no variance reduction possible)
                baselines[trajectory_indices[0]] = 0.0
                continue

            # Convert to tensor for indexing
            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data
            rewards_group = rewards[traj_idx]  # [N]
            grad_norms_sq_group = gradient_norms_squared[traj_idx]  # [N]

            # Check if gradient norms are too small (numerical instability)
            total_grad_norm = grad_norms_sq_group.sum()

            if total_grad_norm < min_grad_norm_threshold:
                # Fall back to uniform weighting (GRPO-style) if gradients are too small
                # This prevents numerical instability when all gradient norms ≈ 0
                optimal_baseline = rewards_group.mean()
            else:
                # Compute EXACT optimal baseline:
                # b* = Σ[||∇||² · R] / Σ[||∇||²]
                numerator = (grad_norms_sq_group * rewards_group).sum()
                denominator = total_grad_norm + epsilon
                optimal_baseline = numerator / denominator

            # Assign same baseline to all trajectories in this group
            baselines[traj_idx] = optimal_baseline

        # Compute advantages: A(τ) = R(τ) - b*
        advantages_per_sequence = rewards - baselines  # [bs]

        # Broadcast to token level (same advantage for all tokens in a sequence)
        advantages = advantages_per_sequence.unsqueeze(-1) * response_mask  # [bs, response_len]
        returns = rewards.unsqueeze(-1) * response_mask  # [bs, response_len]

    return advantages, returns


# =============================================================================
# Theoretical Background
# =============================================================================
#
# Policy Gradient Variance Reduction with Optimal Baseline
# --------------------------------------------------------
#
# The REINFORCE gradient estimator is:
#
#     ĝ(τ) = ∇_θ log π_θ(τ) · (R(τ) - b)
#
# where τ is a trajectory, R(τ) is the return, and b is a baseline.
#
# The variance of this estimator (across trajectories) is:
#
#     Var(ĝ) = E[||∇log π(τ)||² · (R(τ) - b)²]
#
# To find the optimal baseline b*, we take derivative w.r.t. b and set to 0:
#
#     d/db Var(ĝ) = -2 · E[||∇log π(τ)||² · (R(τ) - b)] = 0
#
# Solving for b*:
#
#     b* = E[||∇log π(τ)||² · R(τ)] / E[||∇log π(τ)||²]
#
# This is a weighted average of rewards, weighted by gradient norms squared.
#
# Intuition:
# ----------
# - Trajectories with larger ||∇log π||² contribute more to gradient variance
# - We should subtract more of their reward to reduce their contribution
# - The optimal baseline does exactly this: it's weighted toward high-gradient trajectories
#
# Comparison with Approximations:
# -------------------------------
# 1. GRPO: Assumes ||∇log π(τ)||² = const for all τ
#    → b = E[R(τ)] (simple mean)
#
# 2. OPO: Assumes ||∇log π(τ)||² ∝ length(τ)
#    → b = Σ(length × R) / Σ(length)
#    Reference: https://arxiv.org/pdf/2505.23585
#
# 3. OTB/SOB: Uses logit-gradient proxy
#    → ||∇log π(τ)||² ≈ Σ_t (1 - 2π_t + Σπ_t²)
#    This is derived from variance of score function
#
# 4. EOB (this implementation): Computes exact gradient
#    → b* = Σ(||∇||² × R) / Σ(||∇||²) with actual ||∇||²
#
# =============================================================================
