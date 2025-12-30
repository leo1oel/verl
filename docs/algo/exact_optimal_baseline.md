# Exact Optimal Baseline (EOB)

Last updated: 2025-12-29.

Exact Optimal Baseline (EOB) implements the **theoretically optimal** baseline for policy gradient variance reduction:

```
b* = E[||∇_θ log π(τ)||² · R(τ)] / E[||∇_θ log π(τ)||²]
```

This is the **exact** computation using actual backward passes, not an approximation.

## Theory: Why This is Optimal

In policy gradient methods, the gradient estimator is:

```
ĝ(τ) = ∇_θ log π_θ(τ) · (R(τ) - b)
```

The variance of this estimator is:

```
Var(ĝ) = E[||∇log π(τ)||² · (R(τ) - b)²]
```

The baseline `b*` that **minimizes** this variance is:

```
b* = E[||∇log π(τ)||² · R(τ)] / E[||∇log π(τ)||²]
```

This is a **weighted average** of rewards, where the weights are the gradient norms squared. Trajectories with larger gradients get more weight because they contribute more to the variance.

## Comparison with Approximations

| Method | Weight Approximation | Accuracy | Extra Cost |
|--------|---------------------|----------|------------|
| **GRPO** | `||∇||² ≈ 1` (uniform) | Baseline | 0 |
| **OPO** | `||∇||² ≈ length(τ)` | Good | 0 |
| **OTB** | `||∇||² ≈ Σ(1 - 2π + Σπ²)` | Better | ~0 |
| **EOB** | `||∇||² = exact` | **Exact (optimal)** | **N × (fwd + bwd)** |

## How It Works

EOB requires computing the exact gradient norm for each trajectory. This is done in two steps:

### Step 1: Compute Gradient Norms (in Actor Worker)

For each trajectory τᵢ in the batch:
1. Forward pass to compute log probabilities
2. Sum log probs: `log π(τ) = Σ_t log π(a_t|s_t)`
3. Backward pass to compute `∇_θ log π(τ)`
4. Compute norm: `||∇||² = Σ_param ||∇_param||²`

### Step 2: Compute Optimal Baseline (in Trainer)

For each prompt group:
```python
b* = Σ[||∇||² × R] / Σ[||∇||²]
advantage = R - b*
```

## Usage

Simply set the config:

```yaml
algorithm:
  adv_estimator: exact_optimal_baseline

actor_rollout_ref:
  rollout:
    n: 8  # Group sampling required for variance reduction
```

The trainer will automatically:
1. Call `compute_gradient_norms()` on the actor worker
2. Use the gradient norms to compute the optimal baseline

## Implementation Files

- `verl/trainer/ppo/exact_optimal_baseline.py`: Core implementation
  - `compute_gradient_norms_for_batch()`: Computes ||∇||² for each trajectory
  - `compute_exact_optimal_baseline_advantage()`: Computes optimal baseline

- `verl/workers/actor/dp_actor.py`: Actor method
  - `compute_gradient_norms()`: Wrapper for actor worker

- `verl/workers/fsdp_workers.py`: Worker interface
  - `compute_gradient_norms()`: Distributed interface

- `verl/trainer/ppo/ray_trainer.py`: Trainer integration
  - Calls gradient norm computation before advantage computation

## Computational Cost

**Per batch:**
- N forward passes (one per sample)
- N backward passes (one per sample)
- Where N = batch_size

**Estimated slowdown:**
- ~2N times slower than GRPO for gradient computation phase
- For `rollout.n=8`, expect 8× overhead in the advantage computation phase

## When to Use EOB

**Good use cases:**
- Research experiments to verify theoretical variance reduction
- When you need the provably optimal baseline
- Comparing baseline methods
- When variance is the main training bottleneck

**Not recommended when:**
- Large-scale production training
- Time/cost is a constraint
- Approximations (OTB, OPO) already work well for your task

## Example

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=exact_optimal_baseline \
    actor_rollout_ref.rollout.n=8 \
    data.train_files=YOUR_DATA \
    actor_rollout_ref.model.path=YOUR_MODEL
```

## Theoretical Guarantee

EOB provably gives the **minimum variance** among all scalar baselines. The proof:

1. The variance of the gradient estimator is:
   `Var(ĝ) = E[||∇log π||² · (R - b)²]`

2. Taking derivative w.r.t. b and setting to 0:
   `d/db Var(ĝ) = -2 · E[||∇log π||² · (R - b)] = 0`

3. Solving for b*:
   `b* = E[||∇log π||² · R] / E[||∇log π||²]`

This is exactly what EOB computes.

## References

- [Variance Reduction Techniques for Gradient Estimates (Greensmith et al.)](https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf)
- [The Role of Baselines in Policy Gradient Optimization](https://arxiv.org/abs/2301.06276)
- [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/pdf/2505.23585)
