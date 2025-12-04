# NxPenalties Examples

Examples demonstrating NxPenalties regularization features.

## Running Examples

Run individual examples:

```bash
mix run examples/basic_usage.exs
mix run examples/pipeline_composition.exs
mix run examples/axon_training.exs
mix run examples/axon_full_integration.exs
mix run examples/curriculum_learning.exs
mix run examples/gradient_tracking.exs
mix run examples/polaris_integration.exs
mix run examples/polaris_full_integration.exs
mix run examples/constraints.exs
mix run examples/entropy_normalization.exs
mix run examples/gradient_penalty.exs
```

Run all examples:

```bash
./examples/run_all.sh
```

## Examples

### basic_usage.exs

Core penalty functions: L1, L2, and Elastic Net with different lambda values and reduction modes.

### pipeline_composition.exs

Building penalty pipelines with multiple regularizers, dynamic weight adjustment, enable/disable controls, and gradient-compatible computation.

### axon_training.exs

Basic Axon integration. Shows how to add regularization to training loops.

Requires Axon (`{:axon, "~> 0.6"}`) - falls back to conceptual example if unavailable.

### axon_full_integration.exs

Comprehensive Axon integration demonstrating all patterns:
- **Pattern 1**: Simple loss wrapping with `wrap_loss/3`
- **Pattern 2**: Pipeline-based loss with `wrap_loss_with_pipeline/3`
- **Pattern 3**: Custom training step with `build_train_step/4` for full metrics access
- **Pattern 4**: Weight decay on model parameters with `build_train_step_with_weight_decay/5`
- **Pattern 5**: Curriculum learning with `weight_schedule/1` and dynamic weight adjustment
- **Pattern 6**: Activity regularization with `regularize_activity/4`
- **Pattern 7**: Complete training loop with all features combined

Requires Axon (`{:axon, "~> 0.6"}`).

### curriculum_learning.exs

Dynamic penalty scheduling for curriculum learning:
- Decreasing regularization over epochs
- Phase-based training with different penalty configurations
- Elastic Net ratio shifting
- Gradient flow verification

### gradient_tracking.exs

Monitor gradient norms from regularization penalties:
- Enable `track_grad_norms: true` in pipeline compute
- Per-penalty gradient norm metrics
- Direct GradientTracker usage
- Tensor validation with `NxPenalties.validate/1`

### polaris_integration.exs

Basic Polaris integration with gradient-level transforms:
- L2 weight decay (AdamW-style)
- L1 weight decay (sparsity)
- Elastic Net decay
- Composing multiple transforms

### polaris_full_integration.exs

Comprehensive Polaris integration demonstrating all transforms:
- **L2 Weight Decay**: Decoupled weight decay (AdamW style)
- **L1 Weight Decay**: Sign-based decay for sparsity
- **Elastic Net Decay**: Combined L1+L2 regularization
- **Gradient Clipping**: Global norm clipping to prevent explosion
- **Gradient Noise**: Decaying Gaussian noise for regularization
- **Adaptive Gradient Clipping (AGC)**: Scale-aware clipping per parameter
- **Gradient Centralization**: Zero-mean gradients for stability
- **Composition**: Stacking multiple transforms via piping

### constraints.exs

Structural constraint penalties:
- Orthogonality penalty for decorrelating representations
- Consistency penalty for paired output stability
- Different metrics (MSE, L1, cosine)
- Soft vs hard modes

### entropy_normalization.exs

Entropy regularization with optional normalization:
- Bonus vs penalty modes
- Normalized entropy scaled to [0, 1]
- Impact on pipelines and metrics

### gradient_penalty.exs

Gradient penalty primitives and cheaper proxies:
- Full gradient norm penalty via `gradient_penalty/3`
- WGAN-GP style interpolated penalty
- Output magnitude proxy for performance
