# NxPenalties Examples

Examples demonstrating NxPenalties regularization features.

## Running Examples

Run individual examples:

```bash
mix run examples/basic_usage.exs
mix run examples/pipeline_composition.exs
mix run examples/axon_training.exs
mix run examples/curriculum_learning.exs
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

Integration with Axon neural networks. Shows how to add regularization to training loops.

Requires Axon (`{:axon, "~> 0.6"}`) - falls back to conceptual example if unavailable.

### curriculum_learning.exs

Dynamic penalty scheduling for curriculum learning:
- Decreasing regularization over epochs
- Phase-based training with different penalty configurations
- Elastic Net ratio shifting
- Gradient flow verification
