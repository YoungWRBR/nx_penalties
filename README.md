<p align="center">
  <img src="assets/nx_penalties.svg" alt="NxPenalties" width="400">
</p>

<p align="center">
  <strong>Composable Regularization Penalties for Elixir ML</strong>
</p>

<p align="center">
  <a href="https://hex.pm/packages/nx_penalties">
    <img src="https://img.shields.io/hexpm/v/nx_penalties.svg" alt="Hex.pm Version">
  </a>
  <a href="https://hexdocs.pm/nx_penalties">
    <img src="https://img.shields.io/badge/docs-hexdocs-blue.svg" alt="Documentation">
  </a>
  <a href="https://github.com/North-Shore-AI/nx_penalties/actions">
    <img src="https://github.com/North-Shore-AI/nx_penalties/workflows/CI/badge.svg" alt="CI Status">
  </a>
  <a href="https://coveralls.io/github/North-Shore-AI/nx_penalties">
    <img src="https://coveralls.io/repos/github/North-Shore-AI/nx_penalties/badge.svg" alt="Coverage Status">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

---

## Overview

NxPenalties provides composable regularization functions for machine learning with [Nx](https://github.com/elixir-nx/nx). It fills the gap between [Axon](https://github.com/elixir-nx/axon) (model definition) and [Polaris](https://github.com/elixir-nx/polaris) (optimization) by providing the "missing middleware" for complex training objectives.

### Why NxPenalties?

- **Axon explicitly rejects model-level regularization** - Regularization belongs in training, not model definition
- **Polaris operates on gradients only** - Cannot see intermediate activations needed for activity regularization
- **Scholar is traditional ML only** - Regularization tied to specific estimators, not composable

NxPenalties provides standalone, JIT-compilable penalty functions that work with any Nx-based training loop.

### Features

- **L1 (Lasso)** - Sparse weight regularization via absolute values
- **L2 (Ridge)** - Weight decay via squared norms
- **Elastic Net** - Combined L1+L2 with tunable ratio
- **KL Divergence** - Distribution matching for knowledge distillation
- **JS Divergence** - Symmetric KL for distribution comparison
- **Entropy** - Encourage/discourage prediction confidence
- **Pipeline Composition** - Combine multiple penalties with individual weights
- **Gradient-Compatible** - Full autodiff support for training
- **Telemetry Integration** - Built-in metrics emission

## Installation

Add `nx_penalties` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx_penalties, "~> 0.1.0"}
  ]
end
```

## Quick Start

### Simple Penalties

```elixir
# L1 penalty (promotes sparsity)
l1_loss = NxPenalties.l1(weights)
# => Nx.tensor(6.5)

# L2 penalty (weight decay)
l2_loss = NxPenalties.l2(weights, lambda: 0.01)
# => Nx.tensor(0.1425)

# Elastic Net (combined L1 + L2)
elastic_loss = NxPenalties.elastic_net(weights, l1_ratio: 0.5)
# => Nx.tensor(10.375)

# Add to your training loss
total_loss = Nx.add(base_loss, l1_loss)
```

### Pipeline Composition (Recommended)

For production use, compose multiple penalties with individual weights:

```elixir
# Create a pipeline with multiple penalties
pipeline = NxPenalties.pipeline([
  {:l1, weight: 0.001},           # Sparsity
  {:l2, weight: 0.01},            # Weight decay
  {:entropy, weight: 0.1, mode: :bonus}  # Exploration bonus
])

# Compute all penalties at once
{total_penalty, metrics} = NxPenalties.compute(pipeline, model_outputs)

# metrics contains:
# %{
#   "l1" => 8.5,
#   "l1_weighted" => 0.0085,
#   "l2" => 18.25,
#   "l2_weighted" => 0.1825,
#   "entropy" => ...,
#   "entropy_weighted" => ...,
#   "total" => 0.191...
# }

# Use in training
total_loss = Nx.add(base_loss, total_penalty)
```

### Dynamic Weight Adjustment

Useful for curriculum learning or adaptive regularization:

```elixir
# Update weights during training
pipeline =
  pipeline
  |> NxPenalties.Pipeline.update_weight(:l1, 0.01)  # Increase L1
  |> NxPenalties.Pipeline.update_weight(:l2, 0.001) # Decrease L2

# Enable/disable penalties
pipeline = NxPenalties.Pipeline.set_enabled(pipeline, :entropy, false)
```

### Gradient-Compatible Computation

For use inside `defn` or when you need gradients to flow through:

```elixir
# Returns tensor (not converted to number) - gradient compatible
total = NxPenalties.compute_total(pipeline, tensor)

# Works with Nx.Defn.grad
grad_fn = Nx.Defn.grad(fn t ->
  NxPenalties.compute_total(pipeline, t)
end)
gradients = grad_fn.(tensor)
```

## Divergences

For probability distributions (log-space inputs):

```elixir
# KL Divergence - knowledge distillation
kl_loss = NxPenalties.kl_divergence(student_logprobs, teacher_logprobs)

# JS Divergence - symmetric comparison
js_loss = NxPenalties.js_divergence(p_logprobs, q_logprobs)

# Entropy - encourage/discourage confidence
entropy_penalty = NxPenalties.entropy(logprobs, mode: :penalty)  # Minimize entropy
entropy_bonus = NxPenalties.entropy(logprobs, mode: :bonus)      # Maximize entropy
```

## Axon Integration

Wrap your loss function with regularization:

```elixir
alias NxPenalties.Integration.Axon, as: AxonIntegration

# Create penalty pipeline
pipeline = NxPenalties.pipeline([
  {:l2, weight: 0.01}
])

# Wrap loss function
regularized_loss = AxonIntegration.wrap_loss_with_pipeline(
  &Axon.Losses.mean_squared_error/2,
  pipeline
)

# Use in training
model
|> Axon.Loop.trainer(regularized_loss, optimizer)
|> Axon.Loop.run(data, epochs: 10)
```

## API Reference

### Penalty Functions

| Function | Description | Options |
|----------|-------------|---------|
| `l1/2` | L1 norm (Lasso) | `lambda`, `reduction` |
| `l2/2` | L2 norm squared (Ridge) | `lambda`, `reduction` |
| `elastic_net/2` | Combined L1+L2 | `lambda`, `l1_ratio`, `reduction` |

### Divergence Functions

| Function | Description | Options |
|----------|-------------|---------|
| `kl_divergence/3` | KL(P \|\| Q) | `reduction` |
| `js_divergence/3` | Jensen-Shannon | `reduction` |
| `entropy/2` | Shannon entropy | `mode`, `reduction`, `normalize` |

### Pipeline Functions

| Function | Description |
|----------|-------------|
| `pipeline/1` | Create pipeline from keyword list |
| `compute/3` | Execute pipeline, return `{total, metrics}` |
| `compute_total/3` | Execute pipeline, return tensor only (gradient-safe) |
| `Pipeline.add/4` | Add penalty to pipeline |
| `Pipeline.update_weight/3` | Change penalty weight |
| `Pipeline.set_enabled/3` | Enable/disable penalty |

### Constraints

| Function | Description | Options |
|----------|-------------|---------|
| `Constraints.orthogonality/2` | Decorrelation penalty | `mode`, `normalize` |
| `Constraints.consistency/3` | Paired output consistency | `metric`, `reduction` |

## Telemetry Events

NxPenalties emits telemetry events for monitoring:

```elixir
# Attach handler
:telemetry.attach(
  "nx-penalties-logger",
  [:nx_penalties, :pipeline, :compute, :stop],
  fn _event, measurements, metadata, _config ->
    Logger.info("Pipeline computed in #{measurements.duration}ns")
    Logger.info("Metrics: #{inspect(metadata.metrics)}")
  end,
  nil
)
```

| Event | Measurements | Metadata |
|-------|-------------|----------|
| `[:nx_penalties, :pipeline, :compute, :start]` | `system_time` | `size` |
| `[:nx_penalties, :pipeline, :compute, :stop]` | `duration` | `metrics`, `total` |

## Performance

All penalty functions are implemented using `Nx.Defn` for JIT compilation:

- **GPU acceleration** - Automatically uses EXLA/CUDA when available
- **Fused operations** - Penalties compose efficiently in the computation graph
- **Minimal overhead** - No runtime option parsing in hot path

## Testing

```bash
# Run tests
mix test

# Run with coverage
mix coveralls.html

# Run quality checks
mix quality  # format + credo + dialyzer
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.exs` - Simple penalty functions
- `pipeline_composition.exs` - Pipeline creation and manipulation
- `curriculum_learning.exs` - Dynamic weight adjustment
- `axon_training.exs` - Axon integration

Run examples with:

```bash
mix run examples/basic_usage.exs
```

## Roadmap

### v0.1 (Current)
- Core penalties: L1, L2, Elastic Net
- Divergences: KL, JS, Entropy
- Pipeline composition
- Axon integration
- Telemetry

### (Planned)
- Constraints: orthogonality, consistency (full implementation)
- Gradient tracking / norm monitoring
- Polaris gradient transforms
- `Pipeline.Multi` for named multi-tensor inputs
- Livebook examples

### (Future)
- Activity regularization via layer capture
- Auxiliary loss infrastructure
- Multi-head output support

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs to the `main` branch.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Ensure all checks pass (`mix quality && mix test`)
5. Submit a pull request

## License

MIT License - Copyright (c) 2025 North-Shore-AI

See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Nx](https://github.com/elixir-nx/nx) - Numerical computing for Elixir
- [Axon](https://github.com/elixir-nx/axon) - Neural network library
- [Polaris](https://github.com/elixir-nx/polaris) - Gradient optimization
