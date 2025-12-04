# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-03

### Added

- **Core Penalties**
  - `NxPenalties.l1/2` - L1 (Lasso) regularization with lambda scaling and reduction options
  - `NxPenalties.l2/2` - L2 (Ridge) regularization with lambda scaling and reduction options
  - `NxPenalties.elastic_net/2` - Combined L1+L2 with tunable l1_ratio parameter

- **Divergences**
  - `NxPenalties.kl_divergence/3` - KL divergence for log-probability distributions
  - `NxPenalties.js_divergence/3` - Jensen-Shannon divergence (symmetric KL)
  - `NxPenalties.entropy/2` - Shannon entropy with penalty/bonus modes and normalization

- **Pipeline Composition**
  - `NxPenalties.pipeline/1` - Create pipelines from keyword list specs
  - `NxPenalties.compute/3` - Execute pipeline returning `{total, metrics}`
  - `NxPenalties.compute_total/3` - Gradient-compatible computation (returns tensor)
  - `NxPenalties.Pipeline.add/4` - Add penalties to pipeline
  - `NxPenalties.Pipeline.update_weight/3` - Dynamic weight adjustment
  - `NxPenalties.Pipeline.set_enabled/3` - Enable/disable penalties at runtime

- **Constraints (v0.2 Preview)**
  - `NxPenalties.Constraints.orthogonality/2` - Decorrelation penalty with soft/hard modes
  - `NxPenalties.Constraints.consistency/3` - Paired output consistency with MSE/L1/cosine metrics

- **Gradient Tracking (v0.2 Preview)**
  - `NxPenalties.GradientTracker.compute_grad_norm/2` - Compute gradient L2 norms
  - `NxPenalties.GradientTracker.pipeline_grad_norms/2` - Per-penalty gradient norms
  - `NxPenalties.GradientTracker.total_grad_norm/2` - Total pipeline gradient norm

- **Integrations**
  - `NxPenalties.Integration.Axon.wrap_loss/3` - Wrap loss function with penalty
  - `NxPenalties.Integration.Axon.wrap_loss_with_pipeline/3` - Wrap loss with pipeline
  - `NxPenalties.Integration.Polaris` - Stub for v0.2 gradient transforms

- **Telemetry**
  - `[:nx_penalties, :pipeline, :compute, :start]` event
  - `[:nx_penalties, :pipeline, :compute, :stop]` event with metrics

- **Examples**
  - `examples/basic_usage.exs` - Simple penalty function demos
  - `examples/pipeline_composition.exs` - Pipeline creation and manipulation
  - `examples/curriculum_learning.exs` - Dynamic weight adjustment patterns
  - `examples/axon_training.exs` - Axon integration example

### Technical Notes

- All penalty functions use `Nx.Defn` with `deftransform`/`defnp` pattern for JIT compatibility
- Log-space operations in divergences for numerical stability
- Full gradient flow support through `compute_total/3`
- 73 tests covering penalties, divergences, and pipeline operations

[Unreleased]: https://github.com/North-Shore-AI/nx_penalties/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/North-Shore-AI/nx_penalties/releases/tag/v0.1.0
