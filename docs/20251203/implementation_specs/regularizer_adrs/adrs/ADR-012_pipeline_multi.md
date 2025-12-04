# ADR-012: Pipeline.Multi (Data-Aware Multi-Input Pipelines)

## Status

Proposed

## Context

The current `NxPenalties.Pipeline` is **single-tensor** and assumes penalties take one tensor plus opts. Several penalties and adapters need multiple inputs:

1. **KL/JS** require `(p_logprobs, q_logprobs)`.
2. **Consistency** can take paired outputs (`clean`, `noisy`).
3. **Interpolated gradient penalty** needs `(tensor, reference)`.
4. Integration layers (e.g., Tinkex) already manage data maps; a native multi-input pipeline would reduce glue code and error-prone argument shuffling.

## Decision

Introduce `NxPenalties.Pipeline.Multi`, a parallel composition engine for multi-arity penalties with named inputs.

### Core API

```elixir
defmodule NxPenalties.Pipeline.Multi do
  @type t :: %__MODULE__{
    entries: [{atom(), function(), [atom()], number() | Nx.Tensor.t(), keyword(), boolean()}],
    name: String.t() | nil,
    meta: map()
  }
end

# Builders
pipeline =
  NxPenalties.Pipeline.Multi.new(name: "data-aware")
  |> NxPenalties.Pipeline.Multi.add(
    :kl,
    &NxPenalties.Divergences.kl_divergence/3,
    inputs: [:p_logprobs, :q_logprobs],
    weight: 0.1,
    opts: [reduction: :mean]
  )
  |> NxPenalties.Pipeline.Multi.add(
    :consistency,
    &NxPenalties.Constraints.consistency/3,
    inputs: [:clean_out, :noisy_out],
    weight: 0.2
  )

# Execution
{total, metrics} =
  NxPenalties.Pipeline.Multi.compute(pipeline, %{
    p_logprobs: p,
    q_logprobs: q,
    clean_out: clean,
    noisy_out: noisy
  })
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:inputs` | list(atom) | required | Named inputs (order matches penalty fn arity) |
| `:weight` | number \| Nx.Tensor.t() | `1.0` | Scaling factor |
| `:opts` | keyword | `[]` | Passed to penalty fn |
| `:enabled` | boolean | `true` | Toggle inclusion |
| `:differentiable` | boolean | `true` | For gradient tracking compatibility |

### Behavior

- `compute/3` accepts a map `%{atom => tensor}` and raises on missing inputs.
- Metrics mirror single-tensor pipeline: `"name"`, `"name_weighted"`, `"total"`, plus optional grad norms.
- Compatible with `GradientTracker` when `:differentiable` is true.
- No breaking changes to existing `NxPenalties.pipeline/1`.

## Consequences

### Positive
- Eliminates ad-hoc tuple/map unpacking in callers.
- Enables declarative multi-input objectives (KL, consistency, interpolated GP).
- Preserves existing single-input pipeline for simple cases.

### Negative
- More surface area: additional struct, builders, compute path, docs, and tests.
- Slight overhead to resolve inputs by name at runtime.

### Neutral
- Keeps the “single tensor” guarantee for the existing pipeline; users opt into multi-input explicitly.

## Implementation Notes

1. Structure mirrors `NxPenalties.Pipeline` but entries carry `inputs :: [atom()]`.
2. `compute/3` resolves tensors from the provided map, validates presence and shape when feasible.
3. Consider `extra_args` merging semantics similar to single-tensor pipeline for shared opts.
4. Add tests for:
   - Missing input raises.
   - Multi-arity penalties (2-arg divergences, 3-arg consistency) compute correctly.
   - Enabled/disabled and weight updates.
   - Grad norm tracking respects `:differentiable`.
5. Document in specs/README and API reference; note interoperability with Tinkex adapters.
