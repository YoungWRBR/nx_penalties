# ADR-010: KL Direction & Symmetric Options

## Status

Proposed

## Context

KL divergence in NxPenalties currently assumes forward direction KL(P‖Q) with a `:reduction` option only. Downstream training loops need finer control:

1. **Reverse KL** (KL(Q‖P)) for mode-covering behavior in generative models.
2. **Symmetric KL** to approximate Jensen-Shannon without switching functions.
3. **Adapter parity** – Tinkex adapters already surface reference tensors; they should be able to toggle direction without swapping arguments manually.

## Decision

Extend `NxPenalties.Divergences.kl_divergence/3` with:

- `:direction` – `:forward` (default) or `:reverse`
- `:symmetric` – boolean flag; when `true`, compute 0.5 * (KL(P‖Q) + KL(Q‖P)) and ignore `:direction`

### Interface

```elixir
# Tensor primitive
kl = NxPenalties.Divergences.kl_divergence(p_logprobs, q_logprobs,
  reduction: :mean,
  direction: :forward,   # or :reverse
  symmetric: false       # true => average forward + reverse
)

# Tinkex adapter (data-aware)
defmodule Tinkex.Regularizers.KLDivergence do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    reference = resolve_reference!(data, opts)
    direction = Keyword.get(opts, :direction, :forward)
    symmetric = Keyword.get(opts, :symmetric, false)
    reduction = Keyword.get(opts, :reduction, :mean)

    value =
      NxPenalties.Divergences.kl_divergence(
        logprobs,
        reference,
        reduction: reduction,
        direction: direction,
        symmetric: symmetric
      )

    {value,
     %{
       "kl_direction" => Atom.to_string(direction),
       "kl_symmetric" => symmetric
     }}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:reduction` | `:mean` \| `:sum` \| `:none` | `:mean` | Aggregation method |
| `:direction` | `:forward` \| `:reverse` | `:forward` | Whether to compute KL(P‖Q) or KL(Q‖P) |
| `:symmetric` | boolean | `false` | If true, compute 0.5 * (KL(P‖Q) + KL(Q‖P)) |

## Consequences

### Positive
- Matches use-cases that need reverse KL or symmetric pressure.
- Avoids argument-swapping boilerplate in adapters and pipelines.
- Backward compatible defaults preserve current behavior.

### Negative
- Symmetric mode doubles computation (two KL evaluations).
- More option combinations to validate and test.

### Neutral
- Jensen-Shannon remains available via `js_divergence/3`; symmetric KL provides a lighter-weight alternative when JS isn’t wired into pipelines.

## Implementation Notes

1. Update `Divergences.kl_divergence/3` to pattern match on `{symmetric, direction}` and reuse existing `kl_none_impl/2`.
2. Add NimbleOptions validation for `:direction` and `:symmetric`.
3. Tests:
   - Symmetric equals mean of forward/reverse KL within tolerance.
   - Reverse matches swapping inputs of forward.
   - Defaults remain unchanged.
4. Pipeline/pipeline.fn_for: no change required; options flow through `opts`.
