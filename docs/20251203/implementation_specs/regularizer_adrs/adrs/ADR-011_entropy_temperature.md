# ADR-011: Entropy Temperature Option

## Status

Proposed

## Context

Entropy regularization currently supports `:mode`, `:reduction`, and `:normalize`. Training loops often need to **sharpen or soften** distributions before computing entropy:

1. **Temperature < 1.0** to emphasize confident outputs (sharper).
2. **Temperature > 1.0** to encourage exploration (flatter).
3. Maintain compatibility with log-space inputs and normalization.

## Decision

Add a `:temperature` option to `NxPenalties.Divergences.entropy/2`:

- `:temperature` – positive float, default `1.0`
- Implementation: divide logprobs by temperature, then re-normalize with `logsumexp` before computing entropy. Apply before normalization or mode flip.

### Interface

```elixir
entropy =
  NxPenalties.Divergences.entropy(logprobs,
    mode: :bonus,          # or :penalty
    reduction: :mean,
    normalize: false,
    temperature: 0.7       # <1 sharpens, >1 flattens
  )
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:mode` | `:bonus` \| `:penalty` | `:bonus` | Sign of the entropy contribution |
| `:reduction` | `:mean` \| `:sum` \| `:none` | `:mean` | Aggregation |
| `:normalize` | boolean | `false` | Divide by log(vocab) if true |
| `:temperature` | positive float | `1.0` | Scale logits/logprobs before entropy |

## Consequences

### Positive
- Enables exploration/exploitation tuning without external preprocessing.
- Keeps log-space stability by re-normalizing after scaling.
- Backward compatible: default preserves existing behavior.

### Negative
- Extra logsumexp adds minor compute cost.
- Misconfigured temperatures (< 0 or extreme) need validation guards.

### Neutral
- Can coexist with `:mode` and `:normalize`; combined effects must be documented in tests.

## Implementation Notes

1. Insert temperature handling ahead of entropy computation: `scaled = logprobs / temperature` → `normalized = scaled - logsumexp(scaled)`.
2. Add option validation (positive float) via NimbleOptions.
3. Tests:
   - `temperature: 1.0` parity with current outputs.
   - Temperature < 1 decreases entropy; > 1 increases entropy on a fixed distribution.
   - Works with `normalize: true` and both modes.
