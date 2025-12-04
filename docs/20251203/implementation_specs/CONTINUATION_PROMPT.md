# NxPenalties v0.2 Implementation Continuation Prompt

## Context

You are continuing development of **NxPenalties**, a standalone Elixir library providing composable regularization penalties for the Nx ecosystem. Version 0.1.0 is complete and all tests pass. Your task is to complete the v0.2 features.

## Project Location

```
/home/home/p/g/North-Shore-AI/nx_penalties
```

## Required Reading (Read These First)

Before making any changes, read these files in order:

### 1. Implementation Specifications
```
docs/20251203/implementation_specs/README.md           # Overview and phases
docs/20251203/implementation_specs/00_ARCHITECTURE.md  # Core architecture
docs/20251203/implementation_specs/08_API_REFERENCE.md # Complete API spec
docs/20251203/implementation_specs/07_TEST_STRATEGY.md # TDD approach
docs/20251203/implementation_specs/09_NUMERICAL_STABILITY.md # Stability patterns
```

### 2. v0.2 Feature Specs
```
docs/20251203/implementation_specs/03_CONSTRAINTS.md        # Orthogonality, gradient penalty
docs/20251203/implementation_specs/06_POLARIS_INTEGRATION.md # Gradient transforms
docs/20251203/implementation_specs/11_GRADIENT_TRACKING.md   # Gradient norm monitoring
```

### 3. ADRs (Architecture Decision Records)
```
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-007_gradient_penalty.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-008_orthogonality.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-009_gradient_tracking.md
```

### 4. Current Implementation
```
lib/nx_penalties.ex                        # Main entry module
lib/nx_penalties/penalties.ex              # L1, L2, Elastic Net
lib/nx_penalties/divergences.ex            # KL, JS, Entropy
lib/nx_penalties/pipeline.ex               # Pipeline composition
lib/nx_penalties/constraints.ex            # Orthogonality, consistency (partial)
lib/nx_penalties/gradient_tracker.ex       # Gradient norms (partial)
lib/nx_penalties/integration/axon.ex       # Axon helpers
lib/nx_penalties/integration/polaris.ex    # Polaris stubs
lib/nx_penalties/telemetry.ex              # Telemetry events
```

### 5. Tests
```
test/nx_penalties/penalties_test.exs       # 28 tests
test/nx_penalties/divergences_test.exs     # 22 tests
test/nx_penalties/pipeline_test.exs        # 23 tests
test/support/test_helpers.ex               # Test utilities
```

## Current State (v0.1.0 Complete)

### Implemented and Passing
- Core penalties: L1, L2, Elastic Net ✅
- Divergences: KL, JS, Entropy ✅
- Pipeline composition with weights ✅
- Dynamic weight adjustment ✅
- Enable/disable penalties ✅
- Gradient-compatible `compute_total/3` ✅
- Axon integration helpers ✅
- Telemetry events ✅
- 73 tests passing ✅

### Verification Status
```bash
mix test                        # 73 tests, 0 failures
mix compile --warnings-as-errors # Compiles clean
mix credo --strict              # 0 issues
mix format --check-formatted    # Already formatted
```

## v0.2 Tasks (Your Assignment)

### Priority 1: Wire Gradient Tracking into Pipeline

The `GradientTracker` module exists but `track_grad_norms: true` option is not wired into `Pipeline.compute/3`.

**Task:** Modify `Pipeline.compute/3` to optionally compute and include gradient norms in metrics.

**Spec Reference:** `docs/20251203/implementation_specs/11_GRADIENT_TRACKING.md`

**Expected API:**
```elixir
{total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)
# metrics should include:
# - "l1_grad_norm" => float
# - "l2_grad_norm" => float
# - "total_grad_norm" => float
```

**Tests to add:** `test/nx_penalties/pipeline_test.exs`

### Priority 2: Implement Polaris Gradient Transforms

Currently stubs in `lib/nx_penalties/integration/polaris.ex`. Need real implementations.

**Task:** Implement `add_l2_decay/2`, `add_l1_decay/2`, `add_elastic_net_decay/3`

**Spec Reference:** `docs/20251203/implementation_specs/06_POLARIS_INTEGRATION.md`

**Expected API:**
```elixir
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
```

**Tests to add:** `test/nx_penalties/integration/polaris_test.exs`

### Priority 3: Add Missing API Functions

Per `08_API_REFERENCE.md`, these are missing:

1. **`NxPenalties.validate/1`** - Check tensor for NaN/Inf
   ```elixir
   @spec validate(Nx.Tensor.t()) :: {:ok, Nx.Tensor.t()} | {:error, :nan | :inf}
   ```

2. **L2 `clip` option** - Max absolute value before squaring
   ```elixir
   NxPenalties.l2(tensor, clip: 10.0)
   ```

### Priority 4: Constraints Module Tests

`lib/nx_penalties/constraints.ex` has implementations but NO TESTS.

**Task:** Add comprehensive tests for:
- `orthogonality/2` with soft/hard modes
- `consistency/3` with MSE/L1/cosine metrics
- Edge cases (empty tensors, single element, etc.)
- Gradient flow verification

**Tests to add:** `test/nx_penalties/constraints_test.exs`

### Priority 5: Gradient Tracker Tests

`lib/nx_penalties/gradient_tracker.ex` has implementations but NO TESTS.

**Task:** Add tests for:
- `compute_grad_norm/2`
- `pipeline_grad_norms/2`
- `total_grad_norm/2`
- Error handling for non-differentiable functions

**Tests to add:** `test/nx_penalties/gradient_tracker_test.exs`

## Development Guidelines

### TDD Approach (MANDATORY)

1. **Write tests FIRST** before any implementation
2. Run tests to see them fail
3. Implement the minimum code to pass
4. Refactor while keeping tests green
5. Repeat

### Nx.Defn Patterns

All numerical code must use the `deftransform` + `defnp` pattern:

```elixir
# Option handling in deftransform (can use Keyword.get, case, etc.)
deftransform my_function(tensor, opts \\ []) do
  mode = Keyword.get(opts, :mode, :default)
  case mode do
    :sum -> my_function_sum_impl(tensor)
    :mean -> my_function_mean_impl(tensor)
  end
end

# Numerical computation in defnp (JIT-compiled)
defnp my_function_sum_impl(tensor) do
  Nx.sum(Nx.abs(tensor))
end
```

**DO NOT:**
- Use `||`, `&&`, `if/else` with non-tensor conditions inside `defnp`
- Use `Kernel` functions inside `defnp`
- Use `Enum` or `Map` inside `defnp`
- Call `Nx.to_number` inside gradient-compatible code paths

### Nx API Notes

- Use `Nx.pow/2` (not `Nx.power`)
- Use `Nx.Random.key/1` + `Nx.Random.uniform/4` (key-based API)
- `Nx.axis_size/2` for dynamic dimension access in defn
- `tuple_size/1` must be in `deftransform`, not `defnp`

### Test Helpers Available

```elixir
import NxPenalties.TestHelpers

# Generate random tensors
tensor = random_tensor({4, 8})
tensor = random_tensor({4, 8}, min: 0.0, max: 1.0)

# Generate valid log-probabilities
logprobs = random_logprobs({4, 8})

# Floating point comparisons
assert_in_delta(expected, actual, 1.0e-5)
```

## Quality Gates (MUST PASS)

Before considering any task complete:

```bash
# All tests must pass
mix test
# Expected: 0 failures

# No compiler warnings
mix compile --warnings-as-errors
# Expected: Compiles successfully

# No Credo issues
mix credo --strict
# Expected: 0 issues

# Properly formatted
mix format --check-formatted
# Expected: Already formatted

# Dialyzer (run after PLT is built)
mix dialyzer
# Expected: 0 errors
```

## File Structure

```
lib/
├── nx_penalties.ex                 # Main entry module
└── nx_penalties/
    ├── penalties.ex                # L1, L2, Elastic Net
    ├── penalties/
    │   └── validation.ex           # Option validation
    ├── divergences.ex              # KL, JS, Entropy
    ├── pipeline.ex                 # Pipeline composition
    ├── constraints.ex              # Orthogonality, consistency
    ├── gradient_tracker.ex         # Gradient norm computation
    ├── telemetry.ex                # Telemetry events
    └── integration/
        ├── axon.ex                 # Axon helpers
        └── polaris.ex              # Polaris transforms

test/
├── test_helper.exs
├── support/
│   └── test_helpers.ex             # Test utilities
└── nx_penalties/
    ├── penalties_test.exs          # 28 tests
    ├── divergences_test.exs        # 22 tests
    ├── pipeline_test.exs           # 23 tests
    ├── constraints_test.exs        # TODO: Create
    ├── gradient_tracker_test.exs   # TODO: Create
    └── integration/
        └── polaris_test.exs        # TODO: Create

examples/
├── basic_usage.exs
├── pipeline_composition.exs
├── curriculum_learning.exs
└── axon_training.exs
```

## Example Test Pattern

```elixir
defmodule NxPenalties.ConstraintsTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Constraints

  describe "orthogonality/2" do
    test "returns zero for orthogonal rows" do
      # Identity-like matrix (orthogonal rows)
      tensor = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ])

      result = Constraints.orthogonality(tensor, mode: :soft)
      assert_in_delta Nx.to_number(result), 0.0, 1.0e-5
    end

    test "returns positive penalty for correlated rows" do
      tensor = Nx.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
      ])

      result = Constraints.orthogonality(tensor, mode: :soft)
      assert Nx.to_number(result) > 0.0
    end

    test "gradient flows correctly" do
      tensor = random_tensor({4, 8})

      grad_fn = Nx.Defn.grad(fn t ->
        Constraints.orthogonality(t)
      end)

      grads = grad_fn.(tensor)
      assert Nx.shape(grads) == Nx.shape(tensor)
      refute Nx.all(Nx.equal(grads, 0)) |> Nx.to_number() == 1
    end
  end
end
```

## Commit Guidelines

When committing changes:

```bash
git commit -m "$(cat <<'EOF'
feat: Add gradient tracking to pipeline compute

- Wire track_grad_norms option into Pipeline.compute/3
- Add per-penalty gradient norms to metrics map
- Add total_grad_norm to metrics
- Add comprehensive tests for gradient tracking

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Summary Checklist

- [ ] Read all spec documents listed above
- [ ] Wire `track_grad_norms` into `Pipeline.compute/3`
- [ ] Implement Polaris gradient transforms
- [ ] Add `NxPenalties.validate/1`
- [ ] Add L2 `clip` option
- [ ] Create `constraints_test.exs` with full coverage
- [ ] Create `gradient_tracker_test.exs` with full coverage
- [ ] Create `polaris_test.exs` with full coverage
- [ ] All quality gates pass
- [ ] Update CHANGELOG.md
- [ ] Update version in mix.exs to 0.2.0

## Questions?

If unclear about any requirement, check the spec documents. The API reference (`08_API_REFERENCE.md`) is the source of truth for function signatures and behavior.
