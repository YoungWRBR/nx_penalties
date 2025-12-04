# Polaris Full Integration Example
#
# Run with: mix run examples/polaris_full_integration.exs
#
# This comprehensive example demonstrates all NxPenalties gradient
# transforms for Polaris optimizers.

IO.puts("=== NxPenalties Full Polaris Integration ===\n")

alias NxPenalties.Integration.Polaris, as: PolarisIntegration

# ============================================================================
# Helper: Simple SGD Optimizer
# ============================================================================

# We define a simple SGD for demonstration. In production, use Polaris.Optimizers.
defmodule SimpleOptimizers do
  @moduledoc false

  def sgd(learning_rate) do
    init_fn = fn _params -> %{} end

    update_fn = fn gradients, state, _params ->
      updates = deep_map(gradients, fn g -> Nx.multiply(g, -learning_rate) end)
      {updates, state}
    end

    {init_fn, update_fn}
  end

  def adam(opts \\ []) do
    lr = Keyword.get(opts, :learning_rate, 0.001)
    beta1 = Keyword.get(opts, :beta1, 0.9)
    beta2 = Keyword.get(opts, :beta2, 0.999)
    eps = Keyword.get(opts, :eps, 1.0e-8)

    init_fn = fn params ->
      %{
        m: deep_map(params, fn p -> Nx.broadcast(0.0, Nx.shape(p)) end),
        v: deep_map(params, fn p -> Nx.broadcast(0.0, Nx.shape(p)) end),
        t: 0
      }
    end

    update_fn = fn gradients, state, _params ->
      t = state.t + 1

      # Update biased first moment estimate
      m =
        deep_map2(state.m, gradients, fn m, g ->
          Nx.add(Nx.multiply(m, beta1), Nx.multiply(g, 1 - beta1))
        end)

      # Update biased second moment estimate
      v =
        deep_map2(state.v, gradients, fn v, g ->
          Nx.add(Nx.multiply(v, beta2), Nx.multiply(Nx.pow(g, 2), 1 - beta2))
        end)

      # Bias correction
      m_hat = deep_map(m, fn x -> Nx.divide(x, 1 - :math.pow(beta1, t)) end)
      v_hat = deep_map(v, fn x -> Nx.divide(x, 1 - :math.pow(beta2, t)) end)

      # Compute updates
      updates =
        deep_map2(m_hat, v_hat, fn m_h, v_h ->
          Nx.multiply(Nx.divide(m_h, Nx.add(Nx.sqrt(v_h), eps)), -lr)
        end)

      {updates, %{state | m: m, v: v, t: t}}
    end

    {init_fn, update_fn}
  end

  defp deep_map(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)
  defp deep_map(m, fun) when is_map(m), do: Map.new(m, fn {k, v} -> {k, deep_map(v, fun)} end)

  defp deep_map2(%Nx.Tensor{} = a, %Nx.Tensor{} = b, fun), do: fun.(a, b)

  defp deep_map2(a, b, fun) when is_map(a) and is_map(b) do
    Map.new(a, fn {k, v} -> {k, deep_map2(v, b[k], fun)} end)
  end
end

# ============================================================================
# Helper: Apply Updates
# ============================================================================

defmodule UpdateHelper do
  def apply_updates(%Nx.Tensor{} = p, %Nx.Tensor{} = u), do: Nx.add(p, u)

  def apply_updates(p, u) when is_map(p) and is_map(u) do
    Map.merge(p, u, fn _k, pv, uv -> apply_updates(pv, uv) end)
  end
end

# ============================================================================
# Part 1: L2 Weight Decay
# ============================================================================

IO.puts("--- Part 1: L2 Weight Decay ---")
IO.puts("Decoupled weight decay (AdamW style): g' = g + λw\n")

base_opt = SimpleOptimizers.sgd(0.1)
opt_with_l2 = PolarisIntegration.add_l2_decay(base_opt, 0.1)

{init_fn, update_fn} = opt_with_l2

params = %{
  layer1: %{w: Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), b: Nx.tensor([0.1, 0.2])},
  layer2: %{w: Nx.tensor([[0.5, -0.5]]), b: Nx.tensor([0.0])}
}

# Zero gradients to see pure decay effect
gradients = %{
  layer1: %{w: Nx.tensor([[0.0, 0.0], [0.0, 0.0]]), b: Nx.tensor([0.0, 0.0])},
  layer2: %{w: Nx.tensor([[0.0, 0.0]]), b: Nx.tensor([0.0])}
}

state = init_fn.(params)
{updates, _new_state} = update_fn.(gradients, state, params)

IO.puts("Original layer1.w:\n#{inspect(Nx.to_list(params.layer1.w))}")
IO.puts("Updates (pure L2 decay):\n#{inspect(Nx.to_list(updates.layer1.w))}")

new_params = UpdateHelper.apply_updates(params, updates)
IO.puts("After update:\n#{inspect(Nx.to_list(new_params.layer1.w))}")
IO.puts("")

# ============================================================================
# Part 2: L1 Weight Decay
# ============================================================================

IO.puts("--- Part 2: L1 Weight Decay ---")
IO.puts("Sign-based decay for sparsity: g' = g + λ*sign(w)\n")

opt_with_l1 = PolarisIntegration.add_l1_decay(base_opt, 0.05)
{init_fn, update_fn} = opt_with_l1

state = init_fn.(params)
{updates, _} = update_fn.(gradients, state, params)

IO.puts("Original layer1.w:\n#{inspect(Nx.to_list(params.layer1.w))}")
IO.puts("Sign of weights:\n#{inspect(Nx.to_list(Nx.sign(params.layer1.w)))}")
IO.puts("Updates (L1 decay):\n#{inspect(Nx.to_list(updates.layer1.w))}")
IO.puts("")

# ============================================================================
# Part 3: Elastic Net Decay
# ============================================================================

IO.puts("--- Part 3: Elastic Net Decay ---")
IO.puts("Combined L1+L2: g' = g + λ*(α*sign(w) + (1-α)*w)\n")

opt_elastic = PolarisIntegration.add_elastic_net_decay(base_opt, 0.1, 0.3)
{init_fn, update_fn} = opt_elastic

state = init_fn.(params)
{updates, _} = update_fn.(gradients, state, params)

IO.puts("Elastic Net (λ=0.1, α=0.3 means 30% L1, 70% L2)")
IO.puts("Updates:\n#{inspect(Nx.to_list(updates.layer1.w))}")
IO.puts("")

# ============================================================================
# Part 4: Gradient Clipping
# ============================================================================

IO.puts("--- Part 4: Gradient Clipping ---")
IO.puts("Clip gradients to maximum L2 norm\n")

opt_clipped = PolarisIntegration.add_gradient_clipping(base_opt, 1.0)
{init_fn, update_fn} = opt_clipped

# Large gradients
large_grads = %{
  layer1: %{w: Nx.tensor([[10.0, 20.0], [30.0, 40.0]]), b: Nx.tensor([5.0, 5.0])},
  layer2: %{w: Nx.tensor([[15.0, -15.0]]), b: Nx.tensor([10.0])}
}

# Compute original norm
flat_grads = [
  Nx.flatten(large_grads.layer1.w),
  Nx.flatten(large_grads.layer1.b),
  Nx.flatten(large_grads.layer2.w),
  Nx.flatten(large_grads.layer2.b)
]

original_norm =
  flat_grads
  |> Enum.map(&Nx.sum(Nx.pow(&1, 2)))
  |> Enum.reduce(&Nx.add/2)
  |> Nx.sqrt()
  |> Nx.to_number()

IO.puts("Original gradient norm: #{Float.round(original_norm, 2)}")

state = init_fn.(params)
{updates, _} = update_fn.(large_grads, state, params)

# After clipping, the update norm should be bounded
clipped_updates = [
  Nx.flatten(updates.layer1.w),
  Nx.flatten(updates.layer1.b),
  Nx.flatten(updates.layer2.w),
  Nx.flatten(updates.layer2.b)
]

clipped_norm =
  clipped_updates
  |> Enum.map(&Nx.sum(Nx.pow(&1, 2)))
  |> Enum.reduce(&Nx.add/2)
  |> Nx.sqrt()
  |> Nx.to_number()

# Note: updates are negative lr * clipped_grad, so we check the scaled values
IO.puts("Clipping to max_norm=1.0")
IO.puts("Layer1.w updates (scaled by -lr=0.1):\n#{inspect(Nx.to_list(updates.layer1.w))}")
IO.puts("Update norm after clipping: #{Float.round(clipped_norm, 4)}")
IO.puts("")

# ============================================================================
# Part 5: Gradient Noise
# ============================================================================

IO.puts("--- Part 5: Gradient Noise ---")
IO.puts("Add decaying Gaussian noise for regularization\n")

opt_noisy = PolarisIntegration.add_gradient_noise(base_opt, 0.1, decay: 0.5)
{init_fn, update_fn} = opt_noisy

small_grads = %{
  layer1: %{w: Nx.tensor([[0.1, 0.1], [0.1, 0.1]]), b: Nx.tensor([0.1, 0.1])},
  layer2: %{w: Nx.tensor([[0.1, 0.1]]), b: Nx.tensor([0.1])}
}

state = init_fn.(params)

IO.puts("Noise variance schedule: σ²(t) = 0.1 / (1 + t)^0.5")
IO.puts("Step 0: variance = #{Float.round(0.1 / :math.pow(1, 0.5), 4)}")
IO.puts("Step 10: variance = #{Float.round(0.1 / :math.pow(11, 0.5), 4)}")
IO.puts("Step 100: variance = #{Float.round(0.1 / :math.pow(101, 0.5), 4)}")

# Run a few steps to see noise effect
IO.puts("\nRunning 3 steps with same gradients (noise makes each different):")

{_, _} =
  Enum.reduce(1..3, {state, []}, fn step, {s, hist} ->
    {updates, new_s} = update_fn.(small_grads, s, params)
    val = Nx.to_list(updates.layer1.w) |> List.flatten() |> Enum.at(0)
    IO.puts("  Step #{step}: layer1.w[0,0] update = #{Float.round(val, 6)}")
    {new_s, [updates | hist]}
  end)

IO.puts("")

# ============================================================================
# Part 6: Adaptive Gradient Clipping (AGC)
# ============================================================================

IO.puts("--- Part 6: Adaptive Gradient Clipping (AGC) ---")
IO.puts("Clip based on gradient-to-parameter norm ratio\n")

opt_agc = PolarisIntegration.add_adaptive_gradient_clipping(base_opt, 0.1)
{init_fn, update_fn} = opt_agc

state = init_fn.(params)
{updates, _} = update_fn.(large_grads, state, params)

IO.puts("AGC clips when ||g|| / max(||w||, eps) > clip_factor")

IO.puts(
  "Layer1.w norm: #{Float.round(Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(params.layer1.w, 2)))), 2)}"
)

IO.puts(
  "Layer1.w grad norm: #{Float.round(Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(large_grads.layer1.w, 2)))), 2)}"
)

IO.puts("Updates (AGC clipped):\n#{inspect(Nx.to_list(updates.layer1.w))}")
IO.puts("")

# ============================================================================
# Part 7: Gradient Centralization
# ============================================================================

IO.puts("--- Part 7: Gradient Centralization ---")
IO.puts("Subtract mean from gradients for stability\n")

opt_gc = PolarisIntegration.add_gradient_centralization(base_opt)
{init_fn, update_fn} = opt_gc

# Gradients with non-zero mean
biased_grads = %{
  layer1: %{w: Nx.tensor([[5.0, 6.0], [7.0, 8.0]]), b: Nx.tensor([1.0, 2.0])},
  layer2: %{w: Nx.tensor([[3.0, 4.0]]), b: Nx.tensor([0.5])}
}

IO.puts("Original layer1.w gradient:\n#{inspect(Nx.to_list(biased_grads.layer1.w))}")
IO.puts("Mean per row: #{inspect(Nx.to_list(Nx.mean(biased_grads.layer1.w, axes: [1])))}")

state = init_fn.(params)
{updates, _} = update_fn.(biased_grads, state, params)

# The centralized gradients should have zero mean per row
IO.puts("Updates (centralized, then scaled by -lr):\n#{inspect(Nx.to_list(updates.layer1.w))}")
IO.puts("")

# ============================================================================
# Part 8: Composing Transforms
# ============================================================================

IO.puts("--- Part 8: Composing Multiple Transforms ---")
IO.puts("Stack transforms via piping\n")

composed_opt =
  SimpleOptimizers.adam(learning_rate: 0.001)
  |> PolarisIntegration.add_l2_decay(0.01)
  |> PolarisIntegration.add_gradient_clipping(5.0)
  |> PolarisIntegration.add_gradient_centralization()

{init_fn, update_fn} = composed_opt

state = init_fn.(params)

IO.puts("Optimizer stack: Adam -> L2 decay -> Gradient clipping -> Centralization")
IO.puts("Running 5 update steps:")

{final_params, _} =
  Enum.reduce(1..5, {params, state}, fn step, {p, s} ->
    # Simulate gradients (in practice, from backprop)
    grads = %{
      layer1: %{
        # Proportional to params
        w: Nx.multiply(p.layer1.w, 0.1),
        b: Nx.multiply(p.layer1.b, 0.1)
      },
      layer2: %{
        w: Nx.multiply(p.layer2.w, 0.1),
        b: Nx.multiply(p.layer2.b, 0.1)
      }
    }

    {updates, new_s} = update_fn.(grads, s, p)
    new_p = UpdateHelper.apply_updates(p, updates)

    if rem(step, 2) == 1 do
      w_norm = Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(new_p.layer1.w, 2))))
      IO.puts("  Step #{step}: layer1.w norm = #{Float.round(w_norm, 4)}")
    end

    {new_p, new_s}
  end)

final_norm = Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(final_params.layer1.w, 2))))
IO.puts("Final layer1.w norm after stacked transforms: #{Float.round(final_norm, 4)}")
IO.puts("")

# ============================================================================
# Part 9: Real-World Usage Pattern
# ============================================================================

IO.puts("--- Part 9: Real-World Usage Pattern ---")
IO.puts("")

IO.puts("""
In production with Polaris optimizers:

    alias NxPenalties.Integration.Polaris, as: PolarisIntegration

    # Build regularized optimizer
    optimizer =
      Polaris.Optimizers.adam(learning_rate: 0.001)
      |> PolarisIntegration.add_l2_decay(0.01)           # Weight decay
      |> PolarisIntegration.add_gradient_clipping(1.0)    # Prevent explosion
      |> PolarisIntegration.add_gradient_centralization() # Improve stability

    # Or for sparse models
    sparse_optimizer =
      Polaris.Optimizers.sgd(learning_rate: 0.01)
      |> PolarisIntegration.add_l1_decay(0.001)          # Sparsity
      |> PolarisIntegration.add_gradient_noise(0.01)     # Regularization

    # Use in Axon training
    model
    |> Axon.Loop.trainer(loss, optimizer)
    |> Axon.Loop.run(data, %{}, epochs: 100)
""")

# ============================================================================
# Part 10: Comparison Summary
# ============================================================================

IO.puts("--- Part 10: Transform Comparison ---\n")

IO.puts("""
| Transform                | Purpose                        | When to Use                    |
|--------------------------|--------------------------------|--------------------------------|
| add_l2_decay             | Weight decay (AdamW)           | Most training scenarios        |
| add_l1_decay             | Sparsity                       | Feature selection, compression |
| add_elastic_net_decay    | Combined L1+L2                 | Balanced regularization        |
| add_gradient_clipping    | Prevent explosion              | RNNs, transformers, unstable   |
| add_gradient_noise       | Escape local minima            | Very deep networks             |
| add_adaptive_gc          | Scale-aware clipping           | Varying parameter scales       |
| add_gradient_central.    | Zero-mean gradients            | Training stability             |
""")

IO.puts("=== Done ===")
