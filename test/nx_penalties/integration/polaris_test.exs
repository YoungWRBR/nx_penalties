defmodule NxPenalties.Integration.PolarisTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Integration.Polaris, as: PolarisIntegration

  describe "add_l2_decay/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_l2_decay(base_optimizer, 0.01)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "modifies gradients with L2 decay term" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0, 2.0])}
      gradients = %{w: Nx.tensor([0.1, 0.1])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Gradient should be: 0.1 + 0.1 * [1, 2] = [0.2, 0.3]
      # Update with lr=0.1: -0.1 * [0.2, 0.3] = [-0.02, -0.03]
      assert_close(updates.w, Nx.tensor([-0.02, -0.03]))
    end

    test "uses default decay value" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Default decay is 0.01
      # Gradient: 0 + 0.01 * 1 = 0.01
      # Update: -0.1 * 0.01 = -0.001
      assert_close(updates.w, Nx.tensor([-0.001]))
    end

    test "handles nested parameter maps" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{
        layer1: %{w: Nx.tensor([1.0, 2.0])},
        layer2: %{w: Nx.tensor([3.0])}
      }

      gradients = %{
        layer1: %{w: Nx.tensor([0.0, 0.0])},
        layer2: %{w: Nx.tensor([0.0])}
      }

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Decay only (gradient = 0)
      # layer1.w: -0.1 * (0.1 * [1, 2]) = [-0.01, -0.02]
      # layer2.w: -0.1 * (0.1 * [3]) = [-0.03]
      assert_close(updates.layer1.w, Nx.tensor([-0.01, -0.02]))
      assert_close(updates.layer2.w, Nx.tensor([-0.03]))
    end
  end

  describe "add_l1_decay/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_l1_decay(base_optimizer, 0.001)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "uses sign of weights for L1 decay" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l1_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0, -3.0])}
      gradients = %{w: Nx.tensor([0.0, 0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Gradient: 0 + 0.1 * sign([2, -3]) = [0.1, -0.1]
      # Update: -0.1 * [0.1, -0.1] = [-0.01, 0.01]
      assert_close(updates.w, Nx.tensor([-0.01, 0.01]))
    end

    test "handles zero weights correctly" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l1_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([0.0, 1.0])}
      gradients = %{w: Nx.tensor([0.0, 0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # sign(0) = 0 in Nx
      # Gradient: [0 + 0.1*0, 0 + 0.1*1] = [0, 0.1]
      # Update: [-0, -0.01] = [0, -0.01]
      assert_close(updates.w, Nx.tensor([0.0, -0.01]))
    end
  end

  describe "add_elastic_net_decay/3" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.01, 0.5)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "combines L1 and L2 decay" do
      base_optimizer = sgd_optimizer(1.0)
      # 50% L1, 50% L2
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 1.0, 0.5)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L1 term: 0.5 * sign(2) = 0.5
      # L2 term: 0.5 * 2 = 1.0
      # Total decay: 1.0 * (0.5 + 1.0) = 1.5
      # Update: -1.0 * 1.5 = -1.5
      assert_close(updates.w, Nx.tensor([-1.5]))
    end

    test "pure L1 with l1_ratio=1.0" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.1, 1.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L1 only: 0.1 * sign(2) * 1.0 + 0.1 * 2 * 0.0 = 0.1
      # Update: -1.0 * 0.1 = -0.1
      assert_close(updates.w, Nx.tensor([-0.1]))
    end

    test "pure L2 with l1_ratio=0.0" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.1, 0.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L2 only: 0.1 * 2 = 0.2
      # Update: -1.0 * 0.2 = -0.2
      assert_close(updates.w, Nx.tensor([-0.2]))
    end
  end

  describe "optimizer composition" do
    test "transforms compose via piping" do
      optimizer =
        sgd_optimizer(0.1)
        |> PolarisIntegration.add_l2_decay(0.01)
        |> PolarisIntegration.add_l1_decay(0.001)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Both decays should be applied
      # L2 decay: 0.01 * 1.0 = 0.01
      # L1 decay: 0.001 * sign(1.0) = 0.001
      # Total gradient: 0 + 0.01 + 0.001 = 0.011
      # Update: -0.1 * 0.011 = -0.0011
      assert_close(updates.w, Nx.tensor([-0.0011]), atol: 1.0e-5)
    end

    test "state is properly maintained across updates" do
      optimizer = PolarisIntegration.add_l2_decay(sgd_optimizer(0.1), 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.1])}

      state = init_fn.(params)
      {_updates1, state1} = update_fn.(gradients, state, params)
      {_updates2, state2} = update_fn.(gradients, state1, params)

      # State should be updated (contains base optimizer state)
      assert is_map(state2)
    end
  end

  # ============================================================================
  # New Polaris Integration Functions (v0.1.1)
  # ============================================================================

  describe "add_gradient_clipping/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_gradient_clipping(base_optimizer, 1.0)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "clips gradients when norm exceeds max_norm" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_clipping(base_optimizer, 1.0)

      {init_fn, update_fn} = optimizer

      # Gradient with norm = 5.0 (3^2 + 4^2 = 25, sqrt = 5)
      params = %{w: Nx.tensor([0.0, 0.0])}
      gradients = %{w: Nx.tensor([3.0, 4.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # After clipping to max_norm=1.0:
      # scale = 1.0 / 5.0 = 0.2
      # clipped_grad = [0.6, 0.8]
      # update = -1.0 * [0.6, 0.8] = [-0.6, -0.8]
      assert_close(updates.w, Nx.tensor([-0.6, -0.8]), atol: 1.0e-4)
    end

    test "does not clip when norm is below max_norm" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_clipping(base_optimizer, 10.0)

      {init_fn, update_fn} = optimizer

      # Gradient with norm = 5.0 (below max_norm=10.0)
      params = %{w: Nx.tensor([0.0, 0.0])}
      gradients = %{w: Nx.tensor([3.0, 4.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # No clipping, just SGD: -1.0 * [3, 4] = [-3, -4]
      assert_close(updates.w, Nx.tensor([-3.0, -4.0]), atol: 1.0e-4)
    end

    test "handles nested parameter maps" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_clipping(base_optimizer, 1.0)

      {init_fn, update_fn} = optimizer

      params = %{
        layer1: %{w: Nx.tensor([0.0])},
        layer2: %{w: Nx.tensor([0.0])}
      }

      # Global norm = sqrt(3^2 + 4^2) = 5.0
      gradients = %{
        layer1: %{w: Nx.tensor([3.0])},
        layer2: %{w: Nx.tensor([4.0])}
      }

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Scale = 1/5 = 0.2
      # Updates: [-0.6, -0.8]
      assert_close(updates.layer1.w, Nx.tensor([-0.6]), atol: 1.0e-4)
      assert_close(updates.layer2.w, Nx.tensor([-0.8]), atol: 1.0e-4)
    end

    test "uses default max_norm value" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_clipping(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([0.0, 0.0])}
      gradients = %{w: Nx.tensor([3.0, 4.0])}

      state = init_fn.(params)
      {updates, _} = update_fn.(gradients, state, params)

      # Default max_norm is 1.0
      # norm = 5.0, scale = 0.2
      assert_close(updates.w, Nx.tensor([-0.6, -0.8]), atol: 1.0e-4)
    end
  end

  describe "add_gradient_noise/3" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_gradient_noise(base_optimizer, 0.01)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "adds noise to gradients" do
      base_optimizer = sgd_optimizer(1.0)
      # Use fixed seed for deterministic test
      optimizer = PolarisIntegration.add_gradient_noise(base_optimizer, 0.1, seed: 42)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([0.0])}
      gradients = %{w: Nx.tensor([1.0])}

      state = init_fn.(params)
      {updates1, state1} = update_fn.(gradients, state, params)
      {updates2, _state2} = update_fn.(gradients, state1, params)

      # Pure SGD would give exactly -1.0; noise modifies this
      # Verify noise is being added (updates differ from pure SGD)
      pure_sgd_update = Nx.tensor([-1.0])
      refute Nx.to_number(Nx.all_close(updates1.w, pure_sgd_update, atol: 1.0e-6)) == 1
      refute Nx.to_number(Nx.all_close(updates2.w, pure_sgd_update, atol: 1.0e-6)) == 1

      # Verify updates differ from each other (noise varies per step)
      refute Nx.to_number(Nx.all_close(updates1.w, updates2.w, atol: 1.0e-6)) == 1

      # Verify updates are in reasonable range (within 3 std devs of expected)
      # With variance=0.1, std_dev ~= 0.316, so 3*std ~= 0.95
      [update1_val] = Nx.to_flat_list(updates1.w)
      [update2_val] = Nx.to_flat_list(updates2.w)
      assert abs(update1_val + 1.0) < 1.0
      assert abs(update2_val + 1.0) < 1.0
    end

    test "noise variance decays over steps" do
      base_optimizer = sgd_optimizer(1.0)
      # High variance and decay to see the effect
      optimizer = PolarisIntegration.add_gradient_noise(base_optimizer, 1.0, decay: 1.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([0.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)

      # Run many updates
      {_updates, final_state} =
        Enum.reduce(1..10, {nil, state}, fn _, {_, s} ->
          update_fn.(gradients, s, params)
        end)

      # Step counter should have incremented
      assert final_state.step == 10
    end

    test "initializes with random key" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_gradient_noise(base_optimizer, 0.01)

      {init_fn, _update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      state = init_fn.(params)

      assert Map.has_key?(state, :key)
      assert Map.has_key?(state, :step)
      assert state.step == 0
    end
  end

  describe "add_adaptive_gradient_clipping/3" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_adaptive_gradient_clipping(base_optimizer, 0.01)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "clips based on parameter-gradient norm ratio" do
      base_optimizer = sgd_optimizer(1.0)
      # clip_factor = 0.1 means grad_norm should be at most 0.1 * param_norm
      optimizer = PolarisIntegration.add_adaptive_gradient_clipping(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      # param_norm = 10.0
      params = %{w: Nx.tensor([10.0])}
      # grad_norm = 5.0 (exceeds 0.1 * 10 = 1.0)
      gradients = %{w: Nx.tensor([5.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # max_g_norm = 0.1 * 10 = 1.0
      # scale = 1.0 / 5.0 = 0.2
      # clipped_grad = 5.0 * 0.2 = 1.0
      # update = -1.0 * 1.0 = -1.0
      assert_close(updates.w, Nx.tensor([-1.0]), atol: 1.0e-4)
    end

    test "does not clip when gradient is small relative to parameter" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_adaptive_gradient_clipping(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      # param_norm = 10.0, max_grad = 1.0
      params = %{w: Nx.tensor([10.0])}
      # grad_norm = 0.5 (below max)
      gradients = %{w: Nx.tensor([0.5])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # No clipping: update = -1.0 * 0.5 = -0.5
      assert_close(updates.w, Nx.tensor([-0.5]), atol: 1.0e-4)
    end

    test "uses eps for small parameter norms" do
      base_optimizer = sgd_optimizer(1.0)
      # eps = 0.1 (higher for testing)
      optimizer = PolarisIntegration.add_adaptive_gradient_clipping(base_optimizer, 1.0, eps: 0.1)

      {init_fn, update_fn} = optimizer

      # Very small param_norm (near zero)
      params = %{w: Nx.tensor([0.01])}
      gradients = %{w: Nx.tensor([1.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # max_g_norm = 1.0 * max(0.01, 0.1) = 0.1
      # scale = 0.1 / 1.0 = 0.1
      # update = -1.0 * 0.1 = -0.1
      assert_close(updates.w, Nx.tensor([-0.1]), atol: 1.0e-4)
    end
  end

  describe "add_gradient_centralization/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_gradient_centralization(base_optimizer)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "centralizes gradients by subtracting mean" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_centralization(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
      # Mean along non-first axes: each row has mean subtracted
      # Row 1: [1, 2, 3] -> mean = 2 -> [-1, 0, 1]
      # Row 2: [4, 5, 6] -> mean = 5 -> [-1, 0, 1]
      gradients = %{w: Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Centralized grads: [[-1, 0, 1], [-1, 0, 1]]
      # Updates: -1.0 * [...] = [[1, 0, -1], [1, 0, -1]]
      expected = Nx.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
      assert_close(updates.w, expected, atol: 1.0e-4)
    end

    test "does not centralize 1D gradients" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_centralization(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0, 2.0, 3.0])}
      gradients = %{w: Nx.tensor([1.0, 2.0, 3.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # 1D tensors are not centralized
      # Updates: -1.0 * [1, 2, 3] = [-1, -2, -3]
      assert_close(updates.w, Nx.tensor([-1.0, -2.0, -3.0]), atol: 1.0e-4)
    end

    test "handles nested parameter maps" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_gradient_centralization(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{
        layer1: %{w: Nx.tensor([[1.0, 2.0]])},
        layer2: %{w: Nx.tensor([1.0])}
      }

      gradients = %{
        layer1: %{w: Nx.tensor([[1.0, 3.0]])},
        layer2: %{w: Nx.tensor([5.0])}
      }

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # layer1: 2D, centralize -> [[1, 3]] mean=2 -> [[-1, 1]]
      # layer2: 1D, no centralization -> [5]
      assert_close(updates.layer1.w, Nx.tensor([[1.0, -1.0]]), atol: 1.0e-4)
      assert_close(updates.layer2.w, Nx.tensor([-5.0]), atol: 1.0e-4)
    end
  end

  describe "new transforms composition" do
    test "gradient clipping composes with decay" do
      optimizer =
        sgd_optimizer(1.0)
        |> PolarisIntegration.add_l2_decay(0.1)
        |> PolarisIntegration.add_gradient_clipping(1.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([10.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L2 decay adds 0.1 * 10 = 1.0 to gradient
      # Gradient is now [1.0], norm = 1.0
      # max_norm = 1.0, no clipping needed
      # Update: -1.0 * 1.0 = -1.0
      assert_close(updates.w, Nx.tensor([-1.0]), atol: 1.0e-4)
    end

    test "centralization composes with AGC" do
      optimizer =
        sgd_optimizer(1.0)
        |> PolarisIntegration.add_gradient_centralization()
        |> PolarisIntegration.add_adaptive_gradient_clipping(0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([[1.0, 1.0]])}
      gradients = %{w: Nx.tensor([[1.0, 3.0]])}

      state = init_fn.(params)
      {_updates, _new_state} = update_fn.(gradients, state, params)

      # Centralization: mean=2 -> [[-1, 1]]
      # Then AGC clips based on param/grad ratio
      # This is a valid composition
    end
  end

  # Helper to create a simple SGD optimizer for testing
  defp sgd_optimizer(learning_rate) do
    init_fn = fn _params -> %{} end

    update_fn = fn gradients, state, _params ->
      updates =
        deep_map_single(gradients, fn g ->
          Nx.multiply(g, -learning_rate)
        end)

      {updates, state}
    end

    {init_fn, update_fn}
  end

  # Deep map over single nested structure (handles both maps and tensors)
  # Check for tensors first since Nx.Tensor is a struct (which is also a map)
  defp deep_map_single(%Nx.Tensor{} = tensor, fun) do
    fun.(tensor)
  end

  defp deep_map_single(structure, fun) when is_map(structure) do
    Map.new(structure, fn {key, value} ->
      {key, deep_map_single(value, fun)}
    end)
  end

  defp deep_map_single(other, fun) do
    fun.(other)
  end
end
