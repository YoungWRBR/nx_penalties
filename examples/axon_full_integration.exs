# Axon Full Integration Example
#
# Run with: mix run examples/axon_full_integration.exs
#
# This comprehensive example demonstrates all NxPenalties integration
# patterns with Axon training loops.

IO.puts("=== NxPenalties Full Axon Integration ===\n")

# Helper module for nested maps (must be defined before use)
defmodule DeepMapUtil do
  def deep_map(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)
  def deep_map(m, fun) when is_map(m), do: Map.new(m, fn {k, v} -> {k, deep_map(v, fun)} end)
end

# Check if Axon is available
case Code.ensure_loaded(Axon) do
  {:module, _} ->
    alias NxPenalties.Integration.Axon, as: AxonIntegration

    # ============================================================================
    # Part 1: Basic Model Setup
    # ============================================================================

    IO.puts("--- Part 1: Model Setup ---\n")

    # Create a simple MLP for regression
    model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(64, activation: :relu, name: "hidden1")
      |> Axon.dense(32, activation: :relu, name: "hidden2")
      |> Axon.dense(1, name: "output")

    IO.puts("Model: 10 -> 64 (ReLU) -> 32 (ReLU) -> 1")

    # Generate synthetic data
    key = Nx.Random.key(42)
    {x_train, key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {200, 10}, type: :f32)
    {noise, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {200, 1}, type: :f32)

    # Target: weighted sum of first 5 features + noise
    weights = Nx.tensor([[0.5], [1.0], [-0.5], [0.3], [0.8], [0.0], [0.0], [0.0], [0.0], [0.0]])
    y_train = x_train |> Nx.dot(weights) |> Nx.add(noise)

    IO.puts("Training data: #{elem(Nx.shape(x_train), 0)} samples")
    IO.puts("")

    # ============================================================================
    # Part 2: Pattern 1 - Simple Loss Wrapping
    # ============================================================================

    IO.puts("--- Part 2: Simple Loss Wrapping ---")
    IO.puts("Using wrap_loss/3 to add L2 penalty to MSE loss\n")

    # Wrap MSE with L2 penalty
    simple_loss =
      AxonIntegration.wrap_loss(
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        &NxPenalties.l2/2,
        lambda: 0.01
      )

    # Test the wrapped loss
    test_pred = Nx.tensor([[1.0], [2.0]])
    test_true = Nx.tensor([[1.1], [1.9]])
    wrapped_loss_val = simple_loss.(test_true, test_pred)
    IO.puts("Sample wrapped loss: #{Nx.to_number(wrapped_loss_val)}")
    IO.puts("")

    # ============================================================================
    # Part 3: Pattern 2 - Pipeline-Based Loss
    # ============================================================================

    IO.puts("--- Part 3: Pipeline-Based Loss ---")
    IO.puts("Using wrap_loss_with_pipeline/3 for multiple penalties\n")

    # Create pipeline with multiple penalties
    pipeline =
      NxPenalties.pipeline([
        {:l2, weight: 0.01},
        {:l1, weight: 0.001}
      ])

    pipeline_loss =
      AxonIntegration.wrap_loss_with_pipeline(
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        pipeline
      )

    pipeline_loss_val = pipeline_loss.(test_true, test_pred)
    IO.puts("Pipeline loss (L2 + L1): #{Nx.to_number(pipeline_loss_val)}")
    IO.puts("")

    # ============================================================================
    # Part 4: Pattern 3 - Custom Training Step with Metrics
    # ============================================================================

    IO.puts("--- Part 4: Custom Training Step with Metrics ---")
    IO.puts("Using build_train_step/4 for full control\n")

    # Simple SGD optimizer (could use Polaris.Optimizers.adam)
    sgd_optimizer = fn learning_rate ->
      init_fn = fn _params -> %{} end

      update_fn = fn gradients, state, _params ->
        updates = DeepMapUtil.deep_map(gradients, fn g -> Nx.multiply(g, -learning_rate) end)
        {updates, state}
      end

      {init_fn, update_fn}
    end

    # Build training step
    {init_fn, step_fn} =
      AxonIntegration.build_train_step(
        model,
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        pipeline,
        sgd_optimizer.(0.01)
      )

    # Initialize
    state = init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())
    IO.puts("Initial state keys: #{inspect(Map.keys(state))}")

    # Run a few steps
    IO.puts("\nTraining steps:")

    {_final_state, _} =
      Enum.reduce(1..5, {state, nil}, fn step, {s, _} ->
        {new_s, metrics} = step_fn.(s, {x_train, y_train})

        IO.puts(
          "  Step #{step}: loss=#{Float.round(metrics["loss"], 6)}, " <>
            "base=#{Float.round(metrics["base_loss"], 6)}, " <>
            "penalty=#{Float.round(metrics["penalty_total"], 6)}"
        )

        {new_s, metrics}
      end)

    IO.puts("")

    # ============================================================================
    # Part 5: Weight Decay on Model Parameters
    # ============================================================================

    IO.puts("--- Part 5: Weight Decay on Model Parameters ---")
    IO.puts("Using build_train_step_with_weight_decay/5\n")

    # Parameter penalty function
    param_penalty_fn = fn params ->
      params
      |> AxonIntegration.flatten_params()
      |> Enum.map(&NxPenalties.l2(&1, lambda: 1.0))
      |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
    end

    {wd_init_fn, wd_step_fn} =
      AxonIntegration.build_train_step_with_weight_decay(
        model,
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        param_penalty_fn,
        sgd_optimizer.(0.01),
        lambda: 0.001
      )

    wd_state = wd_init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

    IO.puts("Training with weight decay:")

    {_, _} =
      Enum.reduce(1..5, {wd_state, nil}, fn step, {s, _} ->
        {new_s, metrics} = wd_step_fn.(s, {x_train, y_train})

        IO.puts(
          "  Step #{step}: loss=#{Float.round(metrics["loss"], 6)}, " <>
            "weight_penalty=#{Float.round(metrics["weight_penalty"], 6)}"
        )

        {new_s, metrics}
      end)

    IO.puts("")

    # ============================================================================
    # Part 6: Curriculum Learning with Weight Schedules
    # ============================================================================

    IO.puts("--- Part 6: Curriculum Learning ---")
    IO.puts("Using weight_schedule/1 for dynamic penalty adjustment\n")

    # Define schedule: ramp up L2, warm up L1
    schedule_fn =
      AxonIntegration.weight_schedule(%{
        # Linear ramp from 0 to 0.02
        l2: {:linear, 0.0, 0.02},
        # Warm up to 0.005 over 5 epochs
        l1: {:warmup, 0.005, 5}
      })

    # Show schedule progression
    total_epochs = 20
    IO.puts("Weight schedule over #{total_epochs} epochs:")

    for epoch <- [0, 5, 10, 15, 19] do
      weights = schedule_fn.(epoch, total_epochs)

      IO.puts(
        "  Epoch #{epoch}: L2=#{Float.round(weights.l2, 4)}, L1=#{Float.round(weights.l1, 4)}"
      )
    end

    IO.puts("")

    # Simulate curriculum training
    IO.puts("Simulated curriculum training:")

    curriculum_pipeline =
      NxPenalties.pipeline([
        {:l2, weight: 0.0},
        {:l1, weight: 0.0}
      ])

    {c_init_fn, _c_step_fn} =
      AxonIntegration.build_train_step(
        model,
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        curriculum_pipeline,
        sgd_optimizer.(0.01)
      )

    c_state = c_init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

    # Train with scheduled weights
    Enum.reduce(0..9, c_state, fn epoch, s ->
      # Update pipeline weights for this epoch
      weights = schedule_fn.(epoch, 10)

      updated_pipeline =
        curriculum_pipeline
        |> NxPenalties.Pipeline.update_weight(:l2, weights.l2)
        |> NxPenalties.Pipeline.update_weight(:l1, weights.l1)

      # Rebuild step function with updated pipeline
      {_, epoch_step_fn} =
        AxonIntegration.build_train_step(
          model,
          fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
          updated_pipeline,
          sgd_optimizer.(0.01)
        )

      # Run one epoch (single batch for demo)
      {new_s, metrics} = epoch_step_fn.(s, {x_train, y_train})

      if rem(epoch, 3) == 0 do
        IO.puts(
          "  Epoch #{epoch}: loss=#{Float.round(metrics["loss"], 4)}, " <>
            "L2_w=#{Float.round(weights.l2, 4)}, L1_w=#{Float.round(weights.l1, 4)}"
        )
      end

      new_s
    end)

    IO.puts("")

    # ============================================================================
    # Part 7: Activity Regularization
    # ============================================================================

    IO.puts("--- Part 7: Activity Regularization ---")
    IO.puts("Regularizing intermediate layer activations\n")

    # Build model with activity regularization markers
    activity_model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(64, activation: :relu, name: "hidden1")
      |> AxonIntegration.capture_activation(:hidden1)
      |> Axon.dense(32, activation: :relu, name: "hidden2")
      |> AxonIntegration.capture_activation(:hidden2)
      |> Axon.dense(1, name: "output")

    activity_penalties = %{
      hidden1: {&NxPenalties.l1/2, weight: 0.001},
      hidden2: {&NxPenalties.l1/2, weight: 0.0005}
    }

    {act_init_fn, act_step_fn} =
      AxonIntegration.build_activity_train_step(
        activity_model,
        fn y_true, y_pred -> Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2)) end,
        activity_penalties,
        sgd_optimizer.(0.01)
      )

    act_state = act_init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())
    {_, act_metrics} = act_step_fn.(act_state, {x_train, y_train})

    IO.puts(
      "Activity loss components: base=#{Float.round(act_metrics["base_loss"], 4)}, " <>
        "activity=#{Float.round(act_metrics["activity_loss"], 6)}"
    )

    IO.puts("")

    # ============================================================================
    # Part 8: Complete Training Example
    # ============================================================================

    IO.puts("--- Part 8: Complete Training Loop ---")
    IO.puts("Full training with all features combined\n")

    # Final model
    final_model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(64, activation: :relu)
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(1)

    # Comprehensive pipeline
    final_pipeline =
      NxPenalties.pipeline([
        {:l2, weight: 0.01},
        {:l1, weight: 0.001}
      ])

    # Loss function with regularization
    base_loss_fn = fn y_true, y_pred ->
      Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
    end

    {final_init_fn, final_step_fn} =
      AxonIntegration.build_train_step(
        final_model,
        base_loss_fn,
        final_pipeline,
        sgd_optimizer.(0.01)
      )

    final_state = final_init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

    # Training loop
    IO.puts("Training 20 steps:")

    {_trained_state, history} =
      Enum.reduce(1..20, {final_state, []}, fn step, {s, hist} ->
        {new_s, metrics} = final_step_fn.(s, {x_train, y_train})
        new_hist = [metrics["loss"] | hist]

        if rem(step, 5) == 0 do
          IO.puts("  Step #{step}: loss=#{Float.round(metrics["loss"], 4)}")
        end

        {new_s, new_hist}
      end)

    history = Enum.reverse(history)
    initial_loss = List.first(history)
    final_loss = List.last(history)
    improvement = (initial_loss - final_loss) / initial_loss * 100

    IO.puts("\nTraining summary:")
    IO.puts("  Initial loss: #{Float.round(initial_loss, 4)}")
    IO.puts("  Final loss: #{Float.round(final_loss, 4)}")
    IO.puts("  Improvement: #{Float.round(improvement, 1)}%")

  {:error, _} ->
    IO.puts("""
    Axon is not available.

    To run this example, add Axon to your dependencies:

        {:axon, "~> 0.6"}

    Then run: mix deps.get && mix run examples/axon_full_integration.exs
    """)
end

IO.puts("\n=== Done ===")
