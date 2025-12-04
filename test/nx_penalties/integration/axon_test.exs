defmodule NxPenalties.Integration.AxonTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  # Only run tests if Axon is available
  @moduletag :axon

  setup_all do
    case Code.ensure_loaded(Axon) do
      {:module, _} -> :ok
      {:error, _} -> :skip
    end
  end

  alias NxPenalties.Integration.Axon, as: AxonIntegration

  # Helper to create a simple SGD optimizer
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

  defp deep_map_single(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)

  defp deep_map_single(structure, fun) when is_map(structure) do
    Map.new(structure, fn {key, value} ->
      {key, deep_map_single(value, fun)}
    end)
  end

  # ============================================================================
  # wrap_loss/3
  # ============================================================================

  describe "wrap_loss/3" do
    test "wraps loss function with penalty" do
      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      wrapped =
        AxonIntegration.wrap_loss(
          base_loss,
          &NxPenalties.l2/2,
          lambda: 0.01
        )

      y_true = Nx.tensor([[1.0], [2.0]])
      y_pred = Nx.tensor([[1.0], [2.0]])

      result = wrapped.(y_true, y_pred)

      # Base loss = 0 (perfect prediction)
      # L2 penalty on y_pred = 0.01 * (1^2 + 2^2) / 2 = 0.01 * 2.5 = 0.025
      assert_scalar(result)
      assert Nx.to_number(result) > 0
    end

    test "uses default lambda value" do
      base_loss = fn _, _y_pred -> Nx.tensor(0.0) end
      wrapped = AxonIntegration.wrap_loss(base_loss, &NxPenalties.l1/2)

      result = wrapped.(Nx.tensor([1.0]), Nx.tensor([2.0]))
      assert_scalar(result)
    end

    test "passes penalty_opts to penalty function" do
      base_loss = fn _, _ -> Nx.tensor(0.0) end

      wrapped =
        AxonIntegration.wrap_loss(
          base_loss,
          &NxPenalties.l2/2,
          lambda: 1.0,
          penalty_opts: [lambda: 2.0]
        )

      result = wrapped.(Nx.tensor([0.0]), Nx.tensor([1.0]))
      # L2 with lambda=2.0: 2.0 * 1^2 / 1 = 2.0
      # Total: 0 + 1.0 * 2.0 = 2.0
      assert_close(result, Nx.tensor(2.0))
    end
  end

  # ============================================================================
  # wrap_loss_with_pipeline/3
  # ============================================================================

  describe "wrap_loss_with_pipeline/3" do
    test "wraps loss with pipeline" do
      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      pipeline =
        NxPenalties.pipeline([
          {:l2, weight: 0.01}
        ])

      wrapped = AxonIntegration.wrap_loss_with_pipeline(base_loss, pipeline)

      y_true = Nx.tensor([[1.0]])
      y_pred = Nx.tensor([[1.0]])

      result = wrapped.(y_true, y_pred)
      assert_scalar(result)
    end

    test "combines multiple penalties from pipeline" do
      base_loss = fn _, _ -> Nx.tensor(0.0) end

      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 1.0},
          {:l2, weight: 1.0}
        ])

      wrapped = AxonIntegration.wrap_loss_with_pipeline(base_loss, pipeline)

      y_pred = Nx.tensor([2.0])
      result = wrapped.(Nx.tensor([0.0]), y_pred)

      # L1: 1.0 * 2.0 = 2.0
      # L2: 1.0 * 4.0 / 1 = 4.0
      # Total: 0 + 2.0 + 4.0 = 6.0
      assert_close(result, Nx.tensor(6.0))
    end
  end

  # ============================================================================
  # wrap_loss_with_params/3
  # ============================================================================

  describe "wrap_loss_with_params/3" do
    test "creates loss function with parameter penalty" do
      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      param_penalty = fn params ->
        params
        |> Map.values()
        |> Enum.map(&Nx.sum(Nx.pow(&1, 2)))
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      end

      wrapped =
        AxonIntegration.wrap_loss_with_params(
          base_loss,
          param_penalty,
          lambda: 0.1
        )

      y_true = Nx.tensor([1.0])
      y_pred = Nx.tensor([1.0])
      params = %{w: Nx.tensor([2.0])}

      result = wrapped.(y_true, y_pred, params)

      # Base loss = 0
      # Param penalty = 4.0 (2^2)
      # Total = 0 + 0.1 * 4 = 0.4
      assert_close(result, Nx.tensor(0.4))
    end
  end

  # ============================================================================
  # build_train_step/4
  # ============================================================================

  describe "build_train_step/4" do
    @tag :slow
    test "returns init and step functions" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])
      optimizer = sgd_optimizer(0.01)

      {init_fn, step_fn} =
        AxonIntegration.build_train_step(
          model,
          base_loss,
          pipeline,
          optimizer
        )

      assert is_function(init_fn, 2)
      assert is_function(step_fn, 2)
    end

    @tag :slow
    test "init_fn creates proper state structure" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])
      optimizer = sgd_optimizer(0.01)

      {init_fn, _step_fn} =
        AxonIntegration.build_train_step(
          model,
          base_loss,
          pipeline,
          optimizer
        )

      state = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())

      assert Map.has_key?(state, :model_state)
      assert Map.has_key?(state, :optimizer_state)
    end

    @tag :slow
    test "step_fn returns metrics with base_loss, penalty_total, and loss" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])
      optimizer = sgd_optimizer(0.01)

      {init_fn, step_fn} =
        AxonIntegration.build_train_step(
          model,
          base_loss,
          pipeline,
          optimizer
        )

      state = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())

      x = Nx.tensor([[1.0, 2.0]])
      y = Nx.tensor([[0.5]])

      {_new_state, metrics} = step_fn.(state, {x, y})

      assert Map.has_key?(metrics, "base_loss")
      assert Map.has_key?(metrics, "penalty_total")
      assert Map.has_key?(metrics, "loss")
      assert is_number(metrics["loss"])
    end
  end

  # ============================================================================
  # build_train_step_with_weight_decay/5
  # ============================================================================

  describe "build_train_step_with_weight_decay/5" do
    @tag :slow
    test "returns init and step functions" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      param_penalty = fn params ->
        params
        |> AxonIntegration.flatten_params()
        |> Enum.map(&NxPenalties.l2(&1, lambda: 1.0))
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      end

      optimizer = sgd_optimizer(0.01)

      {init_fn, step_fn} =
        AxonIntegration.build_train_step_with_weight_decay(
          model,
          base_loss,
          param_penalty,
          optimizer,
          lambda: 0.001
        )

      assert is_function(init_fn, 2)
      assert is_function(step_fn, 2)
    end

    @tag :slow
    test "step_fn returns weight_penalty in metrics" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      param_penalty = fn params ->
        params
        |> AxonIntegration.flatten_params()
        |> Enum.map(&NxPenalties.l2(&1, lambda: 1.0))
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      end

      optimizer = sgd_optimizer(0.01)

      {init_fn, step_fn} =
        AxonIntegration.build_train_step_with_weight_decay(
          model,
          base_loss,
          param_penalty,
          optimizer,
          lambda: 0.001
        )

      state = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())

      x = Nx.tensor([[1.0, 2.0]])
      y = Nx.tensor([[0.5]])

      {_new_state, metrics} = step_fn.(state, {x, y})

      assert Map.has_key?(metrics, "weight_penalty")
      assert is_number(metrics["weight_penalty"])
    end
  end

  # ============================================================================
  # weight_schedule/1
  # ============================================================================

  describe "weight_schedule/1" do
    test "linear schedule interpolates correctly" do
      schedule_fn =
        AxonIntegration.weight_schedule(%{
          penalty: {:linear, 0.0, 1.0}
        })

      # At epoch 0 of 10
      weights = schedule_fn.(0, 10)
      assert_in_delta weights.penalty, 0.0, 0.001

      # At epoch 9 of 10 (last epoch)
      weights = schedule_fn.(9, 10)
      assert_in_delta weights.penalty, 1.0, 0.001

      # At midpoint
      weights = schedule_fn.(4, 10)
      assert_in_delta weights.penalty, 0.444, 0.01
    end

    test "warmup schedule ramps up" do
      schedule_fn =
        AxonIntegration.weight_schedule(%{
          penalty: {:warmup, 1.0, 5}
        })

      # Before warmup completes
      weights = schedule_fn.(2, 10)
      assert_in_delta weights.penalty, 0.4, 0.001

      # After warmup
      weights = schedule_fn.(5, 10)
      assert_in_delta weights.penalty, 1.0, 0.001

      weights = schedule_fn.(8, 10)
      assert_in_delta weights.penalty, 1.0, 0.001
    end

    test "decay schedule decreases exponentially" do
      schedule_fn =
        AxonIntegration.weight_schedule(%{
          penalty: {:decay, 1.0, 0.5}
        })

      weights_0 = schedule_fn.(0, 10)
      weights_1 = schedule_fn.(1, 10)
      weights_2 = schedule_fn.(2, 10)

      assert_in_delta weights_0.penalty, 1.0, 0.001
      assert_in_delta weights_1.penalty, 0.5, 0.001
      assert_in_delta weights_2.penalty, 0.25, 0.001
    end

    test "constant schedule returns fixed value" do
      schedule_fn =
        AxonIntegration.weight_schedule(%{
          penalty: {:constant, 0.5}
        })

      for epoch <- 0..9 do
        weights = schedule_fn.(epoch, 10)
        assert_in_delta weights.penalty, 0.5, 0.001
      end
    end

    test "handles multiple schedules" do
      schedule_fn =
        AxonIntegration.weight_schedule(%{
          l1: {:linear, 0.0, 0.1},
          l2: {:constant, 0.01},
          kl: {:warmup, 0.5, 5}
        })

      weights = schedule_fn.(3, 10)

      assert Map.has_key?(weights, :l1)
      assert Map.has_key?(weights, :l2)
      assert Map.has_key?(weights, :kl)
    end
  end

  # ============================================================================
  # apply_scheduled_weights/2
  # ============================================================================

  describe "apply_scheduled_weights/2" do
    test "updates pipeline weights" do
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 0.0},
          {:l2, weight: 0.0}
        ])

      weights = %{l1: 0.1, l2: 0.2}

      updated = AxonIntegration.apply_scheduled_weights(pipeline, weights)

      # Verify weights were updated by computing
      tensor = Nx.tensor([1.0])
      {total, _} = NxPenalties.Pipeline.compute(updated, tensor)

      # L1: 0.1 * 1.0 = 0.1
      # L2: 0.2 * 1.0 = 0.2
      # Total: 0.3
      assert_close(total, Nx.tensor(0.3))
    end
  end

  # ============================================================================
  # flatten_params/1
  # ============================================================================

  describe "flatten_params/1" do
    test "flattens tensor to list" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = AxonIntegration.flatten_params(tensor)

      assert length(result) == 1
      assert Nx.shape(hd(result)) == {4}
    end

    test "flattens nested map" do
      params = %{
        layer1: %{w: Nx.tensor([1.0, 2.0])},
        layer2: %{w: Nx.tensor([3.0])}
      }

      result = AxonIntegration.flatten_params(params)

      assert length(result) == 2
      assert Enum.all?(result, &is_struct(&1, Nx.Tensor))
    end

    test "handles deeply nested maps" do
      params = %{
        a: %{
          b: %{
            c: Nx.tensor([1.0])
          }
        }
      }

      result = AxonIntegration.flatten_params(params)
      assert length(result) == 1
    end

    test "returns empty list for non-tensor values" do
      assert AxonIntegration.flatten_params("string") == []
      assert AxonIntegration.flatten_params(123) == []
    end
  end

  # ============================================================================
  # capture_activation/2 and extract_captures/1
  # ============================================================================

  describe "capture_activation/2 and extract_captures/1" do
    @tag :slow
    test "capture_activation adds layer to model" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> AxonIntegration.capture_activation(:hidden)
        |> Axon.dense(1)

      # Model should compile and run
      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      state = init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 10})
      %{prediction: output, state: _forward_state} = predict_fn.(state, input)

      # Output should have correct shape
      assert Nx.shape(output) == {1, 1}
    end
  end

  # ============================================================================
  # trainer/5
  # ============================================================================

  describe "trainer/5" do
    @tag :slow
    test "creates Axon.Loop with regularization" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      optimizer = Polaris.Optimizers.sgd(learning_rate: 0.01)

      loop = AxonIntegration.trainer(model, base_loss, pipeline, optimizer)

      assert %Axon.Loop{} = loop
    end

    @tag :slow
    test "respects log_penalties option" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])

      base_loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      optimizer = Polaris.Optimizers.sgd(learning_rate: 0.01)

      loop =
        AxonIntegration.trainer(
          model,
          base_loss,
          pipeline,
          optimizer,
          log_penalties: false
        )

      assert %Axon.Loop{} = loop
    end
  end

  # ============================================================================
  # log_penalties/3
  # ============================================================================

  describe "log_penalties/3" do
    @tag :slow
    test "adds event handler to loop" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      pipeline = NxPenalties.pipeline([{:l2, weight: 0.01}])

      loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      optimizer = Polaris.Optimizers.sgd(learning_rate: 0.01)

      loop =
        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> AxonIntegration.log_penalties(pipeline)

      assert %Axon.Loop{} = loop
    end
  end

  # ============================================================================
  # schedule_weights/4
  # ============================================================================

  describe "schedule_weights/4" do
    @tag :slow
    test "adds epoch handler to loop" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1)

      schedule_fn =
        AxonIntegration.weight_schedule(%{
          l2: {:linear, 0.0, 0.01}
        })

      loss = fn y_true, y_pred ->
        Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
      end

      optimizer = Polaris.Optimizers.sgd(learning_rate: 0.01)

      loop =
        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> AxonIntegration.schedule_weights(:pipeline, schedule_fn, total_epochs: 10)

      assert %Axon.Loop{} = loop
    end
  end
end
