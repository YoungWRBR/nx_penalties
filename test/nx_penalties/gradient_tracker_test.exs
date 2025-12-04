defmodule NxPenalties.GradientTrackerTest do
  use ExUnit.Case, async: true

  alias NxPenalties.GradientTracker

  describe "compute_grad_norm/2" do
    test "computes correct L2 norm for L1 penalty gradient" do
      # Gradient of sum(|x|) is sign(x)
      loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
      tensor = Nx.tensor([1.0, -2.0, 3.0])

      norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

      # sign([1, -2, 3]) = [1, -1, 1], L2 norm = sqrt(3)
      assert_in_delta norm, :math.sqrt(3), 1.0e-5
    end

    test "computes correct L2 norm for L2 penalty gradient" do
      # Gradient of sum(x²) is 2x
      loss_fn = fn x -> Nx.sum(Nx.pow(x, 2)) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

      # 2*[1, 2, 3] = [2, 4, 6], L2 norm = sqrt(4 + 16 + 36) = sqrt(56)
      assert_in_delta norm, :math.sqrt(56), 1.0e-5
    end

    test "handles non-differentiable operations gracefully" do
      import ExUnit.CaptureLog

      # Nx.argmax is not differentiable (returns discrete index, no gradient possible)
      loss_fn = fn x -> Nx.argmax(x) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      # Should return nil instead of crashing, and log a warning
      {norm, log} =
        with_log(fn ->
          GradientTracker.compute_grad_norm(loss_fn, tensor)
        end)

      assert norm == nil
      assert log =~ "Gradient computation failed"
      assert log =~ "non-differentiable"
    end

    test "handles multidimensional tensors" do
      loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
      tensor = Nx.tensor([[1.0, -2.0], [-3.0, 4.0]])

      norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

      # sign of each element, L2 norm = sqrt(4) = 2
      assert_in_delta norm, 2.0, 1.0e-5
    end

    test "computes zero gradient norm for constant function" do
      loss_fn = fn _x -> Nx.tensor(0.0) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

      assert_in_delta norm, 0.0, 1.0e-5
    end

    test "handles scalar input" do
      loss_fn = fn x -> Nx.pow(x, 2) end
      tensor = Nx.tensor(3.0)

      norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

      # gradient of x² at x=3 is 2*3=6, L2 norm = 6
      assert_in_delta norm, 6.0, 1.0e-5
    end
  end

  describe "pipeline_grad_norms/2" do
    test "computes norms for all pipeline entries" do
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 0.001},
          {:l2, weight: 0.01}
        ])

      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

      assert Map.has_key?(norms, "l1_grad_norm")
      assert Map.has_key?(norms, "l2_grad_norm")
      assert norms["l1_grad_norm"] > 0
      assert norms["l2_grad_norm"] > 0
    end

    test "skips disabled entries" do
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 0.001},
          {:l2, weight: 0.01}
        ])
        |> NxPenalties.Pipeline.set_enabled(:l2, false)

      tensor = Nx.tensor([1.0, 2.0, 3.0])
      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

      assert Map.has_key?(norms, "l1_grad_norm")
      refute Map.has_key?(norms, "l2_grad_norm")
    end

    test "handles empty pipeline" do
      pipeline = NxPenalties.Pipeline.new()
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

      assert norms == %{}
    end

    test "computes correct norm for L1" do
      pipeline = NxPenalties.pipeline([{:l1, weight: 1.0}])
      tensor = Nx.tensor([1.0, 1.0, 1.0])

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

      # L1 gradient is sign(x) = [1, 1, 1], L2 norm = sqrt(3)
      assert_in_delta norms["l1_grad_norm"], :math.sqrt(3), 1.0e-5
    end
  end

  describe "total_grad_norm/2" do
    test "combines weighted gradients correctly" do
      pipeline = NxPenalties.pipeline([{:l1, weight: 1.0}])
      tensor = Nx.tensor([1.0, 1.0, 1.0])

      norm = GradientTracker.total_grad_norm(pipeline, tensor)

      # L1 gradient is sign(x) = [1, 1, 1], L2 norm = sqrt(3)
      assert_in_delta norm, :math.sqrt(3), 1.0e-5
    end

    test "handles multiple penalties with weights" do
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 0.5},
          {:l2, weight: 0.5}
        ])

      # Use all non-zero values to avoid edge cases with sign(0)
      tensor = Nx.tensor([1.0, 1.0])

      norm = GradientTracker.total_grad_norm(pipeline, tensor)

      # L1 gradient: 0.5 * sign([1, 1]) = [0.5, 0.5]
      # L2 gradient: 0.5 * 2 * [1, 1] = [1, 1]
      # Combined: [1.5, 1.5], norm = sqrt(1.5² + 1.5²) = sqrt(4.5)
      assert_in_delta norm, :math.sqrt(4.5), 1.0e-5
    end

    test "handles empty pipeline" do
      pipeline = NxPenalties.Pipeline.new()
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norm = GradientTracker.total_grad_norm(pipeline, tensor)

      # Empty pipeline returns zero gradient, norm = 0
      assert_in_delta norm, 0.0, 1.0e-5
    end

    test "skips disabled entries" do
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 1.0},
          {:l2, weight: 1.0}
        ])
        |> NxPenalties.Pipeline.set_enabled(:l2, false)

      tensor = Nx.tensor([1.0, 1.0, 1.0])

      norm = GradientTracker.total_grad_norm(pipeline, tensor)

      # Only L1: norm = sqrt(3)
      assert_in_delta norm, :math.sqrt(3), 1.0e-5
    end

    test "skips non-differentiable entries" do
      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 1.0)
        |> NxPenalties.Pipeline.add(:nondiff, fn x, _ -> Nx.argmax(x) end,
          weight: 1.0,
          differentiable: false
        )

      tensor = Nx.tensor([1.0, 1.0, 1.0])

      # Should only compute gradient for l1, skipping nondiff entirely (no warning)
      norm = GradientTracker.total_grad_norm(pipeline, tensor)

      # Only L1: norm = sqrt(3)
      assert_in_delta norm, :math.sqrt(3), 1.0e-5
    end
  end

  describe "differentiable option" do
    test "pipeline_grad_norms skips non-differentiable penalties" do
      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 1.0)
        |> NxPenalties.Pipeline.add(:nondiff, fn x, _ -> Nx.argmax(x) end,
          weight: 1.0,
          differentiable: false
        )

      tensor = Nx.tensor([1.0, 2.0, 3.0])

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

      # Should have l1_grad_norm but NOT nondiff_grad_norm
      assert Map.has_key?(norms, "l1_grad_norm")
      refute Map.has_key?(norms, "nondiff_grad_norm")
    end

    test "differentiable defaults to true" do
      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 1.0)

      # Should be differentiable by default
      assert get_in(pipeline.meta, [:l1, :differentiable]) == true
    end

    test "differentiable: false is stored in meta" do
      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:nondiff, fn x, _ -> x end,
          weight: 1.0,
          differentiable: false
        )

      assert get_in(pipeline.meta, [:nondiff, :differentiable]) == false
    end

    test "remove cleans up meta" do
      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2,
          weight: 1.0,
          differentiable: false
        )
        |> NxPenalties.Pipeline.remove(:l1)

      refute Map.has_key?(pipeline.meta, :l1)
    end
  end
end
