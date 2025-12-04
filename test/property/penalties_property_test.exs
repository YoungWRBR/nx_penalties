defmodule NxPenalties.PenaltiesPropertyTest do
  @moduledoc """
  Property-based tests for penalty functions.

  These tests verify mathematical invariants that should hold for all inputs.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias NxPenalties.Penalties

  # Generators
  defp tensor_1d_generator do
    gen all(
          size <- integer(1..20),
          values <- list_of(float(min: -100.0, max: 100.0), length: size)
        ) do
      Nx.tensor(values, type: :f32)
    end
  end

  defp tensor_2d_generator do
    gen all(
          rows <- integer(1..10),
          cols <- integer(1..10),
          values <- list_of(float(min: -100.0, max: 100.0), length: rows * cols)
        ) do
      Nx.tensor(values, type: :f32) |> Nx.reshape({rows, cols})
    end
  end

  defp positive_float_generator do
    gen all(f <- float(min: 0.001, max: 10.0)) do
      f
    end
  end

  defp l1_ratio_generator do
    gen all(f <- float(min: 0.0, max: 1.0)) do
      f
    end
  end

  # L1 Properties
  describe "L1 penalty properties" do
    property "L1 is always non-negative" do
      check all(tensor <- tensor_1d_generator()) do
        result = Penalties.l1(tensor, lambda: 1.0)
        assert Nx.to_number(result) >= 0
      end
    end

    property "L1 of zero tensor is zero" do
      check all(size <- integer(1..20)) do
        tensor = Nx.broadcast(0.0, {size})
        result = Penalties.l1(tensor, lambda: 1.0)
        assert_in_delta Nx.to_number(result), 0.0, 1.0e-6
      end
    end

    property "L1 scales linearly with lambda" do
      check all(
              tensor <- tensor_1d_generator(),
              lambda <- positive_float_generator()
            ) do
        # L1(x, lambda) = lambda * L1(x, 1.0)
        result_scaled = Penalties.l1(tensor, lambda: lambda)
        result_unscaled = Penalties.l1(tensor, lambda: 1.0)
        expected = Nx.multiply(result_unscaled, lambda)

        assert_in_delta Nx.to_number(result_scaled), Nx.to_number(expected), 1.0e-4
      end
    end

    property "L1(x) = L1(-x) (symmetry)" do
      check all(tensor <- tensor_1d_generator()) do
        result_pos = Penalties.l1(tensor, lambda: 1.0)
        result_neg = Penalties.l1(Nx.negate(tensor), lambda: 1.0)

        assert_in_delta Nx.to_number(result_pos), Nx.to_number(result_neg), 1.0e-5
      end
    end

    property "L1 triangle inequality: L1(x + y) <= L1(x) + L1(y)" do
      check all(
              size <- integer(1..10),
              x_values <- list_of(float(min: -10.0, max: 10.0), length: size),
              y_values <- list_of(float(min: -10.0, max: 10.0), length: size)
            ) do
        x = Nx.tensor(x_values, type: :f32)
        y = Nx.tensor(y_values, type: :f32)

        l1_sum = Penalties.l1(Nx.add(x, y), lambda: 1.0)
        l1_x = Penalties.l1(x, lambda: 1.0)
        l1_y = Penalties.l1(y, lambda: 1.0)

        # Allow small tolerance for floating point
        assert Nx.to_number(l1_sum) <= Nx.to_number(l1_x) + Nx.to_number(l1_y) + 1.0e-5
      end
    end
  end

  # L2 Properties
  describe "L2 penalty properties" do
    property "L2 is always non-negative" do
      check all(tensor <- tensor_1d_generator()) do
        result = Penalties.l2(tensor, lambda: 1.0)
        assert Nx.to_number(result) >= 0
      end
    end

    property "L2 of zero tensor is zero" do
      check all(size <- integer(1..20)) do
        tensor = Nx.broadcast(0.0, {size})
        result = Penalties.l2(tensor, lambda: 1.0)
        assert_in_delta Nx.to_number(result), 0.0, 1.0e-6
      end
    end

    property "L2 scales linearly with lambda" do
      check all(
              tensor <- tensor_1d_generator(),
              lambda <- positive_float_generator()
            ) do
        result_scaled = Penalties.l2(tensor, lambda: lambda)
        result_unscaled = Penalties.l2(tensor, lambda: 1.0)
        expected = Nx.multiply(result_unscaled, lambda)

        assert_in_delta Nx.to_number(result_scaled), Nx.to_number(expected), 1.0e-4
      end
    end

    property "L2(x) = L2(-x) (symmetry)" do
      check all(tensor <- tensor_1d_generator()) do
        result_pos = Penalties.l2(tensor, lambda: 1.0)
        result_neg = Penalties.l2(Nx.negate(tensor), lambda: 1.0)

        assert_in_delta Nx.to_number(result_pos), Nx.to_number(result_neg), 1.0e-5
      end
    end

    property "L2(k*x) = k^2 * L2(x) (quadratic scaling)" do
      check all(
              tensor <- tensor_1d_generator(),
              k <- float(min: 0.1, max: 10.0)
            ) do
        result_scaled = Penalties.l2(Nx.multiply(tensor, k), lambda: 1.0)
        result_original = Penalties.l2(tensor, lambda: 1.0)
        expected = Nx.multiply(result_original, k * k)

        # Use relative tolerance for large values
        expected_val = Nx.to_number(expected)
        result_val = Nx.to_number(result_scaled)
        tolerance = max(1.0e-3, abs(expected_val) * 1.0e-5)

        assert_in_delta result_val, expected_val, tolerance
      end
    end
  end

  # Elastic Net Properties
  describe "Elastic Net penalty properties" do
    property "elastic_net with l1_ratio=1.0 equals L1" do
      check all(tensor <- tensor_1d_generator()) do
        elastic = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: 1.0)
        l1 = Penalties.l1(tensor, lambda: 1.0)

        assert_in_delta Nx.to_number(elastic), Nx.to_number(l1), 1.0e-5
      end
    end

    property "elastic_net with l1_ratio=0.0 equals L2" do
      check all(tensor <- tensor_1d_generator()) do
        elastic = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: 0.0)
        l2 = Penalties.l2(tensor, lambda: 1.0)

        assert_in_delta Nx.to_number(elastic), Nx.to_number(l2), 1.0e-5
      end
    end

    property "elastic_net is always non-negative" do
      check all(
              tensor <- tensor_1d_generator(),
              l1_ratio <- l1_ratio_generator()
            ) do
        result = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: l1_ratio)
        assert Nx.to_number(result) >= 0
      end
    end

    property "elastic_net is bounded by L1 and L2" do
      check all(
              tensor <- tensor_1d_generator(),
              l1_ratio <- l1_ratio_generator()
            ) do
        elastic = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: l1_ratio)
        l1 = Penalties.l1(tensor, lambda: 1.0)
        l2 = Penalties.l2(tensor, lambda: 1.0)

        elastic_val = Nx.to_number(elastic)
        l1_val = Nx.to_number(l1)
        l2_val = Nx.to_number(l2)

        # elastic_net = ratio * L1 + (1 - ratio) * L2
        # So it's a convex combination, bounded by min and max of L1, L2
        min_val = min(l1_val, l2_val)
        max_val = max(l1_val, l2_val)

        assert elastic_val >= min_val - 1.0e-5
        assert elastic_val <= max_val + 1.0e-5
      end
    end

    property "elastic_net scales linearly with lambda" do
      check all(
              tensor <- tensor_1d_generator(),
              lambda <- positive_float_generator(),
              l1_ratio <- l1_ratio_generator()
            ) do
        result_scaled = Penalties.elastic_net(tensor, lambda: lambda, l1_ratio: l1_ratio)
        result_unscaled = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: l1_ratio)
        expected = Nx.multiply(result_unscaled, lambda)

        assert_in_delta Nx.to_number(result_scaled), Nx.to_number(expected), 1.0e-4
      end
    end
  end

  # Gradient Properties
  describe "Gradient properties" do
    property "L1 gradient has correct shape" do
      check all(tensor <- tensor_2d_generator()) do
        grad_fn = Nx.Defn.grad(fn x -> Penalties.l1(x, lambda: 1.0) end)
        grads = grad_fn.(tensor)
        assert Nx.shape(grads) == Nx.shape(tensor)
      end
    end

    property "L2 gradient has correct shape" do
      check all(tensor <- tensor_2d_generator()) do
        grad_fn = Nx.Defn.grad(fn x -> Penalties.l2(x, lambda: 1.0) end)
        grads = grad_fn.(tensor)
        assert Nx.shape(grads) == Nx.shape(tensor)
      end
    end

    property "elastic_net gradient has correct shape" do
      check all(
              tensor <- tensor_2d_generator(),
              l1_ratio <- l1_ratio_generator()
            ) do
        grad_fn =
          Nx.Defn.grad(fn x -> Penalties.elastic_net(x, lambda: 1.0, l1_ratio: l1_ratio) end)

        grads = grad_fn.(tensor)
        assert Nx.shape(grads) == Nx.shape(tensor)
      end
    end
  end
end
