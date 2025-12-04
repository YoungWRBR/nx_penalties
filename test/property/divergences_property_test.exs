defmodule NxPenalties.DivergencesPropertyTest do
  @moduledoc """
  Property-based tests for divergence and entropy functions.

  These tests verify mathematical invariants for information-theoretic measures.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias NxPenalties.Divergences

  # Helper to create normalized log-probabilities
  defp normalize_logprobs(tensor) do
    Nx.subtract(tensor, Nx.logsumexp(tensor, axes: [-1], keep_axes: true))
  end

  # Generators
  defp logprobs_generator(size) do
    gen all(values <- list_of(float(min: -5.0, max: 5.0), length: size)) do
      tensor = Nx.tensor(values, type: :f32)
      normalize_logprobs(tensor)
    end
  end

  # KL Divergence Properties
  describe "KL divergence properties" do
    property "KL divergence is non-negative" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        result = Divergences.kl_divergence(p, q)
        # Allow small negative values due to numerical precision
        assert Nx.to_number(result) >= -1.0e-5
      end
    end

    property "KL(P || P) = 0 (identity)" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size)
            ) do
        result = Divergences.kl_divergence(p, p)
        assert_in_delta Nx.to_number(result), 0.0, 1.0e-5
      end
    end

    property "KL is asymmetric (KL(P||Q) != KL(Q||P) in general)" do
      # This is a probabilistic property - we expect asymmetry for most random distributions
      check all(
              size <- integer(4..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        kl_pq = Divergences.kl_divergence(p, q) |> Nx.to_number()
        kl_qp = Divergences.kl_divergence(q, p) |> Nx.to_number()

        # Both should be non-negative
        assert kl_pq >= -1.0e-5
        assert kl_qp >= -1.0e-5

        # Note: We don't assert they're different since identical distributions would give same value
      end
    end
  end

  # JS Divergence Properties
  describe "JS divergence properties" do
    property "JS divergence is non-negative" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        result = Divergences.js_divergence(p, q)
        assert Nx.to_number(result) >= -1.0e-5
      end
    end

    property "JS divergence is symmetric" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        js_pq = Divergences.js_divergence(p, q) |> Nx.to_number()
        js_qp = Divergences.js_divergence(q, p) |> Nx.to_number()

        assert_in_delta js_pq, js_qp, 1.0e-5
      end
    end

    property "JS(P || P) = 0 (identity)" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size)
            ) do
        result = Divergences.js_divergence(p, p)
        assert_in_delta Nx.to_number(result), 0.0, 1.0e-5
      end
    end

    property "JS is bounded by ln(2)" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        result = Divergences.js_divergence(p, q) |> Nx.to_number()
        # JS bounded by log(2) ≈ 0.693 for natural log
        assert result <= :math.log(2) + 1.0e-4
      end
    end
  end

  # Entropy Properties
  describe "Entropy properties" do
    property "entropy is non-negative (bonus mode)" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        result = Divergences.entropy(logprobs, mode: :bonus)
        assert Nx.to_number(result) >= -1.0e-5
      end
    end

    property "entropy is non-positive (penalty mode)" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        result = Divergences.entropy(logprobs, mode: :penalty)
        assert Nx.to_number(result) <= 1.0e-5
      end
    end

    property "penalty mode negates bonus mode" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        bonus = Divergences.entropy(logprobs, mode: :bonus) |> Nx.to_number()
        penalty = Divergences.entropy(logprobs, mode: :penalty) |> Nx.to_number()

        assert_in_delta penalty, -bonus, 1.0e-5
      end
    end

    property "entropy is bounded by log(n)" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        result = Divergences.entropy(logprobs, mode: :bonus) |> Nx.to_number()
        max_entropy = :math.log(size)
        assert result <= max_entropy + 1.0e-4
      end
    end

    property "normalized entropy is in [0, 1]" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        result = Divergences.entropy(logprobs, mode: :bonus, normalize: true) |> Nx.to_number()
        assert result >= -1.0e-5
        assert result <= 1.0 + 1.0e-5
      end
    end

    property "uniform distribution has maximum entropy" do
      check all(size <- integer(2..10)) do
        # Create uniform distribution
        log_prob = :math.log(1.0 / size)
        uniform_logprobs = Nx.broadcast(Nx.tensor(log_prob), {size})

        result = Divergences.entropy(uniform_logprobs, mode: :bonus) |> Nx.to_number()
        max_entropy = :math.log(size)

        assert_in_delta result, max_entropy, 1.0e-4
      end
    end
  end

  # Gradient Properties
  describe "Gradient properties" do
    property "KL gradient has correct shape" do
      check all(
              size <- integer(2..10),
              p <- logprobs_generator(size),
              q <- logprobs_generator(size)
            ) do
        grad_fn = Nx.Defn.grad(fn x -> Divergences.kl_divergence(x, q) end)
        grads = grad_fn.(p)
        assert Nx.shape(grads) == Nx.shape(p)
      end
    end

    property "entropy gradient has correct shape" do
      check all(
              size <- integer(2..10),
              logprobs <- logprobs_generator(size)
            ) do
        grad_fn = Nx.Defn.grad(fn x -> Divergences.entropy(x) end)
        grads = grad_fn.(logprobs)
        assert Nx.shape(grads) == Nx.shape(logprobs)
      end
    end
  end
end
