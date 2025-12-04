defmodule NxPenalties.DivergencesTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Divergences

  describe "kl_divergence/3" do
    test "returns scalar" do
      p_logprobs = random_logprobs({4})
      q_logprobs = random_logprobs({4})
      result = Divergences.kl_divergence(p_logprobs, q_logprobs)
      assert_scalar(result)
    end

    test "identical distributions have zero divergence" do
      logprobs = random_logprobs({4})
      result = Divergences.kl_divergence(logprobs, logprobs)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "KL divergence is non-negative" do
      for _ <- 1..10 do
        p_logprobs = random_logprobs({8})
        q_logprobs = random_logprobs({8})
        result = Divergences.kl_divergence(p_logprobs, q_logprobs)
        assert Nx.to_number(result) >= -1.0e-6, "KL divergence should be non-negative"
      end
    end

    test "KL is asymmetric" do
      # Use clearly different distributions
      p_logprobs = Nx.tensor([-0.1, -1.5, -3.0, -4.0]) |> normalize_logprobs()
      q_logprobs = Nx.tensor([-2.0, -0.5, -2.0, -3.0]) |> normalize_logprobs()

      kl_pq = Divergences.kl_divergence(p_logprobs, q_logprobs) |> Nx.to_number()
      kl_qp = Divergences.kl_divergence(q_logprobs, p_logprobs) |> Nx.to_number()

      # Both should be positive
      assert kl_pq > 0
      assert kl_qp > 0
      # They should be different (asymmetric)
      refute_in_delta kl_pq, kl_qp, 0.01
    end

    test "sum reduction" do
      p_logprobs = random_logprobs({2, 4})
      q_logprobs = random_logprobs({2, 4})
      result = Divergences.kl_divergence(p_logprobs, q_logprobs, reduction: :sum)
      assert_scalar(result)
    end

    test "none reduction preserves shape" do
      p_logprobs = random_logprobs({2, 4})
      q_logprobs = random_logprobs({2, 4})
      result = Divergences.kl_divergence(p_logprobs, q_logprobs, reduction: :none)
      assert Nx.shape(result) == {2}
    end

    test "gradient flows" do
      grad_fn =
        Nx.Defn.grad(fn {p, q} ->
          Divergences.kl_divergence(p, q)
        end)

      p = random_logprobs({4})
      q = random_logprobs({4})
      {grad_p, grad_q} = grad_fn.({p, q})
      assert_finite(grad_p)
      assert_finite(grad_q)
    end
  end

  describe "js_divergence/3" do
    test "returns scalar" do
      p_logprobs = random_logprobs({4})
      q_logprobs = random_logprobs({4})
      result = Divergences.js_divergence(p_logprobs, q_logprobs)
      assert_scalar(result)
    end

    test "identical distributions have zero divergence" do
      logprobs = random_logprobs({4})
      result = Divergences.js_divergence(logprobs, logprobs)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "JS divergence is symmetric" do
      p_logprobs = random_logprobs({4})
      q_logprobs = random_logprobs({4})

      js_pq = Divergences.js_divergence(p_logprobs, q_logprobs) |> Nx.to_number()
      js_qp = Divergences.js_divergence(q_logprobs, p_logprobs) |> Nx.to_number()

      assert_in_delta js_pq, js_qp, 1.0e-5
    end

    test "JS is bounded between 0 and 1 (for base e)" do
      for _ <- 1..10 do
        p_logprobs = random_logprobs({8})
        q_logprobs = random_logprobs({8})
        result = Divergences.js_divergence(p_logprobs, q_logprobs) |> Nx.to_number()

        assert result >= -1.0e-6, "JS should be non-negative"
        # JS bounded by log(2) ≈ 0.693 for natural log
        assert result <= 0.7, "JS should be bounded"
      end
    end

    test "gradient flows" do
      grad_fn =
        Nx.Defn.grad(fn {p, q} ->
          Divergences.js_divergence(p, q)
        end)

      p = random_logprobs({4})
      q = random_logprobs({4})
      {grad_p, grad_q} = grad_fn.({p, q})
      assert_finite(grad_p)
      assert_finite(grad_q)
    end
  end

  describe "entropy/2" do
    test "returns scalar" do
      logprobs = random_logprobs({4})
      result = Divergences.entropy(logprobs)
      assert_scalar(result)
    end

    test "uniform distribution has maximum entropy" do
      # Uniform distribution: all equal probabilities
      n = 4
      uniform_logprobs = Nx.broadcast(Nx.log(Nx.tensor(1.0 / n)), {n})
      result = Divergences.entropy(uniform_logprobs)

      # Maximum entropy for n categories is log(n)
      expected = :math.log(n)
      assert_close(result, Nx.tensor(expected), atol: 1.0e-5)
    end

    test "deterministic distribution has zero entropy" do
      # One-hot (deterministic) distribution
      # Use softmax-like logprobs approaching one-hot
      logprobs = Nx.tensor([10.0, -100.0, -100.0, -100.0])
      logprobs = normalize_logprobs(logprobs)
      result = Divergences.entropy(logprobs)

      # Should be close to zero
      assert Nx.to_number(result) < 0.01
    end

    test "entropy is non-negative" do
      for _ <- 1..10 do
        logprobs = random_logprobs({8})
        result = Divergences.entropy(logprobs) |> Nx.to_number()
        assert result >= -1.0e-6
      end
    end

    test "penalty mode returns negative entropy" do
      logprobs = random_logprobs({4})

      entropy = Divergences.entropy(logprobs, mode: :bonus) |> Nx.to_number()
      penalty = Divergences.entropy(logprobs, mode: :penalty) |> Nx.to_number()

      assert_in_delta penalty, -entropy, 1.0e-5
    end

    test "mean reduction" do
      logprobs = random_logprobs({2, 4})
      result = Divergences.entropy(logprobs, reduction: :mean)
      assert_scalar(result)
    end

    test "none reduction preserves shape" do
      logprobs = random_logprobs({2, 4})
      result = Divergences.entropy(logprobs, reduction: :none)
      assert Nx.shape(result) == {2}
    end

    test "gradient flows" do
      grad_fn = Nx.Defn.grad(fn x -> Divergences.entropy(x) end)
      logprobs = random_logprobs({4})
      grads = grad_fn.(logprobs)
      assert_finite(grads)
    end
  end

  describe "entropy/2 with normalize option" do
    test "normalized uniform distribution has entropy ~1.0" do
      vocab_size = 10
      # Uniform distribution: each prob = 1/10, logprob = log(1/10) = -log(10)
      uniform_logprobs = Nx.broadcast(Nx.tensor(-:math.log(vocab_size)), {1, vocab_size})
      result = Divergences.entropy(uniform_logprobs, normalize: true, mode: :bonus)
      assert_close(result, Nx.tensor(1.0), atol: 1.0e-4)
    end

    test "normalized one-hot distribution has entropy ~0.0" do
      # One-hot: one token has prob ~1, others ~0
      one_hot_logprobs = Nx.tensor([[0.0, -100.0, -100.0, -100.0]])
      result = Divergences.entropy(one_hot_logprobs, normalize: true, mode: :bonus)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "normalize: false returns raw entropy" do
      vocab_size = 8
      uniform_logprobs = Nx.broadcast(Nx.tensor(-:math.log(vocab_size)), {1, vocab_size})
      result = Divergences.entropy(uniform_logprobs, normalize: false, mode: :bonus)
      # Raw max entropy = log(8) ≈ 2.079
      assert_close(result, Nx.tensor(:math.log(vocab_size)), atol: 1.0e-4)
    end

    test "normalized entropy is in [0, 1] range" do
      for _ <- 1..10 do
        logprobs = random_logprobs({4, 32})
        result = Divergences.entropy(logprobs, normalize: true, mode: :bonus)
        value = Nx.to_number(result)
        assert value >= 0.0 and value <= 1.0 + 1.0e-5, "Normalized entropy #{value} outside [0,1]"
      end
    end

    test "normalize works with penalty mode" do
      vocab_size = 4
      uniform_logprobs = Nx.broadcast(Nx.tensor(-:math.log(vocab_size)), {1, vocab_size})
      bonus = Divergences.entropy(uniform_logprobs, normalize: true, mode: :bonus)
      penalty = Divergences.entropy(uniform_logprobs, normalize: true, mode: :penalty)
      assert_close(Nx.negate(bonus), penalty)
    end

    test "normalize works with all reduction modes" do
      logprobs = random_logprobs({2, 8})

      mean_result = Divergences.entropy(logprobs, normalize: true, reduction: :mean)
      assert Nx.shape(mean_result) == {}

      sum_result = Divergences.entropy(logprobs, normalize: true, reduction: :sum)
      assert Nx.shape(sum_result) == {}

      none_result = Divergences.entropy(logprobs, normalize: true, reduction: :none)
      assert Nx.shape(none_result) == {2}
    end

    test "gradient flows with normalize option" do
      grad_fn = Nx.Defn.grad(fn x -> Divergences.entropy(x, normalize: true) end)
      logprobs = random_logprobs({4, 16})
      grads = grad_fn.(logprobs)
      assert Nx.shape(grads) == {4, 16}
      assert_finite(grads)
    end
  end

  describe "numerical stability" do
    test "KL handles near-zero probabilities" do
      # Very peaked distribution
      p_logprobs = Nx.tensor([-0.01, -10.0, -10.0, -10.0]) |> normalize_logprobs()
      q_logprobs = random_logprobs({4})
      result = Divergences.kl_divergence(p_logprobs, q_logprobs)
      assert_finite(result)
    end

    test "entropy handles near-zero probabilities" do
      # Very peaked distribution
      logprobs = Nx.tensor([-0.01, -20.0, -20.0, -20.0]) |> normalize_logprobs()
      result = Divergences.entropy(logprobs)
      assert_finite(result)
    end
  end

  # ADR-010: KL direction and symmetric options
  describe "kl_divergence/3 with direction option" do
    test "direction: :forward is default behavior (KL(P||Q))" do
      p_logprobs = Nx.tensor([-0.1, -1.5, -3.0, -4.0]) |> normalize_logprobs()
      q_logprobs = Nx.tensor([-2.0, -0.5, -2.0, -3.0]) |> normalize_logprobs()

      default_kl = Divergences.kl_divergence(p_logprobs, q_logprobs)
      forward_kl = Divergences.kl_divergence(p_logprobs, q_logprobs, direction: :forward)

      assert_close(default_kl, forward_kl)
    end

    test "direction: :reverse computes KL(Q||P)" do
      p_logprobs = Nx.tensor([-0.1, -1.5, -3.0, -4.0]) |> normalize_logprobs()
      q_logprobs = Nx.tensor([-2.0, -0.5, -2.0, -3.0]) |> normalize_logprobs()

      # Reverse KL should equal swapping arguments
      reverse_kl = Divergences.kl_divergence(p_logprobs, q_logprobs, direction: :reverse)
      swapped_kl = Divergences.kl_divergence(q_logprobs, p_logprobs, direction: :forward)

      assert_close(reverse_kl, swapped_kl)
    end

    test "symmetric: true computes 0.5 * (KL(P||Q) + KL(Q||P))" do
      p_logprobs = Nx.tensor([-0.1, -1.5, -3.0, -4.0]) |> normalize_logprobs()
      q_logprobs = Nx.tensor([-2.0, -0.5, -2.0, -3.0]) |> normalize_logprobs()

      symmetric_kl = Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true)

      kl_pq = Divergences.kl_divergence(p_logprobs, q_logprobs, direction: :forward)
      kl_qp = Divergences.kl_divergence(q_logprobs, p_logprobs, direction: :forward)
      expected = Nx.divide(Nx.add(kl_pq, kl_qp), 2.0)

      assert_close(symmetric_kl, expected, atol: 1.0e-5)
    end

    test "symmetric KL is symmetric" do
      p_logprobs = random_logprobs({8})
      q_logprobs = random_logprobs({8})

      sym_pq =
        Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true) |> Nx.to_number()

      sym_qp =
        Divergences.kl_divergence(q_logprobs, p_logprobs, symmetric: true) |> Nx.to_number()

      assert_in_delta sym_pq, sym_qp, 1.0e-5
    end

    test "symmetric: true ignores direction option" do
      p_logprobs = random_logprobs({4})
      q_logprobs = random_logprobs({4})

      sym_default = Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true)

      sym_forward =
        Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true, direction: :forward)

      sym_reverse =
        Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true, direction: :reverse)

      assert_close(sym_default, sym_forward)
      assert_close(sym_default, sym_reverse)
    end

    test "direction: :reverse with reduction: :none preserves shape" do
      p_logprobs = random_logprobs({2, 4})
      q_logprobs = random_logprobs({2, 4})

      result =
        Divergences.kl_divergence(p_logprobs, q_logprobs, direction: :reverse, reduction: :none)

      assert Nx.shape(result) == {2}
    end

    test "symmetric: true with reduction: :none preserves shape" do
      p_logprobs = random_logprobs({2, 4})
      q_logprobs = random_logprobs({2, 4})

      result =
        Divergences.kl_divergence(p_logprobs, q_logprobs, symmetric: true, reduction: :none)

      assert Nx.shape(result) == {2}
    end

    test "gradient flows with direction: :reverse" do
      grad_fn =
        Nx.Defn.grad(fn {p, q} ->
          Divergences.kl_divergence(p, q, direction: :reverse)
        end)

      p = random_logprobs({4})
      q = random_logprobs({4})
      {grad_p, grad_q} = grad_fn.({p, q})
      assert_finite(grad_p)
      assert_finite(grad_q)
    end

    test "gradient flows with symmetric: true" do
      grad_fn =
        Nx.Defn.grad(fn {p, q} ->
          Divergences.kl_divergence(p, q, symmetric: true)
        end)

      p = random_logprobs({4})
      q = random_logprobs({4})
      {grad_p, grad_q} = grad_fn.({p, q})
      assert_finite(grad_p)
      assert_finite(grad_q)
    end
  end

  # ADR-011: Entropy temperature option
  describe "entropy/2 with temperature option" do
    test "temperature: 1.0 is default behavior" do
      logprobs = random_logprobs({4})

      default = Divergences.entropy(logprobs)
      with_temp = Divergences.entropy(logprobs, temperature: 1.0)

      assert_close(default, with_temp)
    end

    test "temperature < 1.0 decreases entropy (sharpens distribution)" do
      # Use a non-uniform distribution
      logprobs = Nx.tensor([-0.5, -1.0, -2.0, -3.0]) |> normalize_logprobs()

      entropy_normal = Divergences.entropy(logprobs, mode: :bonus) |> Nx.to_number()

      entropy_sharp =
        Divergences.entropy(logprobs, temperature: 0.5, mode: :bonus) |> Nx.to_number()

      # Lower temperature should give lower entropy (more peaked)
      assert entropy_sharp < entropy_normal
    end

    test "temperature > 1.0 increases entropy (flattens distribution)" do
      # Use a peaked distribution
      logprobs = Nx.tensor([0.0, -3.0, -5.0, -8.0]) |> normalize_logprobs()

      entropy_normal = Divergences.entropy(logprobs, mode: :bonus) |> Nx.to_number()

      entropy_flat =
        Divergences.entropy(logprobs, temperature: 2.0, mode: :bonus) |> Nx.to_number()

      # Higher temperature should give higher entropy (more uniform)
      assert entropy_flat > entropy_normal
    end

    test "temperature works with mode: :penalty" do
      logprobs = random_logprobs({4})

      bonus = Divergences.entropy(logprobs, temperature: 0.5, mode: :bonus) |> Nx.to_number()
      penalty = Divergences.entropy(logprobs, temperature: 0.5, mode: :penalty) |> Nx.to_number()

      assert_in_delta penalty, -bonus, 1.0e-5
    end

    test "temperature works with normalize: true" do
      logprobs = random_logprobs({4})

      result = Divergences.entropy(logprobs, temperature: 0.7, normalize: true)
      value = Nx.to_number(result)

      # Normalized entropy should still be in [0, 1]
      assert value >= 0.0
      assert value <= 1.0 + 1.0e-5
    end

    test "temperature works with all reduction modes" do
      logprobs = random_logprobs({2, 8})

      mean_result = Divergences.entropy(logprobs, temperature: 0.8, reduction: :mean)
      assert Nx.shape(mean_result) == {}

      sum_result = Divergences.entropy(logprobs, temperature: 0.8, reduction: :sum)
      assert Nx.shape(sum_result) == {}

      none_result = Divergences.entropy(logprobs, temperature: 0.8, reduction: :none)
      assert Nx.shape(none_result) == {2}
    end

    test "gradient flows with temperature option" do
      grad_fn = Nx.Defn.grad(fn x -> Divergences.entropy(x, temperature: 0.5) end)
      logprobs = random_logprobs({4})
      grads = grad_fn.(logprobs)
      assert_finite(grads)
    end

    test "temperature combined with normalize and penalty mode" do
      logprobs = random_logprobs({2, 8})

      result =
        Divergences.entropy(logprobs,
          temperature: 0.7,
          normalize: true,
          mode: :penalty,
          reduction: :mean
        )

      value = Nx.to_number(result)
      # Penalty mode + normalized => should be in [-1, 0]
      assert value >= -1.0 - 1.0e-5
      assert value <= 0.0 + 1.0e-5
    end
  end

  # Helper to normalize log probabilities
  defp normalize_logprobs(logprobs) do
    Nx.subtract(logprobs, Nx.logsumexp(logprobs, axes: [-1], keep_axes: true))
  end
end
