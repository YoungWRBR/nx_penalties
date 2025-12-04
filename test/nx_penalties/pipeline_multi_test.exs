defmodule NxPenalties.Pipeline.MultiTest do
  @moduledoc """
  Tests for Pipeline.Multi - multi-input pipeline composition (ADR-012).
  """
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.{Constraints, Divergences}
  alias NxPenalties.Pipeline.Multi

  describe "new/1" do
    test "creates empty multi-input pipeline" do
      pipeline = Multi.new()
      assert pipeline.entries == []
      assert pipeline.name == nil
    end

    test "accepts name option" do
      pipeline = Multi.new(name: "data-aware")
      assert pipeline.name == "data-aware"
    end
  end

  describe "add/4" do
    test "adds multi-input penalty with named inputs" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p_logprobs, :q_logprobs],
          weight: 0.1
        )

      assert length(pipeline.entries) == 1
      {name, _fn, inputs, weight, _opts, enabled} = hd(pipeline.entries)
      assert name == :kl
      assert inputs == [:p_logprobs, :q_logprobs]
      assert weight == 0.1
      assert enabled == true
    end

    test "adds consistency penalty with paired inputs" do
      pipeline =
        Multi.new()
        |> Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean_out, :noisy_out],
          weight: 0.2,
          opts: [metric: :mse]
        )

      assert length(pipeline.entries) == 1
      {name, _fn, inputs, _weight, opts, _enabled} = hd(pipeline.entries)
      assert name == :consistency
      assert inputs == [:clean_out, :noisy_out]
      assert opts[:metric] == :mse
    end

    test "supports enabled: false option" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1,
          enabled: false
        )

      {_name, _fn, _inputs, _weight, _opts, enabled} = hd(pipeline.entries)
      assert enabled == false
    end

    test "supports differentiable: false option" do
      pipeline =
        Multi.new()
        |> Multi.add(:custom, fn _a, _b, _opts -> Nx.tensor(1.0) end,
          inputs: [:a, :b],
          weight: 0.1,
          differentiable: false
        )

      assert pipeline.meta[:custom].differentiable == false
    end

    test "differentiable defaults to true" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )

      assert pipeline.meta[:kl].differentiable == true
    end
  end

  describe "compute/3" do
    test "computes penalty with named inputs from map" do
      p_logprobs = random_logprobs({4})
      q_logprobs = random_logprobs({4})

      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p_logprobs, :q_logprobs],
          weight: 0.1
        )

      {total, metrics} =
        Multi.compute(pipeline, %{
          p_logprobs: p_logprobs,
          q_logprobs: q_logprobs
        })

      assert_scalar(total)
      assert Map.has_key?(metrics, "kl")
      assert Map.has_key?(metrics, "kl_weighted")
      assert Map.has_key?(metrics, "total")
    end

    test "computes consistency penalty with paired outputs" do
      clean = Nx.tensor([1.0, 2.0, 3.0])
      noisy = Nx.tensor([1.1, 2.1, 3.1])

      pipeline =
        Multi.new()
        |> Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean, :noisy],
          weight: 1.0,
          opts: [metric: :mse]
        )

      {total, metrics} = Multi.compute(pipeline, %{clean: clean, noisy: noisy})

      assert_scalar(total)
      # MSE of [0.1, 0.1, 0.1] = 0.01
      assert_in_delta metrics["consistency"], 0.01, 1.0e-4
    end

    test "combines multiple penalties" do
      p = random_logprobs({4})
      q = random_logprobs({4})
      clean = Nx.tensor([1.0, 2.0, 3.0])
      noisy = Nx.tensor([1.1, 2.1, 3.1])

      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )
        |> Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean, :noisy],
          weight: 1.0
        )

      {total, metrics} =
        Multi.compute(pipeline, %{
          p: p,
          q: q,
          clean: clean,
          noisy: noisy
        })

      expected_total = metrics["kl_weighted"] + metrics["consistency_weighted"]
      assert_in_delta Nx.to_number(total), expected_total, 1.0e-5
    end

    test "skips disabled penalties" do
      p = random_logprobs({4})
      q = random_logprobs({4})

      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1,
          enabled: false
        )

      {total, metrics} = Multi.compute(pipeline, %{p: p, q: q})

      assert_close(total, Nx.tensor(0.0))
      refute Map.has_key?(metrics, "kl")
    end

    test "raises on missing input" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p_logprobs, :q_logprobs],
          weight: 0.1
        )

      assert_raise KeyError, fn ->
        Multi.compute(pipeline, %{p_logprobs: random_logprobs({4})})
      end
    end

    test "returns zero for empty pipeline" do
      {total, metrics} = Multi.compute(Multi.new(), %{})

      assert_close(total, Nx.tensor(0.0))
      assert metrics == %{} or metrics["total"] == 0.0
    end
  end

  describe "update_weight/3" do
    test "updates weight for existing penalty" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )
        |> Multi.update_weight(:kl, 0.5)

      {_name, _fn, _inputs, weight, _opts, _enabled} = hd(pipeline.entries)
      assert weight == 0.5
    end

    test "accepts tensor weight" do
      weight_tensor = Nx.tensor(0.02)

      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )
        |> Multi.update_weight(:kl, weight_tensor)

      {_name, _fn, _inputs, weight, _opts, _enabled} = hd(pipeline.entries)
      assert weight == weight_tensor
    end
  end

  describe "set_enabled/3" do
    test "disables penalty" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )
        |> Multi.set_enabled(:kl, false)

      {_name, _fn, _inputs, _weight, _opts, enabled} = hd(pipeline.entries)
      assert enabled == false
    end

    test "re-enables penalty" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )
        |> Multi.set_enabled(:kl, false)
        |> Multi.set_enabled(:kl, true)

      {_name, _fn, _inputs, _weight, _opts, enabled} = hd(pipeline.entries)
      assert enabled == true
    end
  end

  describe "remove/2" do
    test "removes penalty by name" do
      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3, inputs: [:p, :q], weight: 0.1)
        |> Multi.add(:consistency, &Constraints.consistency/3, inputs: [:a, :b], weight: 0.2)
        |> Multi.remove(:kl)

      assert length(pipeline.entries) == 1
      {name, _fn, _inputs, _weight, _opts, _enabled} = hd(pipeline.entries)
      assert name == :consistency
    end
  end

  describe "gradient flow" do
    test "gradient flows through multi-input pipeline" do
      pipeline =
        Multi.new()
        |> Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean, :noisy],
          weight: 1.0,
          opts: [metric: :mse]
        )

      clean = Nx.tensor([1.0, 2.0, 3.0])
      noisy = Nx.tensor([1.1, 2.1, 3.1])

      grad_fn =
        Nx.Defn.grad(fn tensors ->
          Multi.compute_total(pipeline, tensors)
        end)

      grads = grad_fn.(%{clean: clean, noisy: noisy})

      assert is_map(grads)
      assert Map.has_key?(grads, :clean)
      assert Map.has_key?(grads, :noisy)
      assert_finite(grads.clean)
      assert_finite(grads.noisy)
    end
  end

  describe "compute_total/3" do
    test "returns only total without metrics" do
      p = random_logprobs({4})
      q = random_logprobs({4})

      pipeline =
        Multi.new()
        |> Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p, :q],
          weight: 0.1
        )

      total = Multi.compute_total(pipeline, %{p: p, q: q})

      assert_scalar(total)
    end
  end

  describe "gradient tracking" do
    test "tracks gradient norms when enabled" do
      clean = Nx.tensor([1.0, 2.0, 3.0])
      noisy = Nx.tensor([1.1, 2.1, 3.1])

      pipeline =
        Multi.new()
        |> Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean, :noisy],
          weight: 1.0
        )

      {_total, metrics} =
        Multi.compute(pipeline, %{clean: clean, noisy: noisy}, track_grad_norms: true)

      assert Map.has_key?(metrics, "consistency_grad_norm") or
               Map.has_key?(metrics, "total_grad_norm")
    end

    test "skips non-differentiable penalties for gradient tracking" do
      clean = Nx.tensor([1.0, 2.0, 3.0])
      noisy = Nx.tensor([1.1, 2.1, 3.1])

      pipeline =
        Multi.new()
        |> Multi.add(:custom, fn a, _b, _opts -> Nx.sum(Nx.abs(a)) end,
          inputs: [:clean, :noisy],
          weight: 1.0,
          differentiable: false
        )

      # Should not raise when computing
      {total, _metrics} = Multi.compute(pipeline, %{clean: clean, noisy: noisy})
      assert_scalar(total)
    end
  end
end
