defmodule NxPenalties.Pipeline.Multi do
  @moduledoc """
  Multi-input penalty pipeline composition (ADR-012).

  Unlike the standard `Pipeline` which operates on a single tensor,
  `Pipeline.Multi` allows penalties that take multiple named inputs.
  This is useful for:

  - **Divergence penalties**: KL divergence between two distributions
  - **Consistency penalties**: Comparing clean vs augmented outputs
  - **Contrastive losses**: Comparing positive and negative pairs

  ## Example

      # Create multi-input pipeline
      pipeline =
        Pipeline.Multi.new()
        |> Pipeline.Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:student_logprobs, :teacher_logprobs],
          weight: 0.1
        )
        |> Pipeline.Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean_output, :noisy_output],
          weight: 0.2,
          opts: [metric: :mse]
        )

      # Compute with named inputs
      {total, metrics} = Pipeline.Multi.compute(pipeline, %{
        student_logprobs: student_out,
        teacher_logprobs: teacher_out,
        clean_output: clean,
        noisy_output: noisy
      })

  ## Entry Format

  Each entry is a 6-tuple: `{name, penalty_fn, inputs, weight, opts, enabled}`

    * `name` - Unique atom identifier for the penalty
    * `penalty_fn` - Function that takes multiple tensors and options
    * `inputs` - List of input names (atoms) to fetch from the inputs map
    * `weight` - Scaling factor (number or tensor)
    * `opts` - Options passed to the penalty function
    * `enabled` - Boolean to include/exclude from computation
  """

  defstruct entries: [], name: nil, meta: %{}

  @type entry ::
          {atom(), function(), [atom()], number() | Nx.Tensor.t(), keyword(), boolean()}
  @type t :: %__MODULE__{
          entries: [entry()],
          name: String.t() | nil,
          meta: %{atom() => map()}
        }

  @doc """
  Create a new empty multi-input pipeline.

  ## Options

    * `:name` - Optional name for the pipeline (for logging/debugging)

  ## Examples

      pipeline = Pipeline.Multi.new()
      pipeline = Pipeline.Multi.new(name: "data-aware")
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    name = Keyword.get(opts, :name, nil)
    %__MODULE__{entries: [], name: name, meta: %{}}
  end

  @doc """
  Add a multi-input penalty to the pipeline.

  ## Parameters

    * `pipeline` - The pipeline struct
    * `name` - Unique atom identifier for this penalty
    * `penalty_fn` - Function with signature `(tensor1, tensor2, ..., opts) -> scalar_tensor`
    * `opts` - Options:
      * `:inputs` - **Required**. List of input names (atoms) to fetch from inputs map
      * `:weight` - Scaling factor. Default: `1.0`
      * `:opts` - Options passed to the penalty function. Default: `[]`
      * `:enabled` - Whether to include in computation. Default: `true`
      * `:differentiable` - Whether gradient tracking should be attempted. Default: `true`

  ## Examples

      pipeline
      |> Pipeline.Multi.add(:kl, &Divergences.kl_divergence/3,
          inputs: [:p_logprobs, :q_logprobs],
          weight: 0.1
        )
      |> Pipeline.Multi.add(:consistency, &Constraints.consistency/3,
          inputs: [:clean, :noisy],
          weight: 0.2,
          opts: [metric: :mse],
          differentiable: true
        )
  """
  @spec add(t(), atom(), function(), keyword()) :: t()
  def add(%__MODULE__{entries: entries, meta: meta} = pipeline, name, penalty_fn, opts) do
    inputs = Keyword.fetch!(opts, :inputs)
    weight = Keyword.get(opts, :weight, 1.0)
    penalty_opts = Keyword.get(opts, :opts, [])
    enabled = Keyword.get(opts, :enabled, true)
    differentiable = Keyword.get(opts, :differentiable, true)

    entry = {name, penalty_fn, inputs, weight, penalty_opts, enabled}
    new_meta = Map.put(meta, name, %{differentiable: differentiable})

    %{pipeline | entries: entries ++ [entry], meta: new_meta}
  end

  @doc """
  Remove a penalty from the pipeline by name.

  ## Examples

      pipeline = Pipeline.Multi.remove(pipeline, :kl)
  """
  @spec remove(t(), atom()) :: t()
  def remove(%__MODULE__{entries: entries, meta: meta} = pipeline, name) do
    new_entries = Enum.reject(entries, fn {n, _, _, _, _, _} -> n == name end)
    new_meta = Map.delete(meta, name)
    %{pipeline | entries: new_entries, meta: new_meta}
  end

  @doc """
  Update the weight of a penalty.

  ## Examples

      pipeline = Pipeline.Multi.update_weight(pipeline, :kl, 0.05)
      pipeline = Pipeline.Multi.update_weight(pipeline, :kl, Nx.tensor(0.05))
  """
  @spec update_weight(t(), atom(), number() | Nx.Tensor.t()) :: t()
  def update_weight(%__MODULE__{entries: entries} = pipeline, name, new_weight) do
    new_entries =
      Enum.map(entries, fn
        {^name, fn_, inputs, _weight, opts, enabled} ->
          {name, fn_, inputs, new_weight, opts, enabled}

        entry ->
          entry
      end)

    %{pipeline | entries: new_entries}
  end

  @doc """
  Enable or disable a penalty.

  ## Examples

      pipeline = Pipeline.Multi.set_enabled(pipeline, :kl, false)
  """
  @spec set_enabled(t(), atom(), boolean()) :: t()
  def set_enabled(%__MODULE__{entries: entries} = pipeline, name, enabled) do
    new_entries =
      Enum.map(entries, fn
        {^name, fn_, inputs, weight, opts, _enabled} ->
          {name, fn_, inputs, weight, opts, enabled}

        entry ->
          entry
      end)

    %{pipeline | entries: new_entries}
  end

  @doc """
  Compute all penalties and return total with metrics.

  ## Parameters

    * `pipeline` - The pipeline struct
    * `inputs_map` - Map of input name => tensor
    * `opts` - Options:
      * `:extra_args` - Additional args merged into each penalty's opts
      * `:track_grad_norms` - Compute gradient norms (default: `false`)

  ## Returns

  A tuple `{total, metrics}` where:
    * `total` - Scalar tensor with sum of weighted penalties
    * `metrics` - Map with per-penalty and aggregate metrics

  ## Examples

      {total, metrics} = Pipeline.Multi.compute(pipeline, %{
        p_logprobs: p_tensor,
        q_logprobs: q_tensor
      })
  """
  @spec compute(t(), map(), keyword()) :: {Nx.Tensor.t(), map()}
  def compute(%__MODULE__{entries: entries} = pipeline, inputs_map, opts \\ []) do
    extra_args = Keyword.get(opts, :extra_args, [])
    track_grad_norms = Keyword.get(opts, :track_grad_norms, false)

    # Filter enabled entries
    enabled_entries = Enum.filter(entries, fn {_, _, _, _, _, enabled} -> enabled end)

    if enabled_entries == [] do
      {Nx.tensor(0.0), %{}}
    else
      # Compute each penalty
      {total, metrics} =
        Enum.reduce(enabled_entries, {Nx.tensor(0.0), %{}}, fn
          {name, penalty_fn, inputs, weight, penalty_opts, _enabled}, {acc_total, acc_metrics} ->
            # Fetch input tensors from the map
            input_tensors =
              Enum.map(inputs, fn input_name -> Map.fetch!(inputs_map, input_name) end)

            # Merge extra_args with penalty opts
            merged_opts = Keyword.merge(penalty_opts, extra_args)

            # Compute raw penalty - call function with inputs + opts
            raw_penalty = apply(penalty_fn, input_tensors ++ [merged_opts])
            raw_value = Nx.to_number(raw_penalty)

            # Apply weight
            weighted_penalty = Nx.multiply(raw_penalty, weight)
            weighted_value = Nx.to_number(weighted_penalty)

            # Update metrics
            name_str = Atom.to_string(name)

            new_metrics =
              acc_metrics
              |> Map.put(name_str, raw_value)
              |> Map.put("#{name_str}_weighted", weighted_value)

            # Accumulate total
            new_total = Nx.add(acc_total, weighted_penalty)

            {new_total, new_metrics}
        end)

      # Add total to metrics
      total_value = Nx.to_number(total)
      metrics = Map.put(metrics, "total", total_value)

      # Optionally add gradient metrics
      metrics =
        if track_grad_norms do
          grad_metrics = compute_grad_norms(pipeline, inputs_map)
          Map.merge(metrics, grad_metrics)
        else
          metrics
        end

      {total, metrics}
    end
  end

  @doc """
  Compute only the total penalty (gradient-compatible).

  Unlike `compute/3`, this function does NOT call `Nx.to_number` and
  is therefore compatible with `Nx.Defn.grad/1`.

  ## Examples

      # Can be used with gradients
      grad_fn = Nx.Defn.grad(fn inputs ->
        Pipeline.Multi.compute_total(pipeline, inputs)
      end)
  """
  @spec compute_total(t(), map(), keyword()) :: Nx.Tensor.t()
  def compute_total(%__MODULE__{entries: entries}, inputs_map, opts \\ []) do
    extra_args = Keyword.get(opts, :extra_args, [])

    # Filter enabled entries
    enabled_entries = Enum.filter(entries, fn {_, _, _, _, _, enabled} -> enabled end)

    if enabled_entries == [] do
      Nx.tensor(0.0)
    else
      Enum.reduce(enabled_entries, Nx.tensor(0.0), fn
        {_name, penalty_fn, inputs, weight, penalty_opts, _enabled}, acc_total ->
          # Fetch input tensors
          input_tensors =
            Enum.map(inputs, fn input_name -> Map.fetch!(inputs_map, input_name) end)

          # Merge opts
          merged_opts = Keyword.merge(penalty_opts, extra_args)

          # Compute raw penalty
          raw_penalty = apply(penalty_fn, input_tensors ++ [merged_opts])

          # Apply weight and accumulate
          weighted_penalty = Nx.multiply(raw_penalty, weight)
          Nx.add(acc_total, weighted_penalty)
      end)
    end
  end

  # Compute gradient norms for each penalty
  defp compute_grad_norms(%__MODULE__{entries: entries, meta: meta}, inputs_map) do
    enabled_entries = Enum.filter(entries, fn {_, _, _, _, _, enabled} -> enabled end)

    Enum.reduce(enabled_entries, %{}, fn
      {name, penalty_fn, inputs, _weight, penalty_opts, _enabled}, acc_metrics ->
        # Skip non-differentiable penalties
        if Map.get(meta, name, %{})[:differentiable] == false do
          acc_metrics
        else
          try do
            # Create a function that computes the penalty from the first input tensor
            # (simplified gradient computation - could be extended)
            first_input = Map.fetch!(inputs_map, hd(inputs))

            grad_fn =
              Nx.Defn.grad(fn tensor ->
                all_inputs =
                  inputs
                  |> Enum.with_index()
                  |> Enum.map(fn {input_name, idx} ->
                    if idx == 0, do: tensor, else: Map.fetch!(inputs_map, input_name)
                  end)

                apply(penalty_fn, all_inputs ++ [penalty_opts])
              end)

            grads = grad_fn.(first_input)

            grad_norm =
              grads |> Nx.flatten() |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()

            name_str = Atom.to_string(name)
            Map.put(acc_metrics, "#{name_str}_grad_norm", grad_norm)
          rescue
            _ -> acc_metrics
          end
        end
    end)
  end
end
