# BTL Advance

BTL Advance is the reinforcement-learning engine. Its job is to improve a policy through evaluated experience. Prime-RL remains the preferred distributed backend for agentic RL. The local MLX path exists so the learning loop can be tested on the Mac without paid compute.

## Local probe

The local profile is intentionally narrow: a one-step contextual bandit with two actions, a deterministic verifier, group-relative advantages and LoRA updates to the last two layers of Qwen3.5-0.8B. It tests the mechanics of policy sampling, rewards, credit assignment, optimization and recovery. It is not a multi-turn agent environment and does not establish a Tinfield capability result.

Use the cached model or another local MLX-compatible model directory:

```sh
./btl advance local \
  --model /absolute/path/to/Qwen3.5-0.8B \
  --steps 6 \
  --group-size 8
```

The default interpreter is `.mlxvenv/bin/python`; set `BTL_MLX_PYTHON` or pass `--python` to use another MLX environment. Set `--max-seconds` to bound the complete local process. `--out` must remain inside the workspace. An existing checkpoint can be continued with `--resume`; the worker checks the frozen protocol, adapter, optimizer and RNG state before continuing.

The root command creates a BTL Lab run, stores the resolved protocol, launches a bounded child process, attaches output hashes and records the result. It never downloads a model, provisions a GPU, calls a provider or spends money. The model path may be outside the workspace because local HF/MLX caches are not copied into BTL.

## What the worker verifies

- Action tokens are single, distinct tokenizer tokens.
- Train and evaluation tasks have disjoint source IDs.
- Each group receives binary verifier rewards and group-centered advantages.
- Only LoRA parameters are trainable.
- Losses and gradients are finite, with disclosed norm clipping.
- An optimizer update changes the adapter.
- Each step saves adapter, optimizer and random-state data with a protocol hash.
- A resumed split run can reach the same final adapter as an uninterrupted run when all conditions are fixed.
- Evaluation is reported before and after training, with the result labelled descriptive-only.

The local engine uses the model's two-action log-probability distribution with temperature 2.0. This is a policy-gradient contextual bandit, not GRPO over generated text. The reward verifier is part of the synthetic test environment. It is useful for catching loop and checkpoint errors before building an agentic path.

## Qualification boundary

The local probe earns a mechanics receipt. It does not qualify Prime-RL, Unsloth, CUDA kernels, long-horizon tool use, large-MoE training, speed, or model capability. The next BTL Advance gate is a source-disjoint multi-turn environment with a real verifier, then a Prime-RL recipe on one declared model and GPU topology. Match total environment, inference, verification and optimization cost when comparing algorithms.

Prime-RL's asynchronous execution, stale-rollout controls and distributed trainer are the preferred scale path. The local backend and the Prime-RL backend share the BTL protocol and artifact lifecycle; they do not share optimizer state or imply cross-backend resume.

## Local qualification completed

The first local run used Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` on an Apple M2 with 16 GB memory. Six groups of eight actions produced nonzero group-relative advantages on every step. The adapter changed, gradients stayed finite, and the worker wrote a checkpoint after each update. A three-step run followed by a fresh process resumed from checkpoint 3 and reached the same final adapter hash as the uninterrupted six-step run.

The uninterrupted run moved the deterministic parity-bandit evaluation from 16/24 to 17/24. The split continuation finished at 17/24. This is a descriptive result on a synthetic two-action task; it is not evidence of a Tinfield capability gain. The full run used about 54 seconds and 1.81 GB peak MLX allocation. One resumed step took 278 seconds, so throughput and stability are not qualified by this probe.

The local worker therefore passes its mechanics gate: reward collection, group credit, policy updates, checkpoint state and cross-process resume. It does not qualify agentic RL, Prime-RL, long-horizon environments, GPU scale, or model capability. The raw evidence is under `private/workspace/lab-state/advance/20260909T233619-89ded8bbf5/`, with split recovery under `split-a2/` and `split-b2/`. Failed launcher attempts remain in the Lab ledger as failure receipts.

The BTL Lab run for the uninterrupted probe is `20260909T233619-89ded8bbf5`; the split runs are `20260909T233955-99c1ec6048` and `20260909T234038-35d93dac62`. Lab artifact verification passes for the uninterrupted run's output, report and qualification summary.
