# BTL Adapt

The engine has one job: apply supervised LoRA/QLoRA updates to a local model using accepted tokenized examples, and export a traceable adapter. It uses Unsloth Core's model preparation and accelerated training paths. It does not own task generation, RL, teacher selection, inference serving or model release decisions.

The execution code exists. Its first experimental profile targets a single Linux NVIDIA GPU and dense Qwen3. CUDA execution, numerical parity, fresh-process adapter reload and performance qualification have not yet been run. The tests establish local data/control correctness and child-process deadline handling only.

## Commands

Use the root workspace launcher:

```text
./btl adapt check RECIPE.json --data-only
./btl adapt check RECIPE.json --data-only --weights
./btl adapt check RECIPE.json --python /path/to/pinned/environment/bin/python
./btl adapt run RECIPE.json --python /path/to/pinned/environment/bin/python --execute --authorization-ref APPROVED_RESOURCE_AND_BUDGET_REFERENCE
```

Replace the uppercase arguments and interpreter path with real values. `--data-only` works without Torch or Unsloth installed. `--weights` additionally hashes all declared model assets. Runtime checking uses the selected interpreter's installed package metadata and does not install anything. The worker checks CUDA and the actual model when it starts.

The run command writes a frozen request, runs the environment check, and starts the worker as an isolated child process on existing compute. It records the result in Lab. `--experiment ID` links the operation to an open research project with the same project ID. This is a background operation link; the model run is not automatically admitted as an experimental-arm observation.

`wall_seconds` is the worker deadline, including data/model checks and loading. On expiry, the launcher sends SIGTERM, allows up to 15 seconds to finish stopping, then kills its process group. The worker requests a checkpoint at the next complete optimizer step when it can handle SIGTERM. A timeout before that point cannot guarantee a fresh checkpoint; use the last complete one. Cloud resources are not stopped or provisioned. External spending is not measured or capped by this launcher, and the approval-reference string is a record, not proof or enforcement of authorization. The calling operator must have the resource and budget approved before execution.

Model/data downloads are disabled through offline settings and local model loading. No teacher or model API calls are used by this SFT path. External experiment reporting is disabled. GPU dependencies stay in the selected environment rather than being installed on the Mac.

## Recipe contract

The version-1 JSON recipe requires these fields:

| Fields | Meaning |
|---|---|
| `schema_version`, `project_id` | Version 1 and a registered BTL project |
| `model_directory` | Existing local model directory, or a local Hugging Face cache snapshot |
| `model_manifest`, `model_manifest_sha256` | Workspace-local manifest path and its hash |
| `dataset_manifest`, `dataset_manifest_sha256` | Workspace-local accepted dataset manifest and its hash |
| `method`, `precision`, `expected_model_type` | `lora` or `qlora`; `bf16` or `fp16`; currently `qwen3` |
| `max_seq_length`, `max_steps` | Hard example-length limit and final optimizer-step target |
| `micro_batch_size`, `gradient_accumulation_steps` | Explicit batching; no automatic batch adjustment |
| `rank`, `alpha`, `target_modules` | LoRA configuration; target names must exist in the actual model |
| `learning_rate`, `max_grad_norm`, `seed` | Explicit optimization settings |
| `save_steps`, `wall_seconds` | Checkpoint interval and worker deadline |
| `versions` | Exact installed versions of `unsloth`, `unsloth_zoo`, `torch`, `transformers`, `trl`, `datasets`, `peft` |

The initial implementation fixes dropout to zero, bias training off, AdamW with constant learning rate, zero warmup and weight decay, one evaluation example per batch, no packing, no padding-free mode, and no data-loader subprocesses. It retains the latest two Trainer checkpoints. These choices are visible in the saved resolved runtime configuration. Changes to this behavior need a new validated execution profile.

The model manifest lists `files`, each with `path` and `sha256`, plus a `tokenizer` identity object. Include weights, configuration and tokenizer assets. Put the manifest outside the model directory. The full execution audit hashes the assets before loading them. Hugging Face snapshot symlinks may resolve within their cache repository, so weights need not be duplicated into BTL storage.

The dataset manifest has `schema_version: 1`, `accepted: true`, the same `tokenizer` object, an `acceptance_receipt` with local `path` and `sha256`, and `train` / `evaluation` entries with local `path` and `sha256`. Paths in this manifest are relative to its directory. Acceptance is supplied by the data system; this engine verifies its receipt's integrity, not the truth of every acceptance claim.

Each JSONL row contains `id`, `source_id`, `input_ids` and `labels`. A label is either its corresponding token ID or `-100`. Position zero must be masked, and there must be supervised tokens after it. The caller supplies the intended assistant/completion mask. Padding is masked with `-100`; examples are never silently truncated or repacked. Train/evaluation overlap is rejected by row ID, source group and identical token sequence. Semantic leakage still requires upstream data review.

The in-memory profile rejects split files over 512 MiB by default. `max_dataset_bytes` can change that limit explicitly. Larger-scale streaming preparation is future work.

## Checkpoints and evidence

The worker checks that only LoRA tensors train, observes actual optimizer steps, compares adapter weights before and after, measures held-out loss, saves the adapter/tokenizer and verifies serialized adapter weights reload into the existing model instance. It fails on unchanged or nonfinite adapter weights. This is not fresh-process reload verification or a behavioral capability score.

Every complete checkpoint receives a `btl-checkpoint.json` with file hashes and a contract hash covering the recipe and worker source. To resume, use the same recipe plus `--resume` and a workspace-relative checkpoint directory. Model, data, optimizer settings and source changes invalidate the resume contract. `max_steps` is the original final target, not the number of extra steps. A resumed attempt gets a new Lab run and output directory; it does not overwrite the previous attempt.

The output directory contains the request, worker log, per-step timing/token counts, checkpoints, adapter, adapter manifest, resolved Trainer/model configuration and final receipts. Timing counters themselves add overhead. Startup, checkpointing and evaluation must be included when comparing whole jobs. No speedup or capability gain is inferred from successful execution. `release_ready` remains false pending independent behavioral evaluation.

## Next qualification gate

On one explicitly authorized GPU, pin a compatible environment and small dense-Qwen3 checkpoint. Run the same accepted dataset through the reference and accelerated paths, check loss/gradient behavior, complete several timed updates, interrupt/resume, reload the export in a fresh process, and run a behavioral regression suite. Only then publish a supported recipe or performance claim.

The implementation follows the model-preparation and SFT interfaces documented by [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) and the existing BTL training scripts. Notebook implementation code was not copied. Unsloth Core and optional UI components have different licensing scopes; this engine targets Core.

## Local 0.8B validation completed

A separate MLX qualification ran Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` on an Apple M2 with 16 GB memory, using MLX 0.32.0 and MLX-LM 0.31.3. Three fresh-process eight-step runs, a checkpoint resume, a SIGTERM-after-checkpoint recovery and a fresh-process reload produced the same final adapter. The loss and gradient calculations matched an equivalent log-softmax reference on the tested batch. Peak MLX allocation was 2.374 GB.

The task was synthetic field extraction: 24 training rows and 12 fixed evaluation rows. Held-out loss changed from 0.07725811 to 0.07265254; exact-match stayed 12/12. This does not demonstrate a task-solving capability gain. Post-warmup median step time was 0.751 seconds, approximately 110.5 actual training tokens/second on this small profile; there is no baseline speedup comparison.

The reproducible qualification entry point is `python -m btl_train.mlx_qualification`, with `prepare`, `train` and `reload` phases. It uses the shared row validator, batch padding and loss-mask conventions, and its phase driver uses the shared bounded process runner. The local worker's training loop is MLX-specific and does not execute Unsloth code. Fresh processes reused one installed environment; clean-install reproducibility is not yet established.

Local MLX execution gates passed earlier. The actual Unsloth Core CUDA path has now also passed a bounded Qwen3.5-0.8B A100 execution and recovery profile: 8 optimizer steps, adapter serialization, checkpoint resume from step 4, fresh-process reload and fixed-task behavior retention. The production release recipe, a matched non-Unsloth speed baseline and a nontrivial capability evaluation remain unqualified. The private run report and complete evidence are under `private/workspace/lab-state/btl-adapt-a100-08b-20260909/` in the BTL workspace; Lab run ID is recorded in its `lab-run-id.txt`.
