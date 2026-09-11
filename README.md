# BTL Train

Training backend integration for BTL's model work. Prime-RL is the preferred RL foundation. **BTL Adapt** is the supervised fine-tuning engine and uses Unsloth Core as its current CUDA backend.

The [BTL Advance local RL path](ADVANCE.md) now provides a no-spend MLX contextual-bandit probe on a small model. Prime-RL remains the preferred distributed agentic-RL backend.

Evaluation is supplied by the separate BTL Measure package, which can compare its per-item outputs with BTL Adapt or BTL Advance artifacts.

The [BTL Adapt engine](FINETUNE.md) implements accepted tokenized-data checks, an isolated Unsloth Core LoRA/QLoRA worker, checkpoint integrity/resume checks, adapter export, held-out loss reporting and bounded child-process execution. Its Qwen3.5-0.8B LoRA profile passed a bounded A100 execution and recovery qualification; production support, speed comparisons and capability gains remain unqualified.

The current package owns the pinned Prime-RL checkout inspector previously inside BTL Lab. It records revision, tracked changes, submodule state, config hash and parsed configuration, and prepares the native configuration-check command. It does not install dependencies or launch GPU training.

BTL Lab calls `btl_train.prime_rl.inspect_prime_rl` and stores the plan in its ledger. Both packages share the workspace launcher, while this directory has its own package and Git boundary. The integration tests currently live in the sibling Lab repository.

The next gate is an exact model/task/hardware recipe, bounded execution, checkpoint recovery and held-out evaluation. Model weights, credentials, provider configuration and run receipts belong outside this repository. Do not interpret a successful source inspection as a trained Tinfield model.

`btl_train.operations` now provides separate contracts for pretraining, continued pretraining, SFT, preference optimization, distillation, RL, inference, evaluation and export. `validate_spec` returns an immutable operation plan; `preflight` checks local input hashes under a byte budget. It does not load models or verify their compatibility. Lab links these plans and preflight runs to research experiments. Those generic records are not executable; the specialized `adapt` profile implements supervised execution and adapter serialization separately. `finetune` remains a compatibility alias.

BTL-owned code in this repository is released under the MIT License. Unsloth Core, Prime-RL, model checkpoints and other upstream components retain their own licenses.
