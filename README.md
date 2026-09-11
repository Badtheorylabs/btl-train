# BTL Train

Training backend integration for BTL's model work. **BTL Adapt** is the supervised fine-tuning engine and uses Unsloth Core as its current CUDA backend. Reinforcement learning lives in the separate [BTL RL repository](https://github.com/Badtheorylabs/btl-rl), whose product name is BTL Advance.

Evaluation is supplied by the separate BTL Measure package, which can compare its per-item outputs with BTL Adapt or BTL Advance artifacts.

The [BTL Adapt engine](FINETUNE.md) implements accepted tokenized-data checks, an isolated Unsloth Core LoRA/QLoRA worker, checkpoint integrity/resume checks, adapter export, held-out loss reporting and bounded child-process execution. Its Qwen3.5-0.8B LoRA profile passed a bounded A100 execution and recovery qualification; production support, speed comparisons and capability gains remain unqualified.

The current package owns supervised execution and the general operation contracts. It does not install dependencies or launch GPU training by itself.

BTL Lab calls the separate BTL RL package for Prime-RL inspection and local Advance execution, then stores the plan and receipts in its ledger. Both packages share the workspace launcher, while this directory has its own package and Git boundary. The integration tests currently live in the sibling Lab repository.

The next gate is an exact model/task/hardware recipe, bounded execution, checkpoint recovery and held-out evaluation. Model weights, credentials, provider configuration and run receipts belong outside this repository. Do not interpret a successful source inspection as a trained Tinfield model.

`btl_train.operations` provides separate contracts for pretraining, continued pretraining, SFT, preference optimization, distillation, RL, inference, evaluation and export. `validate_spec` returns an immutable operation plan; `preflight` checks local input hashes under a byte budget. It does not load models or verify their compatibility. Lab links these plans and preflight runs to research experiments. Those generic records are not executable; the specialized `adapt` profile implements supervised execution and adapter serialization separately. `finetune` remains a compatibility alias.

BTL-owned code in this repository is released under the MIT License. Unsloth Core, Prime-RL, model checkpoints and other upstream components retain their own licenses.
