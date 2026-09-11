"""Experimental single-GPU Unsloth SFT worker; imported only in its own environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import time
from pathlib import Path

from .finetune_data import PACKAGES, audit, digest, fingerprint, pad_batch, validate_config


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def doctor(config: dict) -> dict:
    validate_config(config)
    blockers, versions = [], {}
    if platform.system() != "Linux":
        blockers.append("This experimental worker requires Linux and one visible NVIDIA GPU")
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        if versions[package] != config["versions"][package]:
            blockers.append(f"Version mismatch: {package}; expected {config['versions'][package]}, found {versions[package]}")
    return {"ready_for_worker_start": not blockers, "blockers": blockers, "versions": versions,
            "gpu_and_numerics_verified": False}


def verify_resume(path: Path, contract_hash: str) -> int:
    receipt = json.loads((path / "btl-checkpoint.json").read_text())
    if receipt["contract_sha256"] != contract_hash:
        raise ValueError("Resume contract differs; changing training settings requires a new recipe")
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"}
    if not required.issubset(receipt["files"]):
        raise ValueError("Checkpoint lacks resume state")
    for name, expected in receipt["files"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()) or digest(file) != expected:
            raise ValueError(f"Changed checkpoint file: {name}")
    return int(json.loads((path / "trainer_state.json").read_text())["global_step"])


def train(request: dict, output: Path) -> dict:
    config = request["config"]
    runtime = doctor(config)
    if not runtime["ready_for_worker_start"]:
        raise ValueError("; ".join(runtime["blockers"]))
    contract_hash = fingerprint({"config": config, "implementation_sources": request["implementation_sources"]})
    resume = Path(request["resume"]) if request.get("resume") else None
    initial_step = verify_resume(resume, contract_hash) if resume else 0
    if config["max_steps"] <= initial_step:
        raise ValueError("Resume checkpoint already reached the configured final step")
    started = time.monotonic()
    prepared = audit(Path(request["workspace"]), config, check_weights=True)
    # Unsloth must install its patches before importing the training libraries.
    from unsloth import FastLanguageModel, FastModel
    import torch
    from datasets import Dataset
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one NVIDIA GPU for this profile")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support the requested BF16 profile")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    loader = FastModel if config["expected_model_type"] == "qwen3_5" else FastLanguageModel
    model, processor = loader.from_pretrained(
        model_name=prepared["model_directory"], max_seq_length=config["max_seq_length"],
        dtype=dtype, load_in_4bit=config["method"] == "qlora", full_finetuning=False,
        use_exact_model_name=True, local_files_only=True, trust_remote_code=False,
        device_map={"": 0},
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if model.config.model_type != config["expected_model_type"]:
        raise ValueError("Loaded architecture differs from the execution profile")
    if bool(getattr(model, "is_loaded_in_4bit", False)) != (config["method"] == "qlora"):
        raise ValueError("Loaded quantization mode differs from the recipe")
    if any(p.is_meta or p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("This profile requires fully materialized GPU-resident parameters")
    modules = dict(model.named_modules())
    for suffix in config["target_modules"]:
        if not any(name.endswith("." + suffix) or name == suffix for name in modules):
            raise ValueError(f"LoRA target missing: {suffix}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has neither a padding nor an EOS token")
    for rows in prepared["datasets"].values():
        if any(max(row["input_ids"]) >= len(tokenizer) for row in rows):
            raise ValueError("Prepared token IDs exceed tokenizer vocabulary")
    model = loader.get_peft_model(
        model, r=config["rank"], lora_alpha=config["alpha"], lora_dropout=0,
        bias="none", target_modules=config["target_modules"],
        use_gradient_checkpointing="unsloth", random_state=config["seed"],
    )
    loader.for_training(model)
    model.config.use_cache = False
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Unexpected trainable parameters; only LoRA tensors are allowed")

    def state_hash(state):
        import hashlib
        h = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    initial_hash = state_hash(load_file(str(resume / "adapter_model.safetensors"))) if resume else state_hash(get_peft_model_state_dict(model))
    if resume:
        set_peft_model_state_dict(model, load_file(str(resume / "adapter_model.safetensors")))
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    meter = {"tokens": 0, "supervised": 0, "started": 0}
    steps = []

    class Collator:
        def __call__(self, rows):
            return {key: torch.tensor(value, dtype=torch.long)
                    for key, value in pad_batch(rows, tokenizer.pad_token_id).items()}

    class Trainer(SFTTrainer):
        def training_step(self, model, inputs, *args, **kwargs):
            meter["tokens"] += int(inputs["attention_mask"].sum().item())
            meter["supervised"] += int((inputs["labels"][:, 1:] != -100).sum().item())
            return super().training_step(model, inputs, *args, **kwargs)

    class Receipts(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [p.grad for _, p in trainable if p.grad is not None]
            if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("Missing or nonfinite adapter gradients")

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            meter.update(tokens=0, supervised=0, started=time.monotonic())

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            steps.append({"step": state.global_step, "seconds": time.monotonic() - meter["started"],
                          "tokens": meter["tokens"], "supervised_tokens": meter["supervised"]})
            write_json(output / "steps.json", steps)
            if stop["requested"]:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            files = {str(p.relative_to(checkpoint)): digest(p) for p in checkpoint.rglob("*")
                     if p.is_file() and p.name != "btl-checkpoint.json"}
            write_json(checkpoint / "btl-checkpoint.json", {"contract_sha256": contract_hash, "files": files})

    args = SFTConfig(
        output_dir=str(output / "checkpoints"), max_steps=config["max_steps"],
        per_device_train_batch_size=config["micro_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type="constant", warmup_steps=0, weight_decay=0,
        bf16=config["precision"] == "bf16", fp16=config["precision"] == "fp16",
        optim="adamw_torch", seed=config["seed"], data_seed=config["seed"],
        gradient_checkpointing=True, max_length=None, packing=False, padding_free=False,
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        dataloader_num_workers=0, report_to="none", logging_steps=1,
        logging_nan_inf_filter=False, save_steps=config["save_steps"], save_total_limit=2,
    )
    trainer = Trainer(model=model, processing_class=tokenizer, args=args,
                      train_dataset=Dataset.from_list(prepared["datasets"]["train"]),
                      eval_dataset=Dataset.from_list(prepared["datasets"]["evaluation"]),
                      data_collator=Collator(), callbacks=[Receipts()])
    write_json(output / "resolved-runtime.json", {"trainer": trainer.args.to_dict(),
                                                "model": model.config.to_dict(),
                                                "versions": runtime["versions"],
                                                "contract_sha256": contract_hash})
    baseline = trainer.evaluate()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    updated = state_hash(get_peft_model_state_dict(model))
    if any(not bool(torch.isfinite(parameter).all()) for _, parameter in trainable):
        raise ValueError("Nonfinite adapter parameters")
    if trainer.state.global_step <= initial_step or initial_hash == updated:
        raise ValueError("No verified optimizer progress or changed adapter")
    evaluation = trainer.evaluate()
    import math
    if not all(math.isfinite(value["eval_loss"]) for value in (baseline, evaluation)):
        raise ValueError("Nonfinite held-out loss")
    adapter = output / "adapter"
    trainer.save_model(str(adapter)); tokenizer.save_pretrained(adapter)
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    with torch.no_grad():
        for _, parameter in trainable:
            parameter.zero_()
    set_peft_model_state_dict(model, saved)
    if state_hash(get_peft_model_state_dict(model)) != updated:
        raise ValueError("Serialized adapter does not reload identically")
    manifest = {str(p.relative_to(adapter)): digest(p) for p in adapter.rglob("*") if p.is_file()}
    write_json(output / "adapter-manifest.json", {"files": manifest, "contract_sha256": contract_hash,
                                                "tokenizer": prepared["tokenizer"], "model_manifest": config["model_manifest_sha256"]})
    return {"status": "interrupted" if stop["requested"] else "completed", "evidence_class": "optimization",
            "initial_step": initial_step, "final_step": trainer.state.global_step,
            "initial_adapter_sha256": initial_hash, "updated_adapter_sha256": updated,
            "serialization_reload_passed": True, "fresh_process_reload_verified": False,
            "baseline_loss": baseline["eval_loss"], "final_loss": evaluation["eval_loss"],
            "behavioral_evaluation": "not-performed", "release_ready": False,
            "gpu": torch.cuda.get_device_name(0), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.monotonic() - started, "steps": steps,
            "timing_scope": "Per-step training with counters; full wall time includes startup/evaluation/checkpoints",
            "versions": runtime["versions"], "data": prepared["summary"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request.get("implementation_sources", {}).items():
        if Path(name).name != name or digest(Path(__file__).parent / name) != expected:
            raise ValueError("Worker source differs from the frozen launch")
    if args.doctor:
        print(json.dumps(doctor(request["config"])))
        return
    if args.output is None:
        parser.error("--output is required for training")
    args.output.mkdir(parents=True, exist_ok=True)
    result = train(request, args.output)
    write_json(args.output / "worker-result.json", result)


if __name__ == "__main__":
    main()
