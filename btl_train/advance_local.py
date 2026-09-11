"""BTL Advance's local, one-step RL qualification path.

This is a deliberately small contextual-bandit environment. It exercises policy
sampling, verifier rewards, group-relative advantages, policy updates and
checkpoint recovery on a local MLX model. Prime-RL remains the preferred
distributed RL backend for agentic training; this module is the cheap local
development backend and is not a Prime-RL replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MAX_SECONDS = 900
DEFAULT_STEPS = 6
DEFAULT_GROUP_SIZE = 8
TEMPERATURE = 2.0
LORA_LAYERS = 2
LORA_RANK = 4
LORA_SCALE = 8.0
LEARNING_RATE = 1e-5
SEED = 8128
MEMORY_LIMIT = 8 * 1024**3


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(protocol: dict) -> str:
    encoded = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_tasks(split: str, count: int) -> list[dict[str, Any]]:
    """Create deterministic verifier tasks with source-disjoint examples."""
    tasks = []
    for index in range(count):
        number = index * 7 + (3 if split == "evaluation" else 1)
        expected = number % 2
        tasks.append(
            {
                "id": f"advance-{split}-{index:03d}",
                "source_id": f"advance-{split}-source-{index:03d}",
                "number": number,
                "expected_action": expected,
                "prompt": (
                    "Choose exactly A or B. Return only one letter. "
                    "The verifier rewards A for an even number and B for an odd number. "
                    f"Number: {number}. Answer:"
                ),
            }
        )
    return tasks


def action_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for text in (" A", " B"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Action {text!r} is not a single token: {encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError("A and B action tokens collide")
    return ids


def prompt_ids(tokenizer: Any, task: dict[str, Any]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Tokenizer returned an unexpected batch")
        ids = ids[0]
    if not isinstance(ids, list) or not ids:
        raise ValueError("Tokenizer returned an empty prompt")
    return [int(value) for value in ids]


def setup(model_path: Path):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(128 * 1024**2)
    mx.random.seed(SEED)
    model, tokenizer = load(str(model_path))
    model.freeze()
    linear_to_lora_layers(
        model,
        LORA_LAYERS,
        {"rank": LORA_RANK, "dropout": 0.0, "scale": LORA_SCALE},
    )
    model.eval()
    mx.eval(model.parameters())
    parameters = dict(flatten_tree(model.trainable_parameters()))
    if not parameters or any("lora" not in name for name in parameters):
        raise ValueError("BTL Advance local profile requires only LoRA trainable parameters")
    return model, tokenizer, parameters


def flatten_tree(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(tree, dict):
        flattened = []
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(flatten_tree(value, name))
        return flattened
    if isinstance(tree, (list, tuple)):
        flattened = []
        for index, value in enumerate(tree):
            flattened.extend(flatten_tree(value, f"{prefix}.{index}"))
        return flattened
    return [(prefix, tree)]


def tree_map(function, tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: tree_map(function, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [tree_map(function, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(function, value) for value in tree)
    return function(tree)


def tree_zip_map(function, left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("Gradient trees have different keys")
        return {key: tree_zip_map(function, left[key], right[key]) for key in left}
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Gradient trees have different lengths")
        return [tree_zip_map(function, a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("Gradient trees have different lengths")
        return tuple(tree_zip_map(function, a, b) for a, b in zip(left, right))
    return function(left, right)


def adapter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(flatten_tree(model.trainable_parameters())):
        digest.update(name.encode())
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def action_distribution(model, tokenizer, task: dict[str, Any], ids: list[int]):
    import mlx.core as mx

    logits = model(mx.array(ids)[None])
    logits = logits[0, -1].astype(mx.float32)
    selected = logits[mx.array(action_token_ids(tokenizer))] / TEMPERATURE
    selected = selected - mx.logsumexp(selected)
    return mx.exp(selected), selected


def reward(task: dict[str, Any], action: int) -> float:
    if action not in {0, 1}:
        raise ValueError("Action must be zero or one")
    return float(action == task["expected_action"])


def group_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("A group needs at least two rollouts")
    mean = float(np.mean(rewards))
    deviation = float(np.std(rewards))
    if not math.isfinite(deviation):
        raise ValueError("Nonfinite reward spread")
    if deviation < 1e-8:
        return [0.0] * len(rewards)
    return [float((value - mean) / (deviation + 1e-8)) for value in rewards]


def policy_loss(model, payload: dict[str, Any]):
    import mlx.core as mx

    ids = mx.array(payload["prompt_ids"])
    action_ids = mx.array(payload["action_token_ids"])
    logits = model(ids[None])[0, -1].astype(mx.float32)
    selected = logits[action_ids] / TEMPERATURE
    log_probs = selected - mx.logsumexp(selected)
    return -payload["advantage"] * log_probs[payload["action"]]


def encode_optimizer_state(state: Any, arrays: dict[str, Any], prefix: str = "state") -> Any:
    import mlx.core as mx

    if isinstance(state, dict):
        return {key: encode_optimizer_state(value, arrays, f"{prefix}.{key}") for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return [encode_optimizer_state(value, arrays, f"{prefix}.{index}") for index, value in enumerate(state)]
    if isinstance(state, mx.array):
        name = f"array_{len(arrays)}"
        arrays[name] = state
        return {"array": name}
    return {"scalar": state}


def decode_optimizer_state(encoded: Any, arrays: dict[str, Any]) -> Any:
    if isinstance(encoded, dict) and "array" in encoded:
        return arrays[encoded["array"]]
    if isinstance(encoded, dict) and "scalar" in encoded:
        return encoded["scalar"]
    if isinstance(encoded, dict):
        return {key: decode_optimizer_state(value, arrays) for key, value in encoded.items()}
    if isinstance(encoded, list):
        return [decode_optimizer_state(value, arrays) for value in encoded]
    raise ValueError("Invalid optimizer state encoding")


def save_checkpoint(model, optimizer, output: Path, step: int, protocol: dict, rng: np.random.Generator) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    mx.eval(model.parameters(), optimizer.state)
    mx.save_safetensors(str(checkpoint / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    arrays: dict[str, Any] = {}
    encoded = encode_optimizer_state(optimizer.state, arrays)
    mx.save_safetensors(str(checkpoint / "optimizer.safetensors"), arrays)
    (checkpoint / "rng.pkl").write_bytes(pickle.dumps(rng.bit_generator.state))
    write_json(checkpoint / "checkpoint.json", {
        "step": step,
        "protocol_sha256": contract_hash(protocol),
        "files": {
            name: sha256(checkpoint / name)
            for name in ("adapter.safetensors", "optimizer.safetensors", "rng.pkl")
        },
        "optimizer_structure": encoded,
    })


def restore_checkpoint(model, optimizer, checkpoint: Path, protocol: dict):
    import mlx.core as mx

    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if metadata["protocol_sha256"] != contract_hash(protocol):
        raise ValueError("Checkpoint protocol differs from the requested run")
    for name, expected in metadata["files"].items():
        path = checkpoint / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Changed checkpoint file: {name}")
    model.load_weights(str(checkpoint / "adapter.safetensors"), strict=False)
    arrays = mx.load(str(checkpoint / "optimizer.safetensors"))
    optimizer.state = decode_optimizer_state(metadata["optimizer_structure"], arrays)
    mx.eval(model.parameters(), optimizer.state)
    rng = np.random.default_rng()
    rng.bit_generator.state = pickle.loads((checkpoint / "rng.pkl").read_bytes())
    return int(metadata["step"]), rng


def evaluate(model, tokenizer, tasks: list[dict[str, Any]], *, greedy: bool) -> dict[str, Any]:
    import mlx.core as mx

    results = []
    action_ids = action_token_ids(tokenizer)
    for task in tasks:
        ids = prompt_ids(tokenizer, task)
        probabilities, _ = action_distribution(model, tokenizer, task, ids)
        mx.eval(probabilities)
        action = int(np.argmax(np.asarray(probabilities))) if greedy else int(np.random.default_rng(task["number"]).choice(2, p=np.asarray(probabilities)))
        results.append({
            "id": task["id"],
            "action": action,
            "token_id": action_ids[action],
            "expected_action": task["expected_action"],
            "reward": reward(task, action),
            "probabilities": np.asarray(probabilities).tolist(),
        })
    return {"reward_sum": sum(result["reward"] for result in results),
            "passed": sum(result["reward"] for result in results),
            "total": len(results), "records": results}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    started = time.monotonic()
    model, tokenizer, parameters = setup(args.model)
    action_ids = action_token_ids(tokenizer)
    protocol = {
        "schema_version": 1,
        "engine": "BTL Advance",
        "backend": "mlx-local",
        "model": {"path": str(args.model), "revision": args.model.name},
        "taskset": {"name": "parity-bandit-v1", "train": 48, "evaluation": 24},
        "objective": "group-relative-policy-gradient",
        "steps": args.total_steps or args.steps,
        "group_size": args.group_size,
        "temperature": TEMPERATURE,
        "lora": {"layers": LORA_LAYERS, "rank": LORA_RANK, "scale": LORA_SCALE},
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
    }
    contract = contract_hash(protocol)
    train_tasks = make_tasks("train", protocol["taskset"]["train"])
    evaluation_tasks = make_tasks("evaluation", protocol["taskset"]["evaluation"])
    optimizer = optim.Adam(learning_rate=LEARNING_RATE, bias_correction=True)
    rng = np.random.default_rng(SEED)
    first_step = 0
    if args.resume:
        first_step, rng = restore_checkpoint(model, optimizer, args.resume, protocol)
    if first_step >= args.steps:
        raise ValueError("Checkpoint already reaches the requested final step")
    initial_hash = adapter_hash(model)
    base_eval = evaluate(model, tokenizer, evaluation_tasks, greedy=True)
    curve = []
    dead_groups = 0
    grad_fn = nn.value_and_grad(model, policy_loss)
    for step in range(first_step, args.steps):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("BTL Advance local run exceeded its wall-clock budget")
        task = train_tasks[step % len(train_tasks)]
        ids = prompt_ids(tokenizer, task)
        probabilities, _ = action_distribution(model, tokenizer, task, ids)
        mx.eval(probabilities)
        probabilities_np = np.asarray(probabilities, dtype=np.float64)
        total_probability = float(probabilities_np.sum())
        if not math.isfinite(total_probability) or total_probability <= 0:
            raise ValueError("Invalid action probability mass")
        probabilities_np /= total_probability
        actions = [int(rng.choice(2, p=probabilities_np)) for _ in range(args.group_size)]
        rewards = [reward(task, action) for action in actions]
        advantages = group_advantages(rewards)
        if not any(abs(value) > 0 for value in advantages):
            dead_groups += 1
            checkpoint = args.out / f"checkpoint-{step + 1}"
            if checkpoint.exists():
                raise ValueError("Checkpoint path already exists")
            save_checkpoint(model, optimizer, args.out, step + 1, protocol, rng)
            curve.append({"step": step + 1, "task_id": task["id"], "actions": actions,
                          "rewards": rewards, "advantages": advantages, "updated": False,
                          "dead_group": True, "seconds": time.monotonic() - started})
            write_json(args.out / "curve.json", curve)
            continue
        gradients = None
        losses = []
        tick = time.monotonic()
        # Qwen3.5's fast inference CustomKernel has no VJP. MLX selects the
        # differentiable Gated DeltaNet implementation only in training mode.
        model.train()
        for action, advantage in zip(actions, advantages):
            payload = {"prompt_ids": ids, "action_token_ids": action_ids,
                       "action": action, "advantage": advantage}
            loss, grad = grad_fn(model, payload)
            mx.eval(loss, grad)
            losses.append(float(loss))
            gradients = grad if gradients is None else tree_zip_map(lambda left, right: left + right, gradients, grad)
        gradients = tree_map(lambda value: value / args.group_size, gradients)
        norm = math.sqrt(sum(float(mx.sum(value.astype(mx.float32) ** 2)) for _, value in tree_flatten(gradients)))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Nonfinite or zero RL gradient")
        gradients = tree_map(lambda value: value * min(1.0, 1.0 / norm), gradients)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        model.eval()
        save_checkpoint(model, optimizer, args.out, step + 1, protocol, rng)
        record = {"step": step + 1, "task_id": task["id"], "actions": actions,
                  "rewards": rewards, "advantages": advantages, "updated": True,
                  "dead_group": False, "loss_mean": float(np.mean(losses)),
                  "gradient_norm_before_clip": norm, "seconds": time.monotonic() - tick,
                  "probabilities": probabilities_np.tolist(), "action_token_ids": action_ids}
        curve.append(record)
        write_json(args.out / "curve.json", curve)
    final_hash = adapter_hash(model)
    if final_hash == initial_hash:
        raise ValueError("RL optimizer did not change adapter parameters")
    final_eval = evaluate(model, tokenizer, evaluation_tasks, greedy=True)
    result = {"engine": "BTL Advance", "backend": "mlx-local", "model": str(args.model),
              "protocol_sha256": contract, "initial_step": first_step, "final_step": args.steps,
              "initial_adapter_sha256": initial_hash, "final_adapter_sha256": final_hash,
              "base_evaluation": base_eval, "final_evaluation": final_eval, "curve": curve,
              "dead_groups": dead_groups, "trainable_parameters": sum(value.size for value in parameters.values()),
              "peak_memory_bytes": mx.get_peak_memory(), "elapsed_seconds": time.monotonic() - started,
              "external_spend_usd": 0, "capability_claim": False,
              "scope": "Local contextual-bandit RL execution and recovery; not agentic-RL or Tinfield capability evidence",
              "gates": {"optimizer_update": any(record["updated"] for record in curve),
                        "finite_gradients": all(math.isfinite(record.get("gradient_norm_before_clip", 0)) for record in curve if record["updated"]),
                        "adapter_changed": final_hash != initial_hash, "behavioral_evaluation": "descriptive-only"}}
    write_json(args.out / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="BTL Advance local RL probe")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--total-steps", type=int, help="Final step target shared by a split run and its resume")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.group_size < 2 or args.max_seconds <= 0:
        parser.error("steps and max-seconds must be positive; group-size must be at least two")
    if args.total_steps is not None and (args.total_steps < args.steps or args.total_steps <= 0):
        parser.error("total-steps must be at least steps")
    if not args.model.is_dir() or not args.model.is_absolute():
        parser.error("--model must be an absolute local model directory")
    if args.resume and (not args.resume.is_dir() or not args.resume.is_absolute()):
        parser.error("--resume must be an absolute checkpoint directory")
    if args.out.exists():
        if any(args.out.iterdir()):
            parser.error("--out must be a new or empty directory")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    result = run(args)
    print(json.dumps({"status": "passed", "final_step": result["final_step"],
                      "base_reward": result["base_evaluation"]["passed"],
                      "final_reward": result["final_evaluation"]["passed"],
                      "adapter_changed": result["gates"]["adapter_changed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
