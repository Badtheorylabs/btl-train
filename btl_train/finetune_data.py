"""Tokenized causal-SFT input contract. No tokenizer or GPU imports."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


PACKAGES = ("unsloth", "unsloth_zoo", "torch", "transformers", "trl", "datasets", "peft")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config: dict):
    if config.get("schema_version") != 1 or config.get("method") not in {"lora", "qlora"}:
        raise ValueError("Fine-tuning v1 supports supervised lora or qlora only")
    for name in ("project_id", "model_directory", "dataset_manifest", "dataset_manifest_sha256", "model_manifest", "model_manifest_sha256"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Missing {name}")
    for name in ("dataset_manifest_sha256", "model_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", config[name]):
            raise ValueError(f"Invalid {name}")
    if config.get("precision") not in {"bf16", "fp16"}:
        raise ValueError("Declare bf16 or fp16 compute precision")
    for name in ("max_seq_length", "max_steps", "micro_batch_size", "gradient_accumulation_steps", "save_steps", "wall_seconds", "rank", "alpha"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.get("seed")) is not int:
        raise ValueError("Declare an integer seed")
    for name in ("learning_rate", "max_grad_norm"):
        n = config.get(name)
        if type(n) not in {int, float} or not math.isfinite(n) or n <= 0:
            raise ValueError(f"Invalid {name}")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets or any(not isinstance(s, str) or not s for s in targets):
        raise ValueError("Declare exact LoRA target module suffixes")
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate LoRA targets")
    if config.get("expected_model_type") not in {"qwen3", "qwen3_5"}:
        raise ValueError("Experimental execution profiles are dense Qwen3 and Qwen3.5")
    if not isinstance(config.get("versions"), dict) or set(config["versions"]) != set(PACKAGES):
        raise ValueError("Pin every backend package version")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", v) for v in config["versions"].values()):
        raise ValueError("Package pins must be exact versions")
    fingerprint(config)


def resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.exists():
        raise ValueError(f"Missing or out-of-root input: {value}")
    return path


def checked_json(root: Path, path: str, expected: str) -> tuple[Path, dict]:
    resolved = resolve(root, path)
    if digest(resolved) != expected:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return resolved, json.loads(resolved.read_text())


def check_row(row: dict, max_length: int) -> dict:
    if not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("source_id"), str) or not row["source_id"]:
        raise ValueError("Each row needs id and source_id")
    tokens, labels = row.get("input_ids"), row.get("labels")
    if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels) or len(tokens) < 2:
        raise ValueError(f"Invalid token/label arrays: {row['id']}")
    if len(tokens) > max_length:
        raise ValueError(f"Sequence exceeds configured length; no truncation: {row['id']}")
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Token IDs must be nonnegative integers")
    if any(type(label) is not int or label not in {-100, token} for token, label in zip(tokens, labels)):
        raise ValueError("Labels must match their token or be -100")
    if labels[0] != -100 or not any(label != -100 for label in labels[1:]):
        raise ValueError("Require causal supervision after position zero")
    return {"input_ids": tokens, "labels": labels}


def load_split(path: Path, max_length: int) -> tuple[list[dict], dict]:
    rows, ids, sources, content = [], set(), set(), set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = check_row(raw, max_length)
            if raw["id"] in ids:
                raise ValueError(f"Duplicate row ID: {raw['id']}")
            ids.add(raw["id"]); sources.add(raw["source_id"])
            content.add(fingerprint({"input_ids": row["input_ids"]}))
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows, {"ids": ids, "sources": sources, "content": content}


def audit(root: Path, config: dict, *, check_weights: bool = False) -> dict:
    validate_config(config)
    path, manifest = checked_json(root, config["dataset_manifest"], config["dataset_manifest_sha256"])
    if manifest.get("schema_version") != 1 or manifest.get("accepted") is not True:
        raise ValueError("A version-1 accepted dataset manifest is required")
    _, model_manifest = checked_json(root, config["model_manifest"], config["model_manifest_sha256"])
    if manifest.get("tokenizer") != model_manifest.get("tokenizer") or not manifest.get("tokenizer"):
        raise ValueError("Dataset and model tokenizer identity differ")
    acceptance = manifest.get("acceptance_receipt", {})
    if not acceptance.get("path") or digest(resolve(path.parent, acceptance["path"])) != acceptance.get("sha256"):
        raise ValueError("Missing or changed upstream dataset acceptance receipt")
    datasets, facts = {}, {}
    for split in ("train", "evaluation"):
        entry = manifest[split]
        source = resolve(path.parent, entry["path"])
        if source.stat().st_size > config.get("max_dataset_bytes", 512 * 1024 * 1024):
            raise ValueError("Dataset exceeds the in-memory profile limit; no rows were truncated")
        before = source.stat()
        if digest(source) != entry["sha256"]:
            raise ValueError(f"Changed {split} data")
        datasets[split], facts[split] = load_split(source, config["max_seq_length"])
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Dataset changed during validation")
    for field in ("ids", "sources", "content"):
        if facts["train"][field] & facts["evaluation"][field]:
            raise ValueError(f"Train/evaluation overlap in {field}")
    model_dir = (root / config["model_directory"]).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError("Local model directory is missing")
    asset_root = model_dir.parent.parent if model_dir.parent.name == "snapshots" else model_dir
    files = model_manifest.get("files", [])
    if not files or not any(item["path"].endswith(".safetensors") for item in files):
        raise ValueError("Model manifest must list weights and assets")
    listed = {item["path"] for item in files}
    actual = {str(p.relative_to(model_dir)) for p in model_dir.rglob("*")
              if p.is_file() and ".cache" not in p.relative_to(model_dir).parts
              and (p.suffix in {".safetensors", ".json", ".jinja", ".model"}
                   or p.name in {"merges.txt", "vocab.txt"})}
    if not actual.issubset(listed):
        raise ValueError(f"Model manifest omits weight/config/tokenizer assets: {sorted(actual - listed)}")
    for index in model_dir.glob("*.safetensors.index.json"):
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards.issubset(listed):
            raise ValueError("Model manifest omits an indexed weight shard")
    if check_weights:
        for item in files:
            name = Path(item["path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Model asset path escapes its manifest")
            asset = (model_dir / name).resolve()
            if not asset.is_relative_to(asset_root) or not asset.is_file() or digest(asset) != item["sha256"]:
                raise ValueError(f"Changed model asset: {item['path']}")
    summary = {split: {"rows": len(rows), "tokens": sum(len(r["input_ids"]) for r in rows),
                       "supervised_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for r in rows)}
               for split, rows in datasets.items()}
    return {"datasets": datasets, "summary": summary, "model_directory": str(model_dir),
            "tokenizer": manifest["tokenizer"], "weights_verified": check_weights,
            "acceptance_scope": "Acceptance is supplied by the upstream data system; this engine checks structure and integrity"}


def pad_batch(rows: list[dict], pad_token_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in rows)
    return {"input_ids": [r["input_ids"] + [pad_token_id] * (width - len(r["input_ids"])) for r in rows],
            "labels": [r["labels"] + [-100] * (width - len(r["labels"])) for r in rows],
            "attention_mask": [[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]}
