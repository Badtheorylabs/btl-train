"""Explicit model-operation contracts and local input-integrity preflight."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


OPERATIONS = {
    "pretrain": {"inputs": {"data", "evaluation"}, "backends": {"torchtitan", "megatron-core"}},
    "continued-pretrain": {"inputs": {"data", "evaluation"}, "backends": {"torchtitan", "megatron-core"}},
    "sft": {"inputs": {"data", "evaluation"}, "backends": {"unsloth-core", "trl", "prime-rl", "nemo-rl"}},
    "preference": {"inputs": {"data", "evaluation"}, "backends": {"trl", "unsloth-core", "nemo-rl"}},
    "distill": {"inputs": {"data", "evaluation"}, "backends": {"prime-rl", "nemo-rl", "skyrl"}},
    "rl": {"inputs": {"evaluation"}, "backends": {"prime-rl", "areal", "skyrl", "nemo-rl", "slime"}},
    "inference": {"inputs": {"requests"}, "backends": {"vllm", "sglang"}},
    "evaluate": {"inputs": {"evaluation"}, "backends": {"verifiers", "harbor"}},
    "export": {"inputs": {"checkpoint"}, "backends": {"transformers", "peft", "llama.cpp"}},
}


def pin(value, field: str):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError(f"{field} requires a full commit or SHA-256 pin")


def reference(value, field: str):
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"].strip():
        raise ValueError(f"{field} needs an id and revision")
    pin(value.get("revision"), field + ".revision")


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Operation specification must be a version-1 object")
    if not isinstance(spec.get("operation"), str) or spec["operation"] not in OPERATIONS:
        raise ValueError("Specify schema_version=1 and an explicit supported operation")
    operation = spec["operation"]
    rule = OPERATIONS[operation]
    if spec.get("backend") not in rule["backends"]:
        raise ValueError(f"Backend is not a candidate for {operation}")
    reference(spec.get("model"), "model")
    reference(spec.get("engine"), "engine")
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, dict) or not rule["inputs"].issubset(inputs):
        raise ValueError(f"{operation} requires input roles: {sorted(rule['inputs'])}")
    for role, item in inputs.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"Input {role} needs a local path")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Input {role} requires SHA-256")
    if operation == "distill":
        reference(spec.get("teacher"), "teacher")
        if spec.get("objective") not in {"teacher-text", "on-policy-kl"}:
            raise ValueError("Distillation requires teacher-text or on-policy-kl objective")
    if operation == "preference" and spec.get("objective") not in {"dpo", "ipo", "orpo"}:
        raise ValueError("Preference optimization requires an explicit objective")
    if operation == "rl":
        environment = spec.get("environment", {})
        for field in ("taskset", "harness", "reward"):
            if not environment.get(field):
                raise ValueError(f"RL requires environment.{field}")
        pin(environment.get("revision"), "environment.revision")
        if not spec.get("objective"):
            raise ValueError("RL requires an explicit algorithm/objective")
    if operation == "export" and not spec.get("format"):
        raise ValueError("Export requires an explicit target format")
    if operation != "export":
        for field in ("context_length", "effective_batch_size"):
            if type(spec.get(field)) is not int or spec[field] <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(spec.get("precision"), str) or not spec["precision"]:
            raise ValueError("Specify precision explicitly")
    budget = spec.get("budget", {})
    if type(budget.get("wall_seconds")) is not int or budget["wall_seconds"] <= 0:
        raise ValueError("Declare a positive wall-clock budget")
    dollars = budget.get("external_spend_usd")
    if type(dollars) not in {int, float} or not math.isfinite(dollars) or dollars < 0:
        raise ValueError("Declare a finite, nonnegative external spending ceiling")
    if not spec.get("project_id") or not spec.get("name"):
        raise ValueError("Operation needs project_id and name")
    encoded = json.dumps(spec, sort_keys=True, allow_nan=False)
    return {"schema_version": 1, "operation": operation, "backend": spec["backend"],
            "spec": json.loads(encoded), "spec_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "stages": {"plan": "implemented", "input-preflight": "implemented",
                       "execute": "not-implemented", "evaluate-output": "not-implemented",
                       "export-output": "not-implemented"},
            "runnable_training": False,
            "scope": "Pinned operation contract; not architecture, hardware or capability validation"}


def preflight(root: Path, resolved: dict, max_hash_bytes: int = 64 * 1024 * 1024) -> dict:
    root = root.resolve()
    spec = resolved["spec"]
    if validate_spec(spec)["spec_sha256"] != resolved["spec_sha256"]:
        raise ValueError("Operation plan changed")
    if type(max_hash_bytes) is not int or max_hash_bytes <= 0:
        raise ValueError("Hash byte budget must be positive")
    checks, consumed = [], 0
    for role, item in spec["inputs"].items():
        path = (root / item["path"]).resolve()
        check = {"role": role, "path": item["path"], "passed": False}
        if not path.is_relative_to(root) or not path.is_file():
            check["error"] = "Input is missing or outside the workspace"
        else:
            before = path.stat()
            if before.st_size > max_hash_bytes - consumed:
                check["error"] = "Hash budget exceeded; reference a small immutable manifest or increase the limit"
            else:
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                consumed += before.st_size
                after = path.stat()
                stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (
                    after.st_size, after.st_mtime_ns, after.st_ino)
                check |= {"passed": stable and actual == item["sha256"], "actual_sha256": actual,
                          "expected_sha256": item["sha256"], "bytes": before.st_size}
        checks.append(check)
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "stage": "input-preflight", "evidence_class": "integrity", "hashed_bytes": consumed,
            "model_calls": 0, "external_spend_usd": 0, "training_ready": False,
            "not_checked": ["remote revisions exist", "model loading", "dataset semantics",
                            "GPU compatibility", "gradients", "checkpoint recovery", "behavioral quality"],
            "scope": "Operation specification and local input integrity only; no model execution"}
