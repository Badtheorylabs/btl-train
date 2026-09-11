"""Bounded child process execution, used only after explicit launch."""
import os
import signal
import subprocess
import time
from pathlib import Path


def run_process(command: list[str], cwd: Path, log: Path, seconds: float, env: dict,
                grace_seconds: float = 15) -> dict:
    if seconds <= 0 or grace_seconds < 0:
        raise ValueError("Invalid process deadline")
    started = time.monotonic()
    timed_out = False
    with log.open("xb") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not timed_out:
                raise
        finally:
            # Children cannot outlive the invocation even if the leader exits early.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return {"exit_code": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started, "log": str(log),
            "external_cost_usd": None, "cost_note": "No provider billing control or cost measurement"}
