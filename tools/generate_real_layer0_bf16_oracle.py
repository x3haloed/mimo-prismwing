#!/usr/bin/env python3
"""Generate the independent PW-0056 real layer-0 BF16 trace."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from safetensors import safe_open
import torch

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
PROMPT_IDS = [151644,8948,198,2610,525,20740,25612,11,264,10950,15235,17847,44936,553,71449,13,151645,151644,872,198,9707,151645,151644,77091,198,151667,151668]
SHARD = "model_pp0_ep0_shard1.safetensors"
PROTECTED_SERVICES = ("ChatGPT", "WindowServer", "nxnode", "syncthing")


def command(*arguments: str) -> str:
    return subprocess.run(arguments, check=True, text=True, capture_output=True).stdout


def memory_free_percent() -> int:
    marker = "System-wide memory free percentage:"
    for line in command("/usr/bin/memory_pressure", "-Q").splitlines():
        if line.strip().startswith(marker):
            return int(line.split(marker, 1)[1].strip().removesuffix("%"))
    raise ValueError("memory_pressure output mismatch")


def swap_used_bytes() -> int:
    fields = command("/usr/sbin/sysctl", "-n", "vm.swapusage").split()
    value = fields[fields.index("used") + 2]
    multiplier = 1024**2 if value.endswith("M") else 1024**3
    return round(float(value[:-1]) * multiplier)


def throttled_pages() -> int:
    for line in command("/usr/bin/vm_stat").splitlines():
        if line.startswith("Pages throttled:"):
            return int(line.split(":", 1)[1].strip().removesuffix("."))
    raise ValueError("vm_stat output mismatch")


def service_pids() -> dict[str, list[int]]:
    result = {name: [] for name in PROTECTED_SERVICES}
    for line in command("/bin/ps", "-axo", "pid=,comm=").splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            name = Path(fields[1]).name
            if name in result:
                result[name].append(int(fields[0]))
    return result


class Safety:
    def __init__(self) -> None:
        self.baseline_swap = swap_used_bytes()
        self.baseline_throttled = throttled_pages()
        self.baseline_services = {k for k, v in service_pids().items() if v}
        self.snapshots: list[dict] = []
        self.check("process_start")

    def check(self, phase: str) -> None:
        gc.collect()
        free = memory_free_percent()
        swap_growth = max(0, swap_used_bytes() - self.baseline_swap)
        new_throttled = max(0, throttled_pages() - self.baseline_throttled)
        rss = int(command("/bin/ps", "-o", "rss=", "-p", str(os.getpid())).strip()) * 1024
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        services = service_pids()
        snapshot = {"phase": phase, "system_memory_free_percent": free,
                    "swap_growth_bytes": swap_growth, "new_throttled_pages": new_throttled,
                    "process_resident_bytes": rss, "process_peak_resident_bytes": peak,
                    "protected_service_pids": services}
        self.snapshots.append(snapshot)
        if free < 20 or rss > 8 * 1024**3 or peak > 8 * 1024**3 or rss > 4 * 1024**3:
            raise RuntimeError(f"safety stop at {phase}: memory limit")
        if swap_growth > 512 * 1024**2 or new_throttled != 0:
            raise RuntimeError(f"safety stop at {phase}: VM pressure")
        if any(not services[name] for name in self.baseline_services):
            raise RuntimeError(f"safety stop at {phase}: protected service disappeared")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_shard(checkpoint: Path, verification_path: Path) -> Path:
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("checkpoint verification SHA-256 mismatch")
    verification = json.loads(verification_path.read_text())
    if (verification.get("revision") != REVISION or not verification.get("complete")):
        raise ValueError("checkpoint verification identity mismatch")
    record = next((x for x in verification["files"] if x["path"] == SHARD), None)
    path = checkpoint / SHARD
    stat = path.stat()
    if (not record or record.get("status") != "verified" or stat.st_size != record["bytes"]
            or stat.st_dev != record["device"] or stat.st_ino != record["inode"]
            or stat.st_mtime_ns != record["modified_ns"]):
        raise ValueError("verified shard identity changed")
    return path


def tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def embedding(path: Path) -> torch.Tensor:
    rows = []
    with safe_open(path, framework="pt", device="cpu") as source:
        view = source.get_slice("model.embed_tokens.weight")
        for token in PROMPT_IDS:
            rows.append(view[token:token + 1])
    result = torch.cat(rows, dim=0)
    if result.dtype != torch.bfloat16 or tuple(result.shape) != (27, 4096):
        raise ValueError("embedding layout mismatch")
    return result


def rms_norm(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = values.float().pow(2).mean(-1, keepdim=True)
    normalized = values.float() * torch.rsqrt(variance + 1e-5)
    return weight * normalized.to(values.dtype)


def dynamic_input(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    grouped = values.float().reshape(rows, columns // 128, 128)
    scales = grouped.abs().amax(-1).clamp(min=1e-10) / 448.0
    encoded = torch.clamp(grouped / scales.unsqueeze(-1), -448.0, 448.0).to(torch.float8_e4m3fn)
    return (encoded.float() * scales.unsqueeze(-1)).reshape(rows, columns)


def dequant_weight(path: Path, name: str, full_qkv: bool = False) -> torch.Tensor:
    weight = tensor(path, name).float()
    scale = tensor(path, name + "_scale_inv").float()
    if full_qkv:
        if tuple(weight.shape) != (13568, 4096) or tuple(scale.shape) != (108, 32):
            raise ValueError("full-QKV layout mismatch")
        rows = torch.empty(13568, dtype=torch.int64)
        rows[:12288] = torch.arange(12288) // 128
        for head in range(4):
            start = 12288 + head * 192
            rows[start:start + 192] = 96 + head * 2 + torch.arange(192) // 128
        rows[13056:] = 104 + torch.arange(512) // 128
        expanded = scale[rows].repeat_interleave(128, 1)
    else:
        expected = ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
        if tuple(scale.shape) != expected or weight.shape[0] % 128 or weight.shape[1] % 128:
            raise ValueError(f"{name}: FP8 layout mismatch")
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight * expanded


def fp8_linear(path: Path, name: str, values: torch.Tensor, full_qkv: bool = False) -> torch.Tensor:
    inputs = dynamic_input(values)
    weight = dequant_weight(path, name, full_qkv)
    output = (inputs @ weight.T).to(torch.bfloat16)
    del inputs, weight
    gc.collect()
    return output


def bf16_linear(path: Path, name: str, values: torch.Tensor) -> torch.Tensor:
    weight = tensor(path, name)
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"{name}: BF16 layout mismatch")
    output = (values.float() @ weight.float().T).to(torch.bfloat16)
    del weight
    gc.collect()
    return output


def apply_rope(values: torch.Tensor, theta: float) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    result = values.clone()
    for position in range(values.shape[0]):
        frequencies = inv * float(position)
        cos = torch.cat((frequencies, frequencies)).cos().to(torch.bfloat16)
        sin = torch.cat((frequencies, frequencies)).sin().to(torch.bfloat16)
        rotating = values[position, :, :64]
        half = torch.cat((-rotating[:, 32:], rotating[:, :32]), dim=-1)
        result[position, :, :64] = (rotating * cos) + (half * sin)
    return result


def write_capture(root: Path, name: str, values: torch.Tensor, safety: Safety) -> dict:
    widened = values.float().contiguous()
    payload = widened.numpy().astype("<f4", copy=False).tobytes()
    path = root / f"{name}.f32"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    safety.check(f"{name}_captured")
    return {"file": path.name, "shape": list(values.shape), "dtype": "BF16_widened_F32", "sha256": hashlib.sha256(payload).hexdigest()}


def generate(checkpoint: Path, verification: Path, output: Path) -> dict:
    started = time.monotonic()
    torch.set_num_threads(1)
    shard = load_verified_shard(checkpoint, verification)
    output.mkdir(parents=True, exist_ok=False)
    safety = Safety()
    captures = {}
    hidden = embedding(shard)
    captures["embedding"] = write_capture(output, "embedding", hidden, safety)
    input_norm = tensor(shard, "model.layers.0.input_layernorm.weight")
    normalized = rms_norm(hidden, input_norm)
    captures["input_norm"] = write_capture(output, "input_norm", normalized, safety)
    qkv = fp8_linear(shard, "model.layers.0.self_attn.qkv_proj.weight", normalized, True)
    captures["qkv"] = write_capture(output, "qkv", qkv, safety)
    q = apply_rope(qkv[:, :12288].reshape(27, 64, 192), 10_000_000.0)
    k = apply_rope(qkv[:, 12288:13056].reshape(27, 4, 192), 10_000_000.0)
    v = (qkv[:, 13056:].reshape(27, 4, 128) * 0.707).to(torch.bfloat16)
    captures["query"] = write_capture(output, "query", q, safety)
    captures["key"] = write_capture(output, "key", k, safety)
    captures["value"] = write_capture(output, "value", v, safety)
    core = torch.empty((27, 64, 128), dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    for position in range(27):
        for head in range(64):
            kv_head = head // 16
            scores = (q[position, head] @ k[:position + 1, kv_head].T) * scale
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            core[position, head] = probabilities @ v[:position + 1, kv_head]
    core = core.reshape(27, 8192)
    captures["attention"] = write_capture(output, "attention", core, safety)
    projected = bf16_linear(shard, "model.layers.0.self_attn.o_proj.weight", core)
    captures["attention_projection"] = write_capture(output, "attention_projection", projected, safety)
    post_attention = (hidden + projected).to(torch.bfloat16)
    captures["post_attention"] = write_capture(output, "post_attention", post_attention, safety)
    post_norm_weight = tensor(shard, "model.layers.0.post_attention_layernorm.weight")
    moe_input = rms_norm(post_attention, post_norm_weight)
    captures["post_attention_norm"] = write_capture(output, "post_attention_norm", moe_input, safety)
    gate = fp8_linear(shard, "model.layers.0.mlp.gate_proj.weight", moe_input)
    up = fp8_linear(shard, "model.layers.0.mlp.up_proj.weight", moe_input)
    captures["gate"] = write_capture(output, "gate", gate, safety)
    captures["up"] = write_capture(output, "up", up, safety)
    swiglu = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    captures["swiglu"] = write_capture(output, "swiglu", swiglu, safety)
    down = fp8_linear(shard, "model.layers.0.mlp.down_proj.weight", swiglu)
    captures["down"] = write_capture(output, "down", down, safety)
    final = (post_attention + down).to(torch.bfloat16)
    captures["final"] = write_capture(output, "final", final, safety)
    manifest = {
        "schema_version": 1,
        "semantic": "mimo_real_layer0_bf16_dynamic_fp8_oracle",
        "revision": REVISION,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "source_shard": SHARD,
        "prompt_token_ids": PROMPT_IDS,
        "numerics": "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries",
        "torch_version": torch.__version__,
        "captures": captures,
        "safety_snapshots": safety.snapshots,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "performance_claim": None,
    }
    atomic_write_new(output / "manifest.json", canonical_json(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.verification, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
