#!/usr/bin/env python3
"""Build a learned MiMo MTP attention fixture from the pinned local tensor file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np
from safetensors import safe_open

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json

SOURCE_SHA256 = "a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
NORM = "model.mtp.layers.0.input_layernorm.weight"
QKV = "model.mtp.layers.0.self_attn.qkv_proj.weight"
SCALE = "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv"
SINK = "model.mtp.layers.0.self_attn.attention_sink_bias"
OUT = "model.mtp.layers.0.self_attn.o_proj.weight"
EXPECTED = {
    NORM: {"dtype":"BF16","shape":[4096],"data_offsets":[67325440,67333632]},
    QKV: {"dtype":"F8_E4M3","shape":[14848,4096],"data_offsets":[604295040,665112448]},
    SCALE: {"dtype":"F32","shape":[116,32],"data_offsets":[49152,64000]},
    SINK: {"dtype":"BF16","shape":[64],"data_offsets":[67341824,67341952]},
    OUT: {"dtype":"BF16","shape":[4096,8192],"data_offsets":[67341952,134450816]},
}
C4 = np.array([-0.173926,-0.117195,-0.089527,-0.068756,-0.051262,-0.035597,-0.020989,-0.006938,0.006938,0.020989,0.035597,0.051262,0.068756,0.089527,0.117195,0.173926], dtype=np.float32)
T4 = np.array([-0.145560,-0.103361,-0.079142,-0.060009,-0.043430,-0.028293,-0.013963,0,0.013963,0.028293,0.043430,0.060009,0.079142,0.103361,0.145560], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_signs(source: str, name: str) -> np.ndarray:
    marker = f"turbo_cpu_{name}[128] = {{"
    start = source.index(marker) + len(marker)
    end = source.index("};", start)
    values = np.array([float(item) for item in source[start:end].replace("\n", "").split(",") if item], dtype=np.float32)
    if values.shape != (128,) or not np.all(np.abs(values) == 1):
        raise ValueError(f"invalid locked WHT {name}")
    return values


def wht(values: np.ndarray, inverse: bool, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32, copy=True)
    first, second = (s2, s1) if inverse else (s1, s2)
    for base in range(0, result.size, 128):
        block = result[base:base+128] * first
        width = 1
        while width < 128:
            block = block.reshape(-1, 2 * width)
            left, right = block[:, :width].copy(), block[:, width:].copy()
            block[:, :width], block[:, width:] = left + right, left - right
            block = block.reshape(128)
            width *= 2
        result[base:base+128] = block * np.float32(1 / math.sqrt(128)) * second
    return result


def quantize4(values: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> bytes:
    payload = bytearray()
    for base in range(0, values.size, 128):
        block = values[base:base+128].astype(np.float32)
        norm = np.float32(np.sqrt(np.sum(block * block, dtype=np.float32)))
        rotated = wht(block / norm if norm > 1e-10 else np.zeros(128, np.float32), False, s1, s2)
        indices = np.searchsorted(T4, rotated, side="right").astype(np.uint8)
        recon = np.float32(np.sqrt(np.sum(C4[indices] ** 2, dtype=np.float32)))
        corrected = np.float16(norm / recon if recon > 1e-10 else norm)
        payload.extend(corrected.tobytes()); payload.extend(b"\0\0")
        payload.extend((indices[0::2] | (indices[1::2] << 4)).tobytes())
    return bytes(payload)


def dequant4(payload: bytes, count: int) -> np.ndarray:
    result = np.empty(count, np.float32)
    for block in range(count // 128):
        offset = block * 68
        norm = np.frombuffer(payload[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
        packed = np.frombuffer(payload[offset+4:offset+68], dtype=np.uint8)
        indices = np.empty(128, np.uint8); indices[0::2] = packed & 15; indices[1::2] = packed >> 4
        result[block*128:(block+1)*128] = C4[indices] * norm
    return result


def affine_reconstruct_rotated(
    values: np.ndarray,
    bits: int,
    s1: np.ndarray,
    s2: np.ndarray,
) -> np.ndarray:
    """WHT-transform and reconstruct independent 128-value affine blocks."""
    if bits not in (4, 5, 6, 8):
        raise ValueError("affine WHT bit depth must be one of 4, 5, 6, or 8")
    if values.ndim != 1 or values.size % 128:
        raise ValueError("affine WHT input must contain complete 128-value blocks")
    rotated = wht(values, False, s1, s2)
    reconstructed = np.empty_like(rotated)
    qmax = (1 << (bits - 1)) - 1
    for base in range(0, values.size, 128):
        block = rotated[base:base + 128]
        maximum = np.max(np.abs(block))
        scale16 = np.float16(maximum / np.float32(qmax)) if maximum > 0 else np.float16(0)
        scale = np.float32(scale16)
        if not np.isfinite(scale):
            raise ValueError("non-finite affine WHT scale")
        if scale == 0:
            reconstructed[base:base + 128] = 0
            continue
        codes = np.clip(np.rint(block / scale), -qmax, qmax).astype(np.int16)
        reconstructed[base:base + 128] = codes.astype(np.float32) * scale
    return reconstructed


def rope(values: np.ndarray, position: int) -> np.ndarray:
    result = values.copy()
    pair = np.arange(32, dtype=np.float32)
    angle = np.float32(position) / np.power(np.float32(10_000), 2 * pair / 64)
    first, second = result[:32].copy(), result[32:64].copy()
    result[:32] = first * np.cos(angle) - second * np.sin(angle)
    result[32:64] = second * np.cos(angle) + first * np.sin(angle)
    return result


def attention(query, keys, values, sinks):
    outputs = np.empty((64, 128), np.float32)
    for head in range(64):
        kv = head // 8
        scores = keys[:, kv] @ query[head] / np.float32(math.sqrt(192))
        logits = np.concatenate([scores, sinks[head:head+1]])
        probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
        outputs[head] = probabilities[:-1] @ values[:, kv]
    return outputs


def generate(checkpoint: Path, atomic_source: Path) -> dict:
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise ValueError("MTP source SHA-256 mismatch")
    with checkpoint.open("rb") as source:
        header_size=struct.unpack("<Q",source.read(8))[0];header=json.loads(source.read(header_size))
    for name,metadata in EXPECTED.items():
        if header.get(name) != metadata: raise ValueError(f"MTP metadata mismatch: {name}")
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        norm = tensors.get_tensor(NORM).float().numpy()
        raw_qkv = tensors.get_tensor(QKV).float().numpy()
        scales = tensors.get_tensor(SCALE).float().numpy()
        sinks = tensors.get_tensor(SINK).float().numpy()
        output_weight = tensors.get_tensor(OUT).float().numpy()
    if norm.shape != (4096,) or raw_qkv.shape != (14848,4096) or scales.shape != (116,32) or sinks.shape != (64,) or output_weight.shape != (4096,8192):
        raise ValueError("learned MTP attention tensor shape mismatch")
    payload_start=8+header_size
    for name,decoded in ((NORM,norm),(SINK,sinks),(OUT,output_weight)):
        start,end=EXPECTED[name]["data_offsets"];raw=np.memmap(checkpoint,dtype="<u2",mode="r",offset=payload_start+start,shape=((end-start)//2,));flat=decoded.reshape(-1)
        for offset in range(0,raw.size,1_000_000):
            manual=(np.asarray(raw[offset:offset+1_000_000],dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
            if not np.array_equal(manual,flat[offset:offset+manual.size]): raise ValueError(f"BF16 decode mismatch: {name}")
    qkv_weight = raw_qkv * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
    rng = np.random.default_rng(260026)
    hidden = rng.standard_normal((17,4096), dtype=np.float32)
    variance = np.mean(hidden * hidden, axis=1, keepdims=True, dtype=np.float32)
    normalized = hidden * np.reciprocal(np.sqrt(variance + np.float32(1e-5))) * norm
    qkv_mx = mx.matmul(mx.array(normalized), mx.array(qkv_weight).T)
    mx.eval(qkv_mx); qkv = np.array(qkv_mx, copy=False)
    samples = [0,1,12288,13824]
    scalar = np.array([np.dot(normalized[0].astype(np.float64), qkv_weight[row].astype(np.float64)) for row in samples])
    sample_error = float(np.max(np.abs(qkv[0,samples].astype(np.float64)-scalar)))
    q = qkv[:,:12288].reshape(17,64,192)
    k = qkv[:,12288:13824].reshape(17,8,192)
    v = qkv[:,13824:].reshape(17,8,128) * np.float32(0.707)
    for token in range(17):
        for head in range(64): q[token,head] = rope(q[token,head], token)
        for head in range(8): k[token,head] = rope(k[token,head], token)
    source_attention = attention(q[-1], k, v, sinks)
    source_output_mx = mx.matmul(mx.array(source_attention.reshape(1,8192)), mx.array(output_weight).T)
    mx.eval(source_output_mx); source_output = np.array(source_output_mx, copy=False).astype(np.float32)
    source_text = atomic_source.read_text()
    s1, s2 = parse_signs(source_text,"s1"), parse_signs(source_text,"s2")
    packed_keys, packed_values = bytearray(), bytearray()
    decoded_k = np.empty_like(k); decoded_v = np.empty_like(v)
    for head in range(8):
        for token in range(17):
            key = np.pad(k[token,head], (0,64)); key_payload = quantize4(key,s1,s2); value_payload = quantize4(v[token,head],s1,s2)
            packed_keys.extend(key_payload); packed_values.extend(value_payload)
            decoded_k[token,head] = dequant4(key_payload,256)[:192]; decoded_v[token,head] = dequant4(value_payload,128)
    # Scores use rotated padded vectors, so reconstruct the exact packed-domain reference.
    candidate_attention = np.empty((64,128),np.float32); rotated_queries=[]
    source_k_turbo_v_attention=np.empty((64,128),np.float32);turbo_k_source_v_attention=np.empty((64,128),np.float32)
    candidate_values=np.empty_like(v)
    for head in range(8):
        for token in range(17):
            payload=bytes(packed_values[(head*17+token)*68:(head*17+token+1)*68]);candidate_values[token,head]=wht(dequant4(payload,128),True,s1,s2)
    for head in range(64):
        query = wht(np.pad(q[-1,head],(0,64)),False,s1,s2); rotated_queries.extend(query.tolist())
        kv=head//8; scores=[]
        for token in range(17):
            packed=bytes(packed_keys[(kv*17+token)*136:(kv*17+token+1)*136]); scores.append(np.dot(query,dequant4(packed,256))/np.float32(math.sqrt(192)))
        logits=np.append(np.array(scores,np.float32),sinks[head]); probs=np.exp(logits-logits.max());probs/=probs.sum()
        rotated=np.zeros(128,np.float32)
        for token in range(17):
            payload=bytes(packed_values[(kv*17+token)*68:(kv*17+token+1)*68]);rotated += probs[token]*dequant4(payload,128)
        candidate_attention[head]=wht(rotated,True,s1,s2)
        source_scores=k[:,kv]@q[-1,head]/np.float32(math.sqrt(192));source_logits=np.append(source_scores,sinks[head]);source_probs=np.exp(source_logits-source_logits.max());source_probs/=source_probs.sum()
        source_k_turbo_v_attention[head]=source_probs[:-1]@candidate_values[:,kv]
        turbo_k_source_v_attention[head]=probs[:-1]@v[:,kv]
    candidate_output_mx=mx.matmul(mx.array(candidate_attention.reshape(1,8192)),mx.array(output_weight).T)
    source_k_turbo_v_output_mx=mx.matmul(mx.array(source_k_turbo_v_attention.reshape(1,8192)),mx.array(output_weight).T)
    turbo_k_source_v_output_mx=mx.matmul(mx.array(turbo_k_source_v_attention.reshape(1,8192)),mx.array(output_weight).T)
    mx.eval(candidate_output_mx,source_k_turbo_v_output_mx,turbo_k_source_v_output_mx);candidate_output=np.array(candidate_output_mx,copy=False).astype(np.float32);source_k_turbo_v_output=np.array(source_k_turbo_v_output_mx,copy=False).astype(np.float32);turbo_k_source_v_output=np.array(turbo_k_source_v_output_mx,copy=False).astype(np.float32)
    def digest(array): return hashlib.sha256(np.asarray(array,dtype='<f4').tobytes()).hexdigest()
    affine_sweep = []
    for bits in (4, 5, 6, 8):
        affine_keys = np.empty((17, 8, 256), np.float32)
        affine_values = np.empty_like(v)
        for token in range(17):
            for kv in range(8):
                affine_keys[token, kv] = affine_reconstruct_rotated(
                    np.pad(k[token, kv], (0, 64)), bits, s1, s2
                )
                rotated_value = affine_reconstruct_rotated(v[token, kv], bits, s1, s2)
                affine_values[token, kv] = wht(rotated_value, True, s1, s2)
        affine_attention = np.empty((64, 128), np.float32)
        for head in range(64):
            kv = head // 8
            rotated_query = wht(np.pad(q[-1, head], (0, 64)), False, s1, s2)
            scores = affine_keys[:, kv] @ rotated_query / np.float32(math.sqrt(192))
            logits = np.append(scores, sinks[head])
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            affine_attention[head] = probabilities[:-1] @ affine_values[:, kv]
        affine_output_mx = mx.matmul(
            mx.array(affine_attention.reshape(1, 8192)), mx.array(output_weight).T
        )
        mx.eval(affine_output_mx)
        affine_output = np.array(affine_output_mx, copy=False).astype(np.float32)
        block_bytes = 2 + math.ceil(128 * bits / 8)
        total_bytes = (
            9 * 4 * 3 * block_bytes * 1_048_576
            + 39 * 8 * 128 * 3 * block_bytes
        )
        affine_sweep.append({
            "bits": bits,
            "scale_dtype": "float16",
            "rounding": "ties_to_even",
            "signed_code_range": [-(1 << (bits - 1)) + 1, (1 << (bits - 1)) - 1],
            "block_bytes": block_bytes,
            "max_context_cache_bytes": total_bytes,
            "max_context_cache_gib": total_bytes / (1024 ** 3),
            "attention_sha256": digest(affine_attention),
            "sublayer_output_sha256": digest(affine_output),
            "attention_relative_l2_vs_source": float(
                np.linalg.norm(affine_attention - source_attention) / np.linalg.norm(source_attention)
            ),
            "sublayer_relative_l2_vs_source": float(
                np.linalg.norm(affine_output - source_output) / np.linalg.norm(source_output)
            ),
        })
    return {"schema_version":1,"semantic":"mimo_mtp_real_attention_context17","source_revision":REVISION,"source_sha256":SOURCE_SHA256,"format":"turbo4","context":17,"q_heads":64,"kv_heads":8,"rotated_queries_f32":rotated_queries,"packed_keys_u8":list(packed_keys),"packed_values_u8":list(packed_values),"sinks_f32":sinks.tolist(),"expected_attention_f32":candidate_attention.reshape(-1).tolist(),"source_attention_sha256":digest(source_attention),"candidate_attention_sha256":digest(candidate_attention),"source_k_turbo_v_attention_sha256":digest(source_k_turbo_v_attention),"turbo_k_source_v_attention_sha256":digest(turbo_k_source_v_attention),"source_sublayer_output_sha256":digest(source_output),"candidate_sublayer_output_sha256":digest(candidate_output),"source_k_turbo_v_sublayer_output_sha256":digest(source_k_turbo_v_output),"turbo_k_source_v_sublayer_output_sha256":digest(turbo_k_source_v_output),"source_sublayer_output_first8":source_output.reshape(-1)[:8].tolist(),"candidate_sublayer_output_first8":candidate_output.reshape(-1)[:8].tolist(),"qkv_sample_rows":samples,"qkv_sample_max_abs_error":sample_error,"attention_relative_l2_vs_source":float(np.linalg.norm(candidate_attention-source_attention)/np.linalg.norm(source_attention)),"sublayer_relative_l2_vs_source":float(np.linalg.norm(candidate_output-source_output)/np.linalg.norm(source_output)),"source_k_turbo_v_attention_relative_l2_vs_source":float(np.linalg.norm(source_k_turbo_v_attention-source_attention)/np.linalg.norm(source_attention)),"source_k_turbo_v_sublayer_relative_l2_vs_source":float(np.linalg.norm(source_k_turbo_v_output-source_output)/np.linalg.norm(source_output)),"turbo_k_source_v_attention_relative_l2_vs_source":float(np.linalg.norm(turbo_k_source_v_attention-source_attention)/np.linalg.norm(source_attention)),"turbo_k_source_v_sublayer_relative_l2_vs_source":float(np.linalg.norm(turbo_k_source_v_output-source_output)/np.linalg.norm(source_output)),"wht_affine_sweep":affine_sweep}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",required=True,type=Path);parser.add_argument("--atomic-source",required=True,type=Path);parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    atomic_write_new(args.output,canonical_json(generate(args.checkpoint,args.atomic_source)))


if __name__ == "__main__": main()
