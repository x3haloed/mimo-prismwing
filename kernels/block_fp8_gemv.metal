#include <metal_stdlib>
using namespace metal;

inline float decode_f8_e4m3fn(uchar bits) {
    const float sign = (bits & 0x80) == 0 ? 1.0f : -1.0f;
    const int exponent = int((bits >> 3) & 0x0f);
    const int mantissa = int(bits & 0x07);
    if (exponent == 0) {
        return sign * exp2(-6.0f) * (float(mantissa) / 8.0f);
    }
    if (exponent == 15 && mantissa == 7) {
        return as_type<float>((uint(bits & 0x80) << 24) | 0x7ff00000u);
    }
    return sign * exp2(float(exponent - 7)) * (1.0f + float(mantissa) / 8.0f);
}

inline float pw_round_bf16(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if ((bits & 0x007fffffu) == 0u) {
            return value;
        }
        return as_type<float>((uint(ushort(bits >> 16) | 0x0040u)) << 16);
    }
    const uint bias = 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + bias) & 0xffff0000u);
}

kernel void dynamic_fp8_dequantized_group128(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float *decode_lut [[buffer(2)]],
    device atomic_uint *error_flags [[buffer(3)]],
    threadgroup float *maximums [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (lanes != 128) {
        if (lane == 0) {
            atomic_fetch_or_explicit(error_flags, 1u, memory_order_relaxed);
        }
        return;
    }
    const uint index = group * 128 + lane;
    const float value = input[index];
    if (!isfinite(value)) {
        atomic_fetch_or_explicit(error_flags, 2u, memory_order_relaxed);
        maximums[lane] = 0.0f;
    } else {
        maximums[lane] = abs(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 64; offset > 0; offset /= 2) {
        if (lane < offset) {
            maximums[lane] = max(maximums[lane], maximums[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = max(maximums[0], 1.0e-10f) / 448.0f;
    const float normalized = clamp(value / scale, -448.0f, 448.0f);
    if (normalized == 0.0f) {
        output[index] = as_type<uint>(normalized) >> 31 ? -0.0f : 0.0f;
        return;
    }
    const float magnitude = abs(normalized);
    uchar best = 0;
    float distance = INFINITY;
    for (uint candidate = 0; candidate <= 0x7e; ++candidate) {
        const float candidate_distance = abs(magnitude - decode_lut[candidate]);
        if (candidate_distance < distance ||
            (candidate_distance == distance && (candidate & 1u) == 0u &&
             (uint(best) & 1u) != 0u)) {
            best = uchar(candidate);
            distance = candidate_distance;
        }
    }
    const uchar encoded = best | (signbit(normalized) ? uchar(0x80) : uchar(0));
    output[index] = decode_lut[encoded] * scale;
}

// Preserve the two operands SGLang's block-FP8 kernels consume instead of
// collapsing the activation code and scale into a dequantized float. One
// threadgroup owns one contiguous 128-value token group.
kernel void dynamic_fp8_quantized_group128(
    device const float *input [[buffer(0)]],
    device uchar *codes [[buffer(1)]],
    device float *scales [[buffer(2)]],
    constant float *decode_lut [[buffer(3)]],
    device atomic_uint *error_flags [[buffer(4)]],
    threadgroup float *maximums [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (lanes != 128) {
        if (lane == 0) {
            atomic_fetch_or_explicit(error_flags, 1u, memory_order_relaxed);
        }
        return;
    }
    const uint index = group * 128 + lane;
    const float value = input[index];
    if (!isfinite(value)) {
        atomic_fetch_or_explicit(error_flags, 2u, memory_order_relaxed);
        maximums[lane] = 0.0f;
    } else {
        maximums[lane] = abs(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 64; offset > 0; offset /= 2) {
        if (lane < offset) {
            maximums[lane] = max(maximums[lane], maximums[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = max(maximums[0], 1.0e-10f) / 448.0f;
    if (lane == 0) {
        scales[group] = scale;
    }
    const float normalized = clamp(value / scale, -448.0f, 448.0f);
    if (normalized == 0.0f) {
        codes[index] = signbit(normalized) ? uchar(0x80) : uchar(0);
        return;
    }
    const float magnitude = abs(normalized);
    uchar best = 0;
    float distance = INFINITY;
    for (uint candidate = 0; candidate <= 0x7e; ++candidate) {
        const float candidate_distance = abs(magnitude - decode_lut[candidate]);
        if (candidate_distance < distance ||
            (candidate_distance == distance && (candidate & 1u) == 0u &&
             (uint(best) & 1u) != 0u)) {
            best = uchar(candidate);
            distance = candidate_distance;
        }
    }
    codes[index] = best | (signbit(normalized) ? uchar(0x80) : uchar(0));
}

kernel void bf16_staged_swiglu(
    device float *gate [[buffer(0)]],
    device float *up [[buffer(1)]],
    device float *hidden [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    device atomic_uint *error_flags [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    const float rounded_gate = pw_round_bf16(gate[index]);
    const float rounded_up = pw_round_bf16(up[index]);
    const float silu = pw_round_bf16(
        rounded_gate / (1.0f + exp(-rounded_gate)));
    const float result = pw_round_bf16(silu * rounded_up);
    if (!isfinite(rounded_gate) || !isfinite(rounded_up) || !isfinite(result)) {
        atomic_fetch_or_explicit(error_flags, 4u, memory_order_relaxed);
    }
    gate[index] = rounded_gate;
    up[index] = rounded_up;
    hidden[index] = result;
}

kernel void bf16_staged_swiglu_lut(
    device float *gate [[buffer(0)]],
    device float *up [[buffer(1)]],
    device float *hidden [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    device atomic_uint *error_flags [[buffer(4)]],
    constant float *silu_lut [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    const float rounded_gate = pw_round_bf16(gate[index]);
    const float rounded_up = pw_round_bf16(up[index]);
    const uint lut_index = as_type<uint>(rounded_gate) >> 16;
    const float silu = silu_lut[lut_index];
    const float result = pw_round_bf16(silu * rounded_up);
    if (!isfinite(rounded_gate) || !isfinite(rounded_up) || !isfinite(result)) {
        atomic_fetch_or_explicit(error_flags, 64u, memory_order_relaxed);
    }
    gate[index] = rounded_gate;
    up[index] = rounded_up;
    hidden[index] = result;
}

struct RoutedReductionShape {
    uint experts;
    uint width;
};

kernel void route_weighted_reduce_bf16(
    device float *expert_outputs [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device float *routed [[buffer(2)]],
    constant RoutedReductionShape &shape [[buffer(3)]],
    device atomic_uint *error_flags [[buffer(4)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= shape.width) {
        return;
    }
    if (shape.experts != 8) {
        atomic_fetch_or_explicit(error_flags, 8u, memory_order_relaxed);
        return;
    }
    float sum = 0.0f;
    for (uint expert = 0; expert < 8; ++expert) {
        const uint index = expert * shape.width + column;
        const float rounded = pw_round_bf16(expert_outputs[index]);
        expert_outputs[index] = rounded;
        sum += rounded * route_weights[expert];
    }
    const float result = pw_round_bf16(sum);
    if (!isfinite(result)) {
        atomic_fetch_or_explicit(error_flags, 16u, memory_order_relaxed);
    }
    routed[column] = result;
}

kernel void bf16_round_in_place(
    device float *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    device atomic_uint *error_flags [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    const float rounded = pw_round_bf16(values[index]);
    if (!isfinite(rounded)) {
        atomic_fetch_or_explicit(error_flags, 32u, memory_order_relaxed);
    }
    values[index] = rounded;
}

struct GemvShape {
    uint rows;
    uint columns;
    uint block_rows;
    uint block_columns;
};

struct BundleGemvOffsets {
    uint weights[5];
    uint scales[5];
};

kernel void swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    const float value = gate[index];
    output[index] = (value / (1.0f + exp(-value))) * up[index];
}

struct ScatterShape {
    uint count;
    uint width;
};

// The checkpoint stores four tensor-parallel QKV shards concatenated as
// [Q3072,K192,V128] each. Runtime outputs are globally deinterleaved Q, K, V.
inline uint full_qkv_source_row(uint logical_row) {
    if (logical_row < 12288) {
        const uint shard = logical_row / 3072;
        return shard * 3392 + logical_row % 3072;
    }
    if (logical_row < 13056) {
        const uint local = logical_row - 12288;
        return (local / 192) * 3392 + 3072 + local % 192;
    }
    const uint local = logical_row - 13056;
    return (local / 128) * 3392 + 3264 + local % 128;
}

inline uint full_qkv_source_scale_row(uint source_row) {
    const uint shard = source_row / 3392;
    const uint local = source_row % 3392;
    if (local < 3072) {
        return shard * 27 + local / 128;
    }
    if (local < 3264) {
        return shard * 27 + 24 + (local - 3072) / 128;
    }
    return shard * 27 + 26;
}

inline uint swa_qkv_source_row(uint logical_row) {
    if (logical_row < 12288) {
        const uint shard = logical_row / 3072;
        return shard * 3712 + logical_row % 3072;
    }
    if (logical_row < 13824) {
        const uint local = logical_row - 12288;
        return (local / 384) * 3712 + 3072 + local % 384;
    }
    const uint local = logical_row - 13824;
    return (local / 256) * 3712 + 3456 + local % 256;
}

kernel void route_weighted_scatter_add_f32(
    device const float *expert_output [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device float *block_output [[buffer(3)]],
    constant ScatterShape &shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint local_position = index / shape.width;
    const uint column = index % shape.width;
    if (local_position >= shape.count || column >= shape.width) {
        return;
    }
    const uint destination = positions[local_position] * shape.width + column;
    block_output[destination] +=
        expert_output[local_position * shape.width + column] *
        route_weights[local_position];
}

kernel void f32_gemm8_shared_weight(
    device const float *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows) {
        return;
    }
    const uint row_offset = row * shape.columns;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < shape.columns; column += lanes) {
        const float weight = weights[row_offset + column];
        for (uint position = 0; position < batch; ++position) {
            sums[position] += weight * input[position * shape.columns + column];
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] +=
                    partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

kernel void bf16_gemm8_shared_weight(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows) {
        return;
    }
    const uint row_offset = row * shape.columns;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < shape.columns; column += lanes) {
        const float weight = as_type<float>(uint(weights[row_offset + column]) << 16);
        for (uint position = 0; position < batch; ++position) {
            sums[position] += weight * input[position * shape.columns + column];
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] +=
                    partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

kernel void bf16_gemv_shared_weight(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows) {
        return;
    }
    const uint row_offset = row * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const float weight = as_type<float>(uint(weights[row_offset + column]) << 16);
        sum += weight * input[column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void compact_mixture_indexed_weighted_heads(
    device const float *features [[buffer(0)]],
    device const ushort *heads [[buffer(1)]],
    device const uint *selected_heads [[buffer(2)]],
    device const float *route_weights [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint4 &shape [[buffer(5)]],
    uint hidden_index [[thread_position_in_grid]]) {
    const uint feature_count = shape.y;
    const uint hidden = shape.z;
    const uint routes = shape.w;
    if (hidden_index >= hidden) {
        return;
    }
    float routed = 0.0f;
    for (uint route = 0; route < routes; ++route) {
        const uint head_offset = selected_heads[route] * feature_count * hidden;
        float expert = 0.0f;
        for (uint feature = 0; feature < feature_count; ++feature) {
            const uint bits = uint(heads[head_offset + feature * hidden + hidden_index]) << 16;
            expert += features[feature] * as_type<float>(bits);
        }
        routed += route_weights[route] * expert;
    }
    output[hidden_index] = routed;
}

kernel void full_qkv_fp8_gemm8_shared_weight_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows || shape.rows != 13568 || shape.columns != 4096 ||
        shape.block_rows != 128 || shape.block_columns != 128) {
        return;
    }
    const uint source_row = full_qkv_source_row(row);
    const uint scale_row = full_qkv_source_scale_row(source_row);
    const uint row_offset = source_row * shape.columns;
    const uint scale_columns = 32;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row * scale_columns + block];
        const uint column_base = block * 128;
        for (uint within = lane; within < 128; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < batch; ++position) {
                sums[position] += weight * input[position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] += partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

kernel void full_qkv_fp8_gemm8_sglang_blockscaled(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const uchar *input_codes [[buffer(2)]],
    device const float *input_scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant GemvShape &shape [[buffer(5)]],
    constant float *decode_lut [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows || lanes != 64 || shape.rows != 13568 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = full_qkv_source_row(row);
    const uint scale_row = full_qkv_source_scale_row(source_row);
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float totals[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        for (uint position = 0; position < batch; ++position) {
            float dot = 0.0f;
            for (uint within = lane; within < 128; within += lanes) {
                const uint column = column_base + within;
                dot += decode_lut[weights[row_offset + column]] *
                    decode_lut[input_codes[position * shape.columns + column]];
            }
            partial[lane * batch + position] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                for (uint position = 0; position < batch; ++position) {
                    partial[lane * batch + position] +=
                        partial[(lane + offset) * batch + position];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float weight_scale = weight_scales[scale_row * blocks + block];
            for (uint position = 0; position < batch; ++position) {
                totals[position] += partial[position] * weight_scale *
                    input_scales[position * blocks + block];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = totals[position];
        }
    }
}

kernel void full_qkv_fp8_gemv_sglang_blockscaled(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const uchar *input_codes [[buffer(2)]],
    device const float *input_scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant GemvShape &shape [[buffer(5)]],
    constant float *decode_lut [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 64 || shape.rows != 13568 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = full_qkv_source_row(row);
    const uint scale_row = full_qkv_source_scale_row(source_row);
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float total = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        float dot = 0.0f;
        for (uint within = lane; within < 128; within += lanes) {
            const uint column = column_base + within;
            dot += decode_lut[weights[row_offset + column]] *
                decode_lut[input_codes[column]];
        }
        partial[lane] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                partial[lane] += partial[lane + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            total += partial[0] * weight_scales[scale_row * blocks + block] *
                input_scales[block];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = total;
    }
}

kernel void full_qkv_fp8_gemv_dequantized(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 64 || shape.rows != 13568 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = full_qkv_source_row(row);
    const uint scale_row = full_qkv_source_scale_row(source_row);
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float sum = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const float scale = weight_scales[scale_row * blocks + block];
        const uint column_base = block * 128;
        for (uint within = lane; within < 128; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale * input[column];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void full_qkv_fp8_gemv_dequantized_simd(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 32 || shape.rows != 13568 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = full_qkv_source_row(row);
    const uint scale_row = full_qkv_source_scale_row(source_row);
    const uint row_offset = source_row * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < 4096; column += 32) {
        const float scale = weight_scales[scale_row * 32 + column / 128];
        sum += decode_lut[weights[row_offset + column]] * scale * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) {
        output[row] = reduced;
    }
}

kernel void swa_qkv_fp8_gemm8_sglang_blockscaled(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const uchar *input_codes [[buffer(2)]],
    device const float *input_scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant GemvShape &shape [[buffer(5)]],
    constant float *decode_lut [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows || lanes != 64 || shape.rows != 14848 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = swa_qkv_source_row(row);
    const uint scale_row = source_row / 128;
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float totals[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        for (uint position = 0; position < batch; ++position) {
            float dot = 0.0f;
            for (uint within = lane; within < 128; within += lanes) {
                const uint column = column_base + within;
                dot += decode_lut[weights[row_offset + column]] *
                    decode_lut[input_codes[position * shape.columns + column]];
            }
            partial[lane * batch + position] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                for (uint position = 0; position < batch; ++position) {
                    partial[lane * batch + position] +=
                        partial[(lane + offset) * batch + position];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float weight_scale = weight_scales[scale_row * blocks + block];
            for (uint position = 0; position < batch; ++position) {
                totals[position] += partial[position] * weight_scale *
                    input_scales[position * blocks + block];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = totals[position];
        }
    }
}

kernel void swa_qkv_fp8_gemv_sglang_blockscaled(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const uchar *input_codes [[buffer(2)]],
    device const float *input_scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant GemvShape &shape [[buffer(5)]],
    constant float *decode_lut [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 64 || shape.rows != 14848 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = swa_qkv_source_row(row);
    const uint scale_row = source_row / 128;
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float total = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        float dot = 0.0f;
        for (uint within = lane; within < 128; within += lanes) {
            const uint column = column_base + within;
            dot += decode_lut[weights[row_offset + column]] *
                decode_lut[input_codes[column]];
        }
        partial[lane] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                partial[lane] += partial[lane + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            total += partial[0] * weight_scales[scale_row * blocks + block] *
                input_scales[block];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = total;
    }
}

kernel void swa_qkv_fp8_gemv_dequantized(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 64 || shape.rows != 14848 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = swa_qkv_source_row(row);
    const uint scale_row = source_row / 128;
    const uint row_offset = source_row * shape.columns;
    constexpr uint blocks = 32;
    float sum = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const float scale = weight_scales[scale_row * blocks + block];
        const uint column_base = block * 128;
        for (uint within = lane; within < 128; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale * input[column];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void swa_qkv_fp8_gemv_dequantized_simd(
    device const uchar *weights [[buffer(0)]],
    device const float *weight_scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 32 || shape.rows != 14848 ||
        shape.columns != 4096 || shape.block_rows != 128 ||
        shape.block_columns != 128) {
        return;
    }
    const uint source_row = swa_qkv_source_row(row);
    const uint scale_row = source_row / 128;
    const uint row_offset = source_row * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < 4096; column += 32) {
        const float scale = weight_scales[scale_row * 32 + column / 128];
        sum += decode_lut[weights[row_offset + column]] * scale * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) {
        output[row] = reduced;
    }
}

kernel void block_fp8_gemv(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    float sum = 0.0f;
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    for (uint column = 0; column < shape.columns; ++column) {
        const float weight = decode_f8_e4m3fn(weights[row_offset + column]);
        const float scale = scales[scale_row_offset + column / shape.block_columns];
        sum += weight * scale * input[column];
    }
    output[row] = sum;
}

kernel void block_fp8_gemv_parallel(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const float weight = decode_f8_e4m3fn(weights[row_offset + column]);
        const float scale = scales[scale_row_offset + column / shape.block_columns];
        sum += weight * scale * input[column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void block_fp8_gemv_parallel_lut(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const float weight = decode_lut[weights[row_offset + column]];
        const float scale = scales[scale_row_offset + column / shape.block_columns];
        sum += weight * scale * input[column];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void block_fp8_gemv_parallel_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale * input[column];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

// Execute up to five independent block-FP8 projections directly from one
// mmap-backed bundle. The shared-input variant serves gate/up projections;
// the batched-input variant serves down projections after per-expert SwiGLU.
kernel void block_fp8_bundle_gemv_parallel_lut_blocked_shared_input(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    constant float *decode_lut [[buffer(4)]],
    constant BundleGemvOffsets &offsets [[buffer(5)]],
    constant uint &experts [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint expert = group / shape.rows;
    const uint row = group % shape.rows;
    if (expert >= experts || expert >= 5 || row >= shape.rows ||
        shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    device const uchar *weights = bundle + offsets.weights[expert];
    device const float *scales =
        reinterpret_cast<device const float *>(bundle + offsets.scales[expert]);
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale * input[column];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[expert * shape.rows + row] = total;
}

kernel void block_fp8_bundle_gate_up_parallel_lut_blocked_shared_input(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *gate_output [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    constant BundleGemvOffsets &gate_offsets [[buffer(6)]],
    constant BundleGemvOffsets &up_offsets [[buffer(7)]],
    constant uint &experts [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint expert = group / shape.rows;
    const uint row = group % shape.rows;
    if (expert >= experts || expert >= 5 || row >= shape.rows ||
        shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    device const uchar *gate_weights = bundle + gate_offsets.weights[expert];
    device const float *gate_scales = reinterpret_cast<device const float *>(
        bundle + gate_offsets.scales[expert]);
    device const uchar *up_weights = bundle + up_offsets.weights[expert];
    device const float *up_scales = reinterpret_cast<device const float *>(
        bundle + up_offsets.scales[expert]);
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float gate_scale = gate_scales[scale_row_offset + block];
        const float up_scale = up_scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float activation = input[column];
            gate_sum += decode_lut[gate_weights[row_offset + column]] * gate_scale * activation;
            up_sum += decode_lut[up_weights[row_offset + column]] * up_scale * activation;
        }
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (lane == 0) {
        gate_output[expert * shape.rows + row] = gate_total;
        up_output[expert * shape.rows + row] = up_total;
    }
}

kernel void block_fp8_bundle_gemv_parallel_lut_blocked_batched_input(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    constant float *decode_lut [[buffer(4)]],
    constant BundleGemvOffsets &offsets [[buffer(5)]],
    constant uint &experts [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint expert = group / shape.rows;
    const uint row = group % shape.rows;
    if (expert >= experts || expert >= 5 || row >= shape.rows ||
        shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    device const uchar *weights = bundle + offsets.weights[expert];
    device const float *scales =
        reinterpret_cast<device const float *>(bundle + offsets.scales[expert]);
    device const float *expert_input = input + expert * shape.columns;
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale * expert_input[column];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[expert * shape.rows + row] = total;
}

kernel void block_fp8_gemv_dequantized_simd(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || lanes != 32 || shape.block_rows == 0 ||
        shape.block_columns == 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += 32) {
        const float scale = scales[scale_row_offset + column / shape.block_columns];
        sum += decode_lut[weights[row_offset + column]] * scale * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) {
        output[row] = reduced;
    }
}

kernel void block_fp8_gemm8_parallel_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    const uint position = group / shape.rows;
    const uint row = group % shape.rows;
    if (position >= batch || row >= shape.rows || shape.block_rows == 0 ||
        shape.block_columns == 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint input_offset = position * shape.columns;
    const uint output_offset = position * shape.rows;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            sum += decode_lut[weights[row_offset + column]] * scale *
                input[input_offset + column];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[output_offset + row] = partial[0];
    }
}

kernel void block_fp8_gemm8_shared_weight_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < batch; ++position) {
                sums[position] += weight * input[position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] +=
                    partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

kernel void block_fp8_gemm_active_shared_weight_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    constant uint &active_count [[buffer(6)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint capacity = 8;
    if (row >= shape.rows || active_count == 0 || active_count > capacity ||
        shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sums[capacity] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < active_count; ++position) {
                sums[position] += weight * input[position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < active_count; ++position) {
        partial[lane * capacity + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < active_count; ++position) {
                partial[lane * capacity + position] +=
                    partial[(lane + offset) * capacity + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < active_count; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

template <uint active>
inline void block_fp8_gemm_specialized_impl(
    device const uchar *weights,
    device const float *scales,
    device const float *input,
    device float *output,
    constant GemvShape &shape,
    constant float *decode_lut,
    threadgroup float *partial,
    uint row,
    uint lane,
    uint lanes) {
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sums[active];
    for (uint position = 0; position < active; ++position) {
        sums[position] = 0.0f;
    }
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < active; ++position) {
                sums[position] += weight * input[position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < active; ++position) {
        partial[lane * active + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < active; ++position) {
                partial[lane * active + position] +=
                    partial[(lane + offset) * active + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < active; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

#define PW_DEFINE_SPECIALIZED_GEMM(WIDTH) \
kernel void block_fp8_gemm##WIDTH##_shared_weight_lut_blocked( \
    device const uchar *weights [[buffer(0)]], \
    device const float *scales [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output [[buffer(3)]], \
    constant GemvShape &shape [[buffer(4)]], \
    constant float *decode_lut [[buffer(5)]], \
    threadgroup float *partial [[threadgroup(0)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint lanes [[threads_per_threadgroup]]) { \
    block_fp8_gemm_specialized_impl<WIDTH>( \
        weights, scales, input, output, shape, decode_lut, partial, row, lane, lanes); \
}

PW_DEFINE_SPECIALIZED_GEMM(1)
PW_DEFINE_SPECIALIZED_GEMM(2)
PW_DEFINE_SPECIALIZED_GEMM(3)
PW_DEFINE_SPECIALIZED_GEMM(4)
PW_DEFINE_SPECIALIZED_GEMM(5)
PW_DEFINE_SPECIALIZED_GEMM(6)
PW_DEFINE_SPECIALIZED_GEMM(7)

#undef PW_DEFINE_SPECIALIZED_GEMM

template <uint active>
inline void block_fp8_gemm_blockscaled_impl(
    device const uchar *weights,
    device const float *weight_scales,
    device const uchar *input_codes,
    device const float *input_scales,
    device float *output,
    constant GemvShape &shape,
    constant float *decode_lut,
    threadgroup float *partial,
    uint row,
    uint lane,
    uint lanes) {
    if (row >= shape.rows || lanes != 64 || shape.block_rows != 128 ||
        shape.block_columns != 128 || shape.columns % 128 != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint blocks = shape.columns / 128;
    const uint weight_scale_offset = (row / 128) * blocks;
    float totals[active];
    for (uint position = 0; position < active; ++position) {
        totals[position] = 0.0f;
    }
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        for (uint position = 0; position < active; ++position) {
            float dot = 0.0f;
            for (uint within = lane; within < 128; within += lanes) {
                const uint column = column_base + within;
                dot += decode_lut[weights[row_offset + column]] *
                    decode_lut[input_codes[position * shape.columns + column]];
            }
            partial[lane * active + position] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                for (uint position = 0; position < active; ++position) {
                    partial[lane * active + position] +=
                        partial[(lane + offset) * active + position];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float weight_scale = weight_scales[weight_scale_offset + block];
            for (uint position = 0; position < active; ++position) {
                const float input_scale = input_scales[position * blocks + block];
                totals[position] += partial[position] * weight_scale * input_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < active; ++position) {
            output[position * shape.rows + row] = totals[position];
        }
    }
}

#define PW_DEFINE_BLOCKSCALED_GEMM(WIDTH) \
kernel void block_fp8_gemm##WIDTH##_sglang_blockscaled( \
    device const uchar *weights [[buffer(0)]], \
    device const float *weight_scales [[buffer(1)]], \
    device const uchar *input_codes [[buffer(2)]], \
    device const float *input_scales [[buffer(3)]], \
    device float *output [[buffer(4)]], \
    constant GemvShape &shape [[buffer(5)]], \
    constant float *decode_lut [[buffer(6)]], \
    threadgroup float *partial [[threadgroup(0)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint lanes [[threads_per_threadgroup]]) { \
    block_fp8_gemm_blockscaled_impl<WIDTH>(weights, weight_scales, input_codes, \
        input_scales, output, shape, decode_lut, partial, row, lane, lanes); \
}

PW_DEFINE_BLOCKSCALED_GEMM(1)
PW_DEFINE_BLOCKSCALED_GEMM(2)
PW_DEFINE_BLOCKSCALED_GEMM(3)
PW_DEFINE_BLOCKSCALED_GEMM(4)
PW_DEFINE_BLOCKSCALED_GEMM(5)
PW_DEFINE_BLOCKSCALED_GEMM(6)
PW_DEFINE_BLOCKSCALED_GEMM(7)
PW_DEFINE_BLOCKSCALED_GEMM(8)
PW_DEFINE_BLOCKSCALED_GEMM(9)
PW_DEFINE_BLOCKSCALED_GEMM(10)
PW_DEFINE_BLOCKSCALED_GEMM(11)
PW_DEFINE_BLOCKSCALED_GEMM(12)
PW_DEFINE_BLOCKSCALED_GEMM(13)
PW_DEFINE_BLOCKSCALED_GEMM(14)
PW_DEFINE_BLOCKSCALED_GEMM(15)
PW_DEFINE_BLOCKSCALED_GEMM(16)
PW_DEFINE_BLOCKSCALED_GEMM(17)
PW_DEFINE_BLOCKSCALED_GEMM(18)
PW_DEFINE_BLOCKSCALED_GEMM(19)
PW_DEFINE_BLOCKSCALED_GEMM(20)
PW_DEFINE_BLOCKSCALED_GEMM(21)
PW_DEFINE_BLOCKSCALED_GEMM(22)
PW_DEFINE_BLOCKSCALED_GEMM(23)
PW_DEFINE_BLOCKSCALED_GEMM(24)
PW_DEFINE_BLOCKSCALED_GEMM(25)
PW_DEFINE_BLOCKSCALED_GEMM(26)
PW_DEFINE_BLOCKSCALED_GEMM(27)
PW_DEFINE_BLOCKSCALED_GEMM(28)
PW_DEFINE_BLOCKSCALED_GEMM(29)
PW_DEFINE_BLOCKSCALED_GEMM(30)
PW_DEFINE_BLOCKSCALED_GEMM(31)
PW_DEFINE_BLOCKSCALED_GEMM(32)

#undef PW_DEFINE_BLOCKSCALED_GEMM

template <uint active>
inline void fused_block_fp8_gate_up_swiglu_impl(
    device const uchar *gate_weights,
    device const float *gate_weight_scales,
    device const uchar *up_weights,
    device const float *up_weight_scales,
    device const uchar *input_codes,
    device const float *input_scales,
    device float *hidden,
    constant GemvShape &shape,
    constant float *decode_lut,
    device atomic_uint *error_flags,
    threadgroup float *partial,
    uint row,
    uint lane,
    uint lanes) {
    if (row >= shape.rows || lanes != 64 || shape.block_rows != 128 ||
        shape.block_columns != 128 || shape.columns % 128 != 0) {
        return;
    }
    const uint row_offset = row * shape.columns;
    const uint blocks = shape.columns / 128;
    const uint weight_scale_offset = (row / 128) * blocks;
    float gate_totals[active];
    float up_totals[active];
    for (uint position = 0; position < active; ++position) {
        gate_totals[position] = 0.0f;
        up_totals[position] = 0.0f;
    }
    for (uint block = 0; block < blocks; ++block) {
        const uint column_base = block * 128;
        for (uint position = 0; position < active; ++position) {
            float gate_dot = 0.0f;
            float up_dot = 0.0f;
            for (uint within = lane; within < 128; within += lanes) {
                const uint column = column_base + within;
                const float activation =
                    decode_lut[input_codes[position * shape.columns + column]];
                gate_dot += decode_lut[gate_weights[row_offset + column]] * activation;
                up_dot += decode_lut[up_weights[row_offset + column]] * activation;
            }
            const uint partial_base = (lane * active + position) * 2;
            partial[partial_base] = gate_dot;
            partial[partial_base + 1] = up_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = lanes / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                for (uint position = 0; position < active; ++position) {
                    const uint destination = (lane * active + position) * 2;
                    const uint source = ((lane + offset) * active + position) * 2;
                    partial[destination] += partial[source];
                    partial[destination + 1] += partial[source + 1];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (uint position = 0; position < active; ++position) {
                const uint reduced = position * 2;
                const float input_scale = input_scales[position * blocks + block];
                gate_totals[position] += partial[reduced] *
                    gate_weight_scales[weight_scale_offset + block] * input_scale;
                up_totals[position] += partial[reduced + 1] *
                    up_weight_scales[weight_scale_offset + block] * input_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < active; ++position) {
            const float rounded_gate = pw_round_bf16(gate_totals[position]);
            const float rounded_up = pw_round_bf16(up_totals[position]);
            const float silu = pw_round_bf16(
                rounded_gate / (1.0f + exp(-rounded_gate)));
            const float result = pw_round_bf16(silu * rounded_up);
            if (!isfinite(result)) {
                atomic_fetch_or_explicit(error_flags, 128u, memory_order_relaxed);
            }
            hidden[position * shape.rows + row] = result;
        }
    }
}

#define PW_DEFINE_FUSED_GATE_UP_SWIGLU(WIDTH) \
kernel void fused_block_fp8_gate_up_swiglu##WIDTH( \
    device const uchar *gate_weights [[buffer(0)]], \
    device const float *gate_weight_scales [[buffer(1)]], \
    device const uchar *up_weights [[buffer(2)]], \
    device const float *up_weight_scales [[buffer(3)]], \
    device const uchar *input_codes [[buffer(4)]], \
    device const float *input_scales [[buffer(5)]], \
    device float *hidden [[buffer(6)]], \
    constant GemvShape &shape [[buffer(7)]], \
    constant float *decode_lut [[buffer(8)]], \
    device atomic_uint *error_flags [[buffer(9)]], \
    threadgroup float *partial [[threadgroup(0)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], \
    uint lanes [[threads_per_threadgroup]]) { \
    fused_block_fp8_gate_up_swiglu_impl<WIDTH>(gate_weights, gate_weight_scales, \
        up_weights, up_weight_scales, input_codes, input_scales, hidden, shape, \
        decode_lut, error_flags, partial, row, lane, lanes); \
}

PW_DEFINE_FUSED_GATE_UP_SWIGLU(1)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(2)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(3)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(4)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(5)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(6)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(7)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(8)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(9)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(10)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(11)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(12)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(13)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(14)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(15)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(16)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(17)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(18)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(19)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(20)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(21)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(22)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(23)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(24)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(25)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(26)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(27)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(28)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(29)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(30)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(31)
PW_DEFINE_FUSED_GATE_UP_SWIGLU(32)

#undef PW_DEFINE_FUSED_GATE_UP_SWIGLU

kernel void block_fp8_gemm8_simdgroup_matrix_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *weight_tile [[threadgroup(0)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row_base = output_tile * 8;
    if (row_base + 8 > shape.rows || shape.block_rows == 0 ||
        shape.block_columns == 0 || shape.columns % 8 != 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint scale_columns = shape.columns / shape.block_columns;
    simdgroup_float8x8 accumulator(0.0f);
    for (uint column_base = 0; column_base < shape.columns; column_base += 8) {
        for (uint index = lane; index < 64; index += 32) {
            const uint inner = index / 8;
            const uint output_column = index % 8;
            const uint row = row_base + output_column;
            const uint column = column_base + inner;
            const uint scale_index = (row / shape.block_rows) * scale_columns +
                column / shape.block_columns;
            weight_tile[index] =
                decode_lut[weights[row * shape.columns + column]] * scales[scale_index];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 activation_matrix;
        simdgroup_float8x8 weight_matrix;
        simdgroup_load(activation_matrix, input + column_base, shape.columns);
        simdgroup_load(weight_matrix, weight_tile, 8);
        simdgroup_multiply_accumulate(
            accumulator, activation_matrix, weight_matrix, accumulator);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accumulator, output + row_base, shape.rows);
}

kernel void block_fp8_gemm8_fused_gate_up_lut_blocked(
    device const uchar *gate_weights [[buffer(0)]],
    device const float *gate_scales [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const float *up_scales [[buffer(3)]],
    device const float *input [[buffer(4)]],
    device float *gate_output [[buffer(5)]],
    device float *up_output [[buffer(6)]],
    constant GemvShape &shape [[buffer(7)]],
    constant float *decode_lut [[buffer(8)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint fused_row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    const bool is_up = fused_row >= shape.rows;
    const uint row = is_up ? fused_row - shape.rows : fused_row;
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    device const uchar *weights = is_up ? up_weights : gate_weights;
    device const float *scales = is_up ? up_scales : gate_scales;
    device float *output = is_up ? up_output : gate_output;
    const uint row_offset = row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = (row / shape.block_rows) * scale_columns;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < batch; ++position) {
                sums[position] += weight * input[position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] +=
                    partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[position * shape.rows + row] = partial[position];
        }
    }
}

kernel void block_fp8_expert_union_gemm8_shared_weight_lut_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    constant float *decode_lut [[buffer(5)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    const uint row = group.x;
    const uint expert = group.y;
    const uint lane = thread_index.x;
    const uint lanes = threads.x;
    if (row >= shape.rows || shape.block_rows == 0 || shape.block_columns == 0 ||
        shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint weight_expert_stride = shape.rows * shape.columns;
    const uint row_offset = expert * weight_expert_stride + row * shape.columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_rows = (shape.rows + shape.block_rows - 1) / shape.block_rows;
    const uint scale_expert_stride = scale_rows * scale_columns;
    const uint scale_row_offset = expert * scale_expert_stride +
        (row / shape.block_rows) * scale_columns;
    const uint input_expert_offset = expert * batch * shape.columns;
    const uint output_expert_offset = expert * batch * shape.rows;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint column_base = block * shape.block_columns;
        for (uint within = lane; within < shape.block_columns; within += lanes) {
            const uint column = column_base + within;
            const float weight = decode_lut[weights[row_offset + column]] * scale;
            for (uint position = 0; position < batch; ++position) {
                sums[position] += weight *
                    input[input_expert_offset + position * shape.columns + column];
            }
        }
    }
    for (uint position = 0; position < batch; ++position) {
        partial[lane * batch + position] = sums[position];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint position = 0; position < batch; ++position) {
                partial[lane * batch + position] +=
                    partial[(lane + offset) * batch + position];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint position = 0; position < batch; ++position) {
            output[output_expert_offset + position * shape.rows + row] = partial[position];
        }
    }
}

inline float decode_signed_int4(uchar nibble) {
    const uchar value = nibble & 0x0f;
    return value < 8 ? float(value) : float(value) - 16.0f;
}

kernel void group_int4_gemv_parallel_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || shape.block_columns == 0 ||
        shape.block_columns % 2 != 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint packed_columns = shape.columns / 2;
    const uint packed_block_columns = shape.block_columns / 2;
    const uint row_offset = row * packed_columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = row * scale_columns;
    float sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint packed_base = block * packed_block_columns;
        for (uint within = lane; within < packed_block_columns; within += lanes) {
            const uchar bits = weights[row_offset + packed_base + within];
            const uint column = block * shape.block_columns + within * 2;
            sum += decode_signed_int4(bits) * scale * input[column];
            sum += decode_signed_int4(bits >> 4) * scale * input[column + 1];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[row] = partial[0];
    }
}

kernel void group_int4_gemm8_parallel_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    constexpr uint batch = 8;
    if (row >= shape.rows || shape.block_columns == 0 ||
        shape.block_columns % 2 != 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint packed_columns = shape.columns / 2;
    const uint packed_block_columns = shape.block_columns / 2;
    const uint row_offset = row * packed_columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = row * scale_columns;
    float sums[batch] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint packed_base = block * packed_block_columns;
        for (uint within = lane; within < packed_block_columns; within += lanes) {
            const uchar bits = weights[row_offset + packed_base + within];
            const float low = decode_signed_int4(bits) * scale;
            const float high = decode_signed_int4(bits >> 4) * scale;
            const uint column = block * shape.block_columns + within * 2;
            for (uint item = 0; item < batch; ++item) {
                const uint input_offset = item * shape.columns + column;
                sums[item] += low * input[input_offset];
                sums[item] += high * input[input_offset + 1];
            }
        }
    }
    for (uint item = 0; item < batch; ++item) {
        partial[item * lanes + lane] = sums[item];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (uint item = 0; item < batch; ++item) {
                partial[item * lanes + lane] += partial[item * lanes + lane + offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint item = 0; item < batch; ++item) {
            output[item * shape.rows + row] = partial[item * lanes];
        }
    }
}

kernel void group_int4_gemm8_vector_parallel_blocked(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    threadgroup float4 *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows || shape.block_columns == 0 ||
        shape.block_columns % 2 != 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    const uint packed_columns = shape.columns / 2;
    const uint packed_block_columns = shape.block_columns / 2;
    const uint row_offset = row * packed_columns;
    const uint scale_columns = shape.columns / shape.block_columns;
    const uint scale_row_offset = row * scale_columns;
    float4 low_batch_sum = 0.0f;
    float4 high_batch_sum = 0.0f;
    for (uint block = 0; block < scale_columns; ++block) {
        const float scale = scales[scale_row_offset + block];
        const uint packed_base = block * packed_block_columns;
        for (uint within = lane; within < packed_block_columns; within += lanes) {
            const uchar bits = weights[row_offset + packed_base + within];
            const float low = decode_signed_int4(bits) * scale;
            const float high = decode_signed_int4(bits >> 4) * scale;
            const uint column = block * shape.block_columns + within * 2;
            low_batch_sum += low * input[column * 2];
            high_batch_sum += low * input[column * 2 + 1];
            low_batch_sum += high * input[(column + 1) * 2];
            high_batch_sum += high * input[(column + 1) * 2 + 1];
        }
    }
    partial[lane] = low_batch_sum;
    partial[lanes + lane] = high_batch_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
            partial[lanes + lane] += partial[lanes + lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float4 first = partial[0];
        const float4 second = partial[lanes];
        output[row] = first.x;
        output[shape.rows + row] = first.y;
        output[shape.rows * 2 + row] = first.z;
        output[shape.rows * 3 + row] = first.w;
        output[shape.rows * 4 + row] = second.x;
        output[shape.rows * 5 + row] = second.y;
        output[shape.rows * 6 + row] = second.z;
        output[shape.rows * 7 + row] = second.w;
    }
}
