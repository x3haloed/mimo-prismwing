#include <metal_stdlib>
using namespace metal;

struct QtipArenaProjectionLayout {
    uint slot_stride_bytes;
    uint packed_offset_bytes;
    uint left_sign_offset_bytes;
    uint right_sign_offset_bytes;
    uint global_scale_offset_bytes;
    uint row_scale_offset_bytes;
    uint correction_left_offset_bytes;
    uint correction_right_offset_bytes;
};

kernel void dynamic_fp8_dequantized_group128_binary(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float *decode_lut [[buffer(2)]],
    device atomic_uint *error_flags [[buffer(3)]],
    constant float *boundaries [[buffer(4)]],
    threadgroup float *maximums [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (lanes != 128u) {
        if (lane == 0u) atomic_fetch_or_explicit(error_flags, 1u, memory_order_relaxed);
        return;
    }
    const uint index = group * 128u + lane;
    const float value = input[index];
    maximums[lane] = isfinite(value) ? abs(value) : 0.0f;
    if (!isfinite(value)) atomic_fetch_or_explicit(error_flags, 2u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 64u; offset > 0u; offset >>= 1u) {
        if (lane < offset) maximums[lane] = max(maximums[lane], maximums[lane + offset]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = max(maximums[0], 1.0e-10f) / 448.0f;
    const float normalized = clamp(value / scale, -448.0f, 448.0f);
    if (normalized == 0.0f) {
        output[index] = signbit(normalized) ? -0.0f : 0.0f;
        return;
    }
    const float magnitude = abs(normalized);
    uint low = 0u, high = 126u;
    while (low < high) {
        const uint middle = (low + high) >> 1u;
        if (boundaries[middle] < magnitude) low = middle + 1u;
        else high = middle;
    }
    uint code = low;
    if (code < 126u && magnitude == boundaries[code] && (code & 1u) != 0u) {
        code += 1u;
    }
    code |= signbit(normalized) ? 0x80u : 0u;
    output[index] = decode_lut[code] * scale;
}

kernel void qtip_fwht_signed_fused(
    device const float *input [[buffer(0)]],
    device const char *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[index] * float(signs[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_signed_fused_batched(
    device const float *input [[buffer(0)]],
    device const char *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint base = expert * count;
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[index] * float(signs[base + index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[base + index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_signed_fused_batched_inputs(
    device const float *input [[buffer(0)]],
    device const char *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint base = expert * count;
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[base + index] * float(signs[base + index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[base + index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_signed_fused_arena_batched(
    device const uchar *arena [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant QtipArenaProjectionLayout &layout [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint base = expert * count;
    device const char *signs =
        (device const char *)(arena + expert * layout.slot_stride_bytes +
                              layout.right_sign_offset_bytes);
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[index] * float(signs[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[base + index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_signed_fused_arena_batched_inputs(
    device const uchar *arena [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant QtipArenaProjectionLayout &layout [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint base = expert * count;
    device const char *signs =
        (device const char *)(arena + expert * layout.slot_stride_bytes +
                              layout.right_sign_offset_bytes);
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[base + index] * float(signs[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[base + index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_fused(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[index] = scratch[index] * normalization;
    }
}

kernel void qtip_fwht_fused_batched(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint base = expert * count;
    for (uint index = lane; index < count; index += lanes) {
        scratch[index] = input[base + index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < count; stride <<= 1u) {
        for (uint pair = lane; pair < count / 2u; pair += lanes) {
            const uint block = pair / stride;
            const uint offset = pair % stride;
            const uint left_index = block * stride * 2u + offset;
            const uint right_index = left_index + stride;
            const float left = scratch[left_index];
            const float right = scratch[right_index];
            scratch[left_index] = left + right;
            scratch[right_index] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float normalization = rsqrt(float(count));
    for (uint index = lane; index < count; index += lanes) {
        output[base + index] = scratch[index] * normalization;
    }
}

inline ushort qtip_parallel_trellis_state(
    device const ushort *packed,
    uint state_index) {
    const uint bit_offset = state_index * 4u;
    const uint word_index = bit_offset >> 4;
    const uint word_shift = bit_offset & 15u;
    const uint joined = (uint(packed[word_index]) << 16) |
                        uint(packed[(word_index + 1u) & 31u]);
    return ushort((joined >> (16u - word_shift)) & 0xffffu);
}

inline uint qtip_parallel_permuted_index(uint original_index) {
    uint value = original_index;
    const uint e = value & 1u;
    value >>= 1;
    const uint d = value & 3u;
    value >>= 2;
    const uint c = value & 1u;
    value >>= 1;
    const uint b = value & 7u;
    value >>= 3;
    const uint a = value;
    return ((((b * 4u + d) * 2u + c) * 2u + a) * 2u + e);
}

inline float qtip_parallel_value(
    device const ushort *packed,
    constant const float *tlut,
    uint original_index) {
    const uint permuted = qtip_parallel_permuted_index(original_index);
    const uint state = uint(qtip_parallel_trellis_state(packed, permuted >> 1));
    const uint mixed = state * (state + 1u);
    const uint tlut_index = (mixed >> 6) & 511u;
    const uint component = permuted & 1u;
    float value = tlut[tlut_index * 2u + component];
    if (component == 0u && ((mixed >> 15) & 1u) != 0u) {
        value = -value;
    }
    return value;
}

struct QtipProjectionShape {
    uint rows;
    uint columns;
    uint tile_columns;
    uint rank;
};

kernel void qtip_trellis_projection_gemv_parallel(
    device const ushort *packed [[buffer(0)]],
    constant const float *tlut [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipProjectionShape &shape [[buffer(4)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= shape.rows) {
        return;
    }
    const uint tile_row = row >> 4;
    const uint local_row = row & 15u;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const uint tile_column = column >> 4;
        const uint local_column = column & 15u;
        const uint tile_index = tile_row * shape.tile_columns + tile_column;
        device const ushort *tile = packed + tile_index * 32u;
        sum += qtip_parallel_value(
                   tile, tlut, local_row * 16u + local_column) * input[column];
    }
    partials[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes >> 1; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            partials[lane] += partials[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        output[row] = partials[0];
    }
}

kernel void qtip_trellis_projection_gemv_parallel_batched(
    device const ushort *packed [[buffer(0)]],
    constant const float *tlut [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipProjectionShape &shape [[buffer(4)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]) {
    const uint row = group.x;
    const uint expert = group.y;
    const uint lane = thread_position.x;
    const uint lanes = threadgroup_size.x;
    if (row >= shape.rows) {
        return;
    }
    const uint tile_row = row >> 4;
    const uint local_row = row & 15u;
    const uint packed_expert_stride = shape.rows * shape.columns / 8u;
    device const ushort *expert_packed = packed + expert * packed_expert_stride;
    device const float *expert_input = input + expert * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const uint tile_column = column >> 4;
        const uint local_column = column & 15u;
        const uint tile_index = tile_row * shape.tile_columns + tile_column;
        device const ushort *tile = expert_packed + tile_index * 32u;
        sum += qtip_parallel_value(
                   tile, tlut, local_row * 16u + local_column) *
               expert_input[column];
    }
    partials[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes >> 1; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            partials[lane] += partials[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        output[expert * shape.rows + row] = partials[0];
    }
}

kernel void qtip_trellis_projection_gemv_parallel_arena_batched(
    device const uchar *arena [[buffer(0)]],
    constant const float *tlut [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipProjectionShape &shape [[buffer(4)]],
    constant QtipArenaProjectionLayout &layout [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]) {
    const uint row = group.x;
    const uint expert = group.y;
    const uint lane = thread_position.x;
    const uint lanes = threadgroup_size.x;
    if (row >= shape.rows) return;
    const uint tile_row = row >> 4;
    const uint local_row = row & 15u;
    device const ushort *expert_packed = (device const ushort *)(
        arena + expert * layout.slot_stride_bytes + layout.packed_offset_bytes);
    device const float *expert_input = input + expert * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const uint tile_column = column >> 4;
        const uint local_column = column & 15u;
        const uint tile_index = tile_row * shape.tile_columns + tile_column;
        device const ushort *tile = expert_packed + tile_index * 32u;
        sum += qtip_parallel_value(tile, tlut, local_row * 16u + local_column) *
               expert_input[column];
    }
    partials[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes >> 1; offset > 0u; offset >>= 1u) {
        if (lane < offset) partials[lane] += partials[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) output[expert * shape.rows + row] = partials[0];
}

kernel void qtip_low_rank_right_gemv_batched(
    device const half *right [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipProjectionShape &shape [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank) {
        return;
    }
    device const half *expert_right =
        right + expert * shape.rank * shape.columns;
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(expert_right[rank_index * shape.columns + column]) *
               input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_low_rank_right_gemv_batched_inputs(
    device const half *right [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipProjectionShape &shape [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank) {
        return;
    }
    device const half *expert_right =
        right + expert * shape.rank * shape.columns;
    device const float *expert_input = input + expert * shape.columns;
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(expert_right[rank_index * shape.columns + column]) *
               expert_input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_low_rank_right_gemv_arena_batched(
    device const uchar *arena [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipProjectionShape &shape [[buffer(3)]],
    constant QtipArenaProjectionLayout &layout [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank) return;
    device const half *right = (device const half *)(
        arena + expert * layout.slot_stride_bytes +
        layout.correction_right_offset_bytes);
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(right[rank_index * shape.columns + column]) * input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_low_rank_right_gemv_arena_batched_inputs(
    device const uchar *arena [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipProjectionShape &shape [[buffer(3)]],
    constant QtipArenaProjectionLayout &layout [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank) return;
    device const half *right = (device const half *)(
        arena + expert * layout.slot_stride_bytes +
        layout.correction_right_offset_bytes);
    device const float *expert_input = input + expert * shape.columns;
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(right[rank_index * shape.columns + column]) *
               expert_input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_corrected_projection_finish_batched(
    device const float *base [[buffer(0)]],
    device const char *left_sign [[buffer(1)]],
    device const half *row_scale [[buffer(2)]],
    device const half *low_rank_left [[buffer(3)]],
    device const float *low_rank_right_output [[buffer(4)]],
    device float *output [[buffer(5)]],
    device const float *global_scale [[buffer(6)]],
    constant QtipProjectionShape &shape [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint row = position.x;
    const uint expert = position.y;
    if (row >= shape.rows) {
        return;
    }
    const uint row_base = expert * shape.rows;
    const uint left_base = expert * shape.rows * shape.rank;
    const uint rank_base = expert * shape.rank;
    float correction = 0.0f;
    for (uint index = 0u; index < shape.rank; ++index) {
        correction += float(low_rank_left[left_base + row * shape.rank + index]) *
                      low_rank_right_output[rank_base + index];
    }
    output[row_base + row] =
        base[row_base + row] * float(left_sign[row_base + row]) *
        global_scale[expert] * float(row_scale[row_base + row]) + correction;
}

kernel void qtip_corrected_projection_finish_arena_batched(
    device const uchar *arena [[buffer(0)]],
    device const float *base [[buffer(1)]],
    device const float *low_rank_right_output [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipProjectionShape &shape [[buffer(4)]],
    constant QtipArenaProjectionLayout &layout [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint row = position.x;
    const uint expert = position.y;
    if (row >= shape.rows) return;
    device const uchar *slot = arena + expert * layout.slot_stride_bytes;
    device const char *left_sign =
        (device const char *)(slot + layout.left_sign_offset_bytes);
    device const float *global_scale =
        (device const float *)(slot + layout.global_scale_offset_bytes);
    device const half *row_scale =
        (device const half *)(slot + layout.row_scale_offset_bytes);
    device const half *low_rank_left =
        (device const half *)(slot + layout.correction_left_offset_bytes);
    const uint row_base = expert * shape.rows;
    const uint rank_base = expert * shape.rank;
    float correction = 0.0f;
    for (uint index = 0u; index < shape.rank; ++index) {
        correction += float(low_rank_left[row * shape.rank + index]) *
                      low_rank_right_output[rank_base + index];
    }
    output[row_base + row] =
        base[row_base + row] * float(left_sign[row]) * global_scale[0] *
        float(row_scale[row]) + correction;
}

kernel void qtip_trellis_projection_gemv_kahan(
    device const ushort *packed [[buffer(0)]],
    constant const float *tlut [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipProjectionShape &shape [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= shape.rows) {
        return;
    }
    const uint tile_row = row >> 4;
    const uint local_row = row & 15u;
    float sum = 0.0f;
    float compensation = 0.0f;
    for (uint column = 0; column < shape.columns; ++column) {
        const uint tile_column = column >> 4;
        const uint local_column = column & 15u;
        device const ushort *tile = packed +
            (tile_row * shape.tile_columns + tile_column) * 32u;
        const float product = qtip_parallel_value(
            tile, tlut, local_row * 16u + local_column) * input[column];
        const float corrected = product - compensation;
        const float next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }
    output[row] = sum;
}
