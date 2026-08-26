#include <metal_stdlib>
using namespace metal;

constant uint K4_BUNDLE_EXPERTS = 5u;

struct QtipK4BundleShape {
    uint rows;
    uint columns;
    uint tile_columns;
    uint rank;
};

struct QtipK4BundleOffsets {
    uint packed[5];
    uint left_sign[5];
    uint right_sign[5];
    uint global_scale[5];
    uint row_scale[5];
    uint correction_left[5];
    uint correction_right[5];
};

inline ushort qtip_k4_bundle_state(device const ushort *packed, uint state_index) {
    const uint bit_offset = state_index * 8u;
    const uint word_index = bit_offset >> 4;
    const uint word_shift = bit_offset & 15u;
    const uint joined = (uint(packed[word_index]) << 16) |
                        uint(packed[(word_index + 1u) & 63u]);
    return ushort((joined >> (16u - word_shift)) & 0xffffu);
}

inline uint qtip_k4_bundle_permuted_index(uint original_index) {
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

inline float qtip_k4_bundle_value(
    device const ushort *packed,
    constant const float *tlut,
    uint original_index) {
    const uint permuted = qtip_k4_bundle_permuted_index(original_index);
    const uint state = uint(qtip_k4_bundle_state(packed, permuted >> 1));
    const uint mixed = state * (state + 1u);
    const uint tlut_index = (mixed >> 6) & 511u;
    const uint component = permuted & 1u;
    float value = tlut[tlut_index * 2u + component];
    if (component == 0u && ((mixed >> 15) & 1u) != 0u) value = -value;
    return value;
}

inline void qtip_k4_bundle_fwht(
    device const float *input,
    device const char *signs,
    device float *output,
    uint count,
    threadgroup float *scratch,
    uint lane,
    uint lanes) {
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

kernel void qtip_k4_bundle_fwht_signed_shared(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant QtipK4BundleOffsets &offsets [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (expert >= K4_BUNDLE_EXPERTS) return;
    device const char *signs = (device const char *)(bundle + offsets.right_sign[expert]);
    qtip_k4_bundle_fwht(input, signs, output + expert * count, count, scratch, lane, lanes);
}

kernel void qtip_k4_bundle_fwht_signed_inputs(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant QtipK4BundleOffsets &offsets [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (expert >= K4_BUNDLE_EXPERTS) return;
    device const char *signs = (device const char *)(bundle + offsets.right_sign[expert]);
    qtip_k4_bundle_fwht(input + expert * count, signs, output + expert * count,
                        count, scratch, lane, lanes);
}

kernel void qtip_k4_bundle_projection(
    device const uchar *bundle [[buffer(0)]],
    constant const float *tlut [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipK4BundleShape &shape [[buffer(4)]],
    constant QtipK4BundleOffsets &offsets [[buffer(5)]],
    threadgroup float *partials [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]]) {
    const uint row = group.x;
    const uint expert = group.y;
    const uint lane = thread_position.x;
    const uint lanes = threadgroup_size.x;
    if (row >= shape.rows || expert >= K4_BUNDLE_EXPERTS) return;
    device const ushort *packed =
        (device const ushort *)(bundle + offsets.packed[expert]);
    device const float *expert_input = input + expert * shape.columns;
    const uint tile_row = row >> 4;
    const uint local_row = row & 15u;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += lanes) {
        const uint tile_column = column >> 4;
        const uint local_column = column & 15u;
        const uint tile_index = tile_row * shape.tile_columns + tile_column;
        sum += qtip_k4_bundle_value(
            packed + tile_index * 64u, tlut, local_row * 16u + local_column) *
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

kernel void qtip_k4_bundle_low_rank_shared(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipK4BundleShape &shape [[buffer(3)]],
    constant QtipK4BundleOffsets &offsets [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank || expert >= K4_BUNDLE_EXPERTS) return;
    device const half *right =
        (device const half *)(bundle + offsets.correction_right[expert]);
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(right[rank_index * shape.columns + column]) * input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_k4_bundle_low_rank_inputs(
    device const uchar *bundle [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant QtipK4BundleShape &shape [[buffer(3)]],
    constant QtipK4BundleOffsets &offsets [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint rank_index = position.x;
    const uint expert = position.y;
    if (rank_index >= shape.rank || expert >= K4_BUNDLE_EXPERTS) return;
    device const half *right =
        (device const half *)(bundle + offsets.correction_right[expert]);
    device const float *expert_input = input + expert * shape.columns;
    float sum = 0.0f;
    for (uint column = 0u; column < shape.columns; ++column) {
        sum += float(right[rank_index * shape.columns + column]) * expert_input[column];
    }
    output[expert * shape.rank + rank_index] = sum;
}

kernel void qtip_k4_bundle_finish(
    device const uchar *bundle [[buffer(0)]],
    device const float *base [[buffer(1)]],
    device const float *low_rank_right_output [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant QtipK4BundleShape &shape [[buffer(4)]],
    constant QtipK4BundleOffsets &offsets [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint row = position.x;
    const uint expert = position.y;
    if (row >= shape.rows || expert >= K4_BUNDLE_EXPERTS) return;
    device const char *left_sign =
        (device const char *)(bundle + offsets.left_sign[expert]);
    device const float *global_scale =
        (device const float *)(bundle + offsets.global_scale[expert]);
    device const half *row_scale =
        (device const half *)(bundle + offsets.row_scale[expert]);
    device const half *left =
        (device const half *)(bundle + offsets.correction_left[expert]);
    float correction = 0.0f;
    for (uint index = 0u; index < shape.rank; ++index) {
        correction += float(left[row * shape.rank + index]) *
                      low_rank_right_output[expert * shape.rank + index];
    }
    output[expert * shape.rows + row] =
        base[expert * shape.rows + row] * float(left_sign[row]) *
        global_scale[0] * float(row_scale[row]) + correction;
}
