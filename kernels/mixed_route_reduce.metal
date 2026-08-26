#include <metal_stdlib>
using namespace metal;

inline float mixed_bf16(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    const uint bias = 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + bias) & 0xffff0000u);
}

kernel void mixed_route_weighted_reduce_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_output [[buffer(1)]],
    constant const float *route_weights [[buffer(2)]],
    device float *routed [[buffer(3)]],
    device atomic_uint *error_flags [[buffer(4)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    float sum = 0.0f;
    for (uint expert = 0u; expert < 7u; ++expert) {
        sum += mixed_bf16(k4_outputs[expert * 4096u + column]) * route_weights[expert];
    }
    sum += mixed_bf16(source_output[column]) * route_weights[7];
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}

kernel void mixed_route_weighted_reduce_dynamic_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_output [[buffer(1)]],
    device const uint *selected_experts [[buffer(2)]],
    device const float *selected_weights [[buffer(3)]],
    constant const uint *k4_experts [[buffer(4)]],
    constant const uint &source_expert [[buffer(5)]],
    device float *routed [[buffer(6)]],
    device atomic_uint *error_flags [[buffer(7)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    float sum = 0.0f;
    for (uint route = 0u; route < 8u; ++route) {
        const uint expert = selected_experts[route];
        float value = 0.0f;
        bool found = false;
        if (expert == source_expert) {
            value = mixed_bf16(source_output[column]);
            found = true;
        } else {
            for (uint slot = 0u; slot < 7u; ++slot) {
                if (expert == k4_experts[slot]) {
                    value = mixed_bf16(k4_outputs[slot * 4096u + column]);
                    found = true;
                    break;
                }
            }
        }
        if (!found) atomic_fetch_or_explicit(error_flags, 1024u, memory_order_relaxed);
        sum += value * selected_weights[route];
    }
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}

kernel void mixed_route_weighted_reduce_dynamic_two_source_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_output_0 [[buffer(1)]],
    device const float *source_output_1 [[buffer(2)]],
    device const uint *selected_experts [[buffer(3)]],
    device const float *selected_weights [[buffer(4)]],
    constant const uint *k4_experts [[buffer(5)]],
    constant const uint *source_experts [[buffer(6)]],
    device float *routed [[buffer(7)]],
    device atomic_uint *error_flags [[buffer(8)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    float sum = 0.0f;
    for (uint route = 0u; route < 8u; ++route) {
        const uint expert = selected_experts[route];
        float value = 0.0f;
        bool found = false;
        if (expert == source_experts[0]) {
            value = mixed_bf16(source_output_0[column]);
            found = true;
        } else if (expert == source_experts[1]) {
            value = mixed_bf16(source_output_1[column]);
            found = true;
        } else {
            for (uint slot = 0u; slot < 6u; ++slot) {
                if (expert == k4_experts[slot]) {
                    value = mixed_bf16(k4_outputs[slot * 4096u + column]);
                    found = true;
                    break;
                }
            }
        }
        if (!found) atomic_fetch_or_explicit(error_flags, 1024u, memory_order_relaxed);
        sum += value * selected_weights[route];
    }
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}

kernel void mixed_route_weighted_reduce_dynamic_three_source_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_output_0 [[buffer(1)]],
    device const float *source_output_1 [[buffer(2)]],
    device const float *source_output_2 [[buffer(3)]],
    device const uint *selected_experts [[buffer(4)]],
    device const float *selected_weights [[buffer(5)]],
    constant const uint *k4_experts [[buffer(6)]],
    constant const uint *source_experts [[buffer(7)]],
    device float *routed [[buffer(8)]],
    device atomic_uint *error_flags [[buffer(9)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    float sum = 0.0f;
    for (uint route = 0u; route < 8u; ++route) {
        const uint expert = selected_experts[route];
        float value = 0.0f;
        bool found = false;
        if (expert == source_experts[0]) {
            value = mixed_bf16(source_output_0[column]); found = true;
        } else if (expert == source_experts[1]) {
            value = mixed_bf16(source_output_1[column]); found = true;
        } else if (expert == source_experts[2]) {
            value = mixed_bf16(source_output_2[column]); found = true;
        } else {
            for (uint slot = 0u; slot < 5u; ++slot) {
                if (expert == k4_experts[slot]) {
                    value = mixed_bf16(k4_outputs[slot * 4096u + column]);
                    found = true;
                    break;
                }
            }
        }
        if (!found) atomic_fetch_or_explicit(error_flags, 1024u, memory_order_relaxed);
        sum += value * selected_weights[route];
    }
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}

kernel void mixed_route_weighted_reduce_dynamic_five_source_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_output_0 [[buffer(1)]],
    device const float *source_output_1 [[buffer(2)]],
    device const float *source_output_2 [[buffer(3)]],
    device const float *source_output_3 [[buffer(4)]],
    device const float *source_output_4 [[buffer(5)]],
    device const uint *selected_experts [[buffer(6)]],
    device const float *selected_weights [[buffer(7)]],
    constant const uint *k4_experts [[buffer(8)]],
    constant const uint *source_experts [[buffer(9)]],
    device float *routed [[buffer(10)]],
    device atomic_uint *error_flags [[buffer(11)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    device const float *source_outputs[5] = {
        source_output_0, source_output_1, source_output_2,
        source_output_3, source_output_4
    };
    float sum = 0.0f;
    for (uint route = 0u; route < 8u; ++route) {
        const uint expert = selected_experts[route];
        float value = 0.0f;
        bool found = false;
        for (uint slot = 0u; slot < 5u; ++slot) {
            if (expert == source_experts[slot]) {
                value = mixed_bf16(source_outputs[slot][column]);
                found = true;
                break;
            }
        }
        if (!found) {
            for (uint slot = 0u; slot < 3u; ++slot) {
                if (expert == k4_experts[slot]) {
                    value = mixed_bf16(k4_outputs[slot * 4096u + column]);
                    found = true;
                    break;
                }
            }
        }
        if (!found) atomic_fetch_or_explicit(error_flags, 1024u, memory_order_relaxed);
        sum += value * selected_weights[route];
    }
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}

kernel void mixed_route_weighted_reduce_dynamic_source_panel_bf16(
    device const float *k4_outputs [[buffer(0)]],
    device const float *source_outputs [[buffer(1)]],
    device const uint *selected_experts [[buffer(2)]],
    device const float *selected_weights [[buffer(3)]],
    constant const uint *k4_experts [[buffer(4)]],
    constant const uint *source_experts [[buffer(5)]],
    constant const uint &k4_count [[buffer(6)]],
    constant const uint &source_count [[buffer(7)]],
    device float *routed [[buffer(8)]],
    device atomic_uint *error_flags [[buffer(9)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 4096u) return;
    float sum = 0.0f;
    for (uint route = 0u; route < 8u; ++route) {
        const uint expert = selected_experts[route];
        float value = 0.0f;
        bool found = false;
        for (uint slot = 0u; slot < source_count; ++slot) {
            if (expert == source_experts[slot]) {
                value = mixed_bf16(source_outputs[slot * 4096u + column]);
                found = true;
                break;
            }
        }
        if (!found) {
            for (uint slot = 0u; slot < k4_count; ++slot) {
                if (expert == k4_experts[slot]) {
                    value = mixed_bf16(k4_outputs[slot * 4096u + column]);
                    found = true;
                    break;
                }
            }
        }
        if (!found) atomic_fetch_or_explicit(error_flags, 1024u, memory_order_relaxed);
        sum += value * selected_weights[route];
    }
    const float result = mixed_bf16(sum);
    if (!isfinite(result)) atomic_fetch_or_explicit(error_flags, 256u, memory_order_relaxed);
    routed[column] = result;
}
