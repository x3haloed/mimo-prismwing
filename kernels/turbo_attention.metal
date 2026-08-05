#include <metal_stdlib>
using namespace metal;

struct TurboAttentionShape {
    uint context;
    uint format;
    uint key_stride;
    uint value_stride;
    uint q_heads;
    uint kv_heads;
    uint use_sinks;
};

constant float turbo3_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
constant float turbo4_centroids[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static void fwht(thread float * values) {
    for (uint width = 1; width < 128; width *= 2) {
        for (uint base = 0; base < 128; base += 2 * width) {
            for (uint offset = 0; offset < width; ++offset) {
                float first = values[base + offset];
                float second = values[base + offset + width];
                values[base + offset] = first + second;
                values[base + offset + width] = first - second;
            }
        }
    }
    for (uint index = 0; index < 128; ++index) values[index] *= 0.08838834764831845f;
}

static void rotate_forward(thread float * values, device const float * signs1, device const float * signs2);
static void rotate_inverse(thread float * values, device const float * signs1, device const float * signs2);
static float dequant(device const uchar * row, uint format, uint column);

kernel void turbo_gqa_attention_256_128_parallel32(
        device const uchar * keys [[buffer(0)]],
        device const uchar * values [[buffer(1)]],
        device const float * queries [[buffer(2)]],
        device float * guarded_outputs [[buffer(3)]],
        constant TurboAttentionShape & shape [[buffer(4)]],
        device const float * signs1 [[buffer(5)]],
        device const float * signs2 [[buffer(6)]],
        device const float * sinks [[buffer(7)]],
        uint lane [[thread_index_in_threadgroup]],
        uint q_head [[threadgroup_position_in_grid]],
        uint threads [[threads_per_threadgroup]]) {
    if (threads != 32 || q_head >= shape.q_heads || shape.q_heads != 64 ||
        (shape.kv_heads != 4 && shape.kv_heads != 8) ||
        (shape.format != 3 && shape.format != 4 && shape.format != 8)) return;
    uint kv_head = q_head / (shape.q_heads / shape.kv_heads);
    device const uchar * head_keys = keys + kv_head * shape.context * shape.key_stride;
    device const uchar * head_values = values + kv_head * shape.context * shape.value_stride;

    float rotated_query[256];
    for (uint index = 0; index < 256; ++index) rotated_query[index] = queries[q_head * 256 + index];
    rotate_forward(rotated_query, signs1, signs2);
    rotate_forward(rotated_query + 128, signs1, signs2);
    float accumulator[128];
    for (uint column = 0; column < 128; ++column) accumulator[column] = 0.0f;
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint token = lane; token < shape.context; token += 32) {
        device const uchar * key = head_keys + token * shape.key_stride;
        device const uchar * value = head_values + token * shape.value_stride;
        float score = 0.0f;
        for (uint column = 0; column < 256; ++column) score += rotated_query[column] * dequant(key, shape.format, column);
        score *= 0.07216878364870322f;
        float next_maximum = max(maximum, score);
        float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float current_scale = exp(score - next_maximum);
        denominator = denominator * previous_scale + current_scale;
        for (uint column = 0; column < 128; ++column) accumulator[column] = accumulator[column] * previous_scale + current_scale * dequant(value, shape.format, column);
        maximum = next_maximum;
    }
    threadgroup float partial_maximum[32];
    threadgroup float partial_denominator[32];
    threadgroup float partial_output[32 * 128];
    partial_maximum[lane] = maximum; partial_denominator[lane] = denominator;
    for (uint column = 0; column < 128; ++column) partial_output[lane * 128 + column] = accumulator[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float merged_maximum = -INFINITY;
        for (uint item = 0; item < 32; ++item) if (partial_denominator[item] > 0.0f) merged_maximum = max(merged_maximum, partial_maximum[item]);
        float merged_denominator = 0.0f, merged[128];
        for (uint column = 0; column < 128; ++column) merged[column] = 0.0f;
        for (uint item = 0; item < 32; ++item) if (partial_denominator[item] > 0.0f) {
            float scale = exp(partial_maximum[item] - merged_maximum);
            merged_denominator += partial_denominator[item] * scale;
            for (uint column = 0; column < 128; ++column) merged[column] += partial_output[item * 128 + column] * scale;
        }
        if (shape.use_sinks != 0) {
            float sink = sinks[q_head];
            float final_maximum = max(merged_maximum, sink);
            float data_scale = exp(merged_maximum - final_maximum);
            merged_denominator = merged_denominator * data_scale + exp(sink - final_maximum);
            for (uint column = 0; column < 128; ++column) merged[column] *= data_scale;
        }
        for (uint column = 0; column < 128; ++column) merged[column] /= merged_denominator;
        rotate_inverse(merged, signs1, signs2);
        device float * output = guarded_outputs + q_head * 130;
        for (uint column = 0; column < 128; ++column) output[column + 1] = merged[column];
    }
}

kernel void turbo_gqa_attention_shared_kv(
        device const uchar * keys [[buffer(0)]],
        device const uchar * values [[buffer(1)]],
        device const float * rotated_queries [[buffer(2)]],
        device float * guarded_outputs [[buffer(3)]],
        constant TurboAttentionShape & shape [[buffer(4)]],
        device const float * signs1 [[buffer(5)]],
        device const float * signs2 [[buffer(6)]],
        device const float * sinks [[buffer(7)]],
        uint thread_index [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint query_in_group [[simdgroup_index_in_threadgroup]],
        uint kv_head [[threadgroup_position_in_grid]],
        uint threads [[threads_per_threadgroup]]) {
    const uint queries_per_kv = shape.q_heads / shape.kv_heads;
    if (shape.q_heads != 64 || (shape.kv_heads != 4 && shape.kv_heads != 8) ||
        threads != queries_per_kv * 32 || kv_head >= shape.kv_heads ||
        query_in_group >= queries_per_kv ||
        (shape.format != 3 && shape.format != 4 && shape.format != 8)) return;
    const uint q_head = kv_head * queries_per_kv + query_in_group;
    device const uchar * head_keys = keys + kv_head * shape.context * shape.key_stride;
    device const uchar * head_values = values + kv_head * shape.context * shape.value_stride;
    threadgroup float tile[8 * 384];
    threadgroup float reconstructed[16 * 128];
    float output[4] = {0, 0, 0, 0};
    float maximum = -INFINITY, denominator = 0.0f;

    for (uint tile_base = 0; tile_base < shape.context; tile_base += 8) {
        uint tile_count = min(8u, shape.context - tile_base);
        for (uint item = thread_index; item < tile_count * 384; item += threads) {
            uint token = item / 384, column = item % 384;
            tile[item] = column < 256
                ? dequant(head_keys + (tile_base + token) * shape.key_stride, shape.format, column)
                : dequant(head_values + (tile_base + token) * shape.value_stride, shape.format, column - 256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint token = 0; token < tile_count; ++token) {
            float partial = 0.0f;
            for (uint column = lane; column < 256; column += 32) {
                partial += rotated_queries[q_head * 256 + column] * tile[token * 384 + column];
            }
            float score = simd_sum(partial) * 0.07216878364870322f;
            float next_maximum = max(maximum, score);
            float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
            float current_scale = exp(score - next_maximum);
            denominator = denominator * previous_scale + current_scale;
            for (uint part = 0; part < 4; ++part) {
                uint column = lane + part * 32;
                output[part] = output[part] * previous_scale + current_scale * tile[token * 384 + 256 + column];
            }
            maximum = next_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (shape.use_sinks != 0) {
        float sink = sinks[q_head], next_maximum = max(maximum, sink);
        float data_scale = exp(maximum - next_maximum);
        denominator = denominator * data_scale + exp(sink - next_maximum);
        for (uint part = 0; part < 4; ++part) output[part] *= data_scale;
    }
    for (uint part = 0; part < 4; ++part) {
        uint column = lane + part * 32;
        reconstructed[query_in_group * 128 + column] = output[part] / denominator * signs2[column];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 128; width *= 2) {
        for (uint butterfly = lane; butterfly < 64; butterfly += 32) {
            uint group = butterfly / width, offset = butterfly % width;
            uint first_index = group * 2 * width + offset;
            threadgroup float * row = reconstructed + query_in_group * 128;
            float first = row[first_index], second = row[first_index + width];
            row[first_index] = first + second; row[first_index + width] = first - second;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float * destination = guarded_outputs + q_head * 130;
    for (uint part = 0; part < 4; ++part) {
        uint column = lane + part * 32;
        destination[column + 1] = reconstructed[query_in_group * 128 + column] * 0.08838834764831845f * signs1[column];
    }
}

static void rotate_forward(thread float * values, device const float * signs1, device const float * signs2) {
    for (uint index = 0; index < 128; ++index) values[index] *= signs1[index];
    fwht(values);
    for (uint index = 0; index < 128; ++index) values[index] *= signs2[index];
}

static void rotate_inverse(thread float * values, device const float * signs1, device const float * signs2) {
    for (uint index = 0; index < 128; ++index) values[index] *= signs2[index];
    fwht(values);
    for (uint index = 0; index < 128; ++index) values[index] *= signs1[index];
}

static float dequant(device const uchar * row, uint format, uint column) {
    uint block = column / 128;
    uint within = column % 128;
    if (format == 3) {
        device const uchar * packed = row + block * 50;
        float norm = float(*((device const half *)packed));
        uint low = (packed[2 + within / 4] >> ((within % 4) * 2)) & 3;
        uint high = (packed[34 + within / 8] >> (within % 8)) & 1;
        return turbo3_centroids[low | (high << 2)] * norm;
    }
    if (format == 8) {
        device const uchar * packed = row + block * 130;
        float scale = float(*((device const half *)packed));
        device const char * codes = (device const char *)(packed + 2);
        return float(codes[within]) * scale;
    }
    device const uchar * packed = row + block * 68;
    float norm = float(*((device const half *)packed));
    uint index = (packed[4 + within / 2] >> ((within % 2) * 4)) & 15;
    return turbo4_centroids[index] * norm;
}

kernel void turbo_attention_256_128(
        device const uchar * keys [[buffer(0)]],
        device const uchar * values [[buffer(1)]],
        device const float * query [[buffer(2)]],
        device float * output_with_guards [[buffer(3)]],
        constant TurboAttentionShape & shape [[buffer(4)]],
        device const float * signs1 [[buffer(5)]],
        device const float * signs2 [[buffer(6)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0 ||
        (shape.format != 3 && shape.format != 4 && shape.format != 8)) return;

    float rotated_query[256];
    for (uint index = 0; index < 256; ++index) rotated_query[index] = query[index];
    rotate_forward(rotated_query, signs1, signs2);
    rotate_forward(rotated_query + 128, signs1, signs2);

    float accumulator[128];
    for (uint column = 0; column < 128; ++column) accumulator[column] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint token = 0; token < shape.context; ++token) {
        device const uchar * key = keys + token * shape.key_stride;
        device const uchar * value = values + token * shape.value_stride;
        float score = 0.0f;
        for (uint column = 0; column < 256; ++column) {
            score += rotated_query[column] * dequant(key, shape.format, column);
        }
        score *= 0.07216878364870322f; // 1/sqrt(192)
        float next_maximum = max(maximum, score);
        float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float current_scale = exp(score - next_maximum);
        denominator = denominator * previous_scale + current_scale;
        for (uint column = 0; column < 128; ++column) {
            accumulator[column] = accumulator[column] * previous_scale
                + current_scale * dequant(value, shape.format, column);
        }
        maximum = next_maximum;
    }
    for (uint column = 0; column < 128; ++column) accumulator[column] /= denominator;
    rotate_inverse(accumulator, signs1, signs2);
    for (uint column = 0; column < 128; ++column) output_with_guards[column + 1] = accumulator[column];
}

kernel void turbo_attention_256_128_parallel32(
        device const uchar * keys [[buffer(0)]],
        device const uchar * values [[buffer(1)]],
        device const float * query [[buffer(2)]],
        device float * output_with_guards [[buffer(3)]],
        constant TurboAttentionShape & shape [[buffer(4)]],
        device const float * signs1 [[buffer(5)]],
        device const float * signs2 [[buffer(6)]],
        uint lane [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    if (threads != 32 ||
        (shape.format != 3 && shape.format != 4 && shape.format != 8)) return;

    float rotated_query[256];
    for (uint index = 0; index < 256; ++index) rotated_query[index] = query[index];
    rotate_forward(rotated_query, signs1, signs2);
    rotate_forward(rotated_query + 128, signs1, signs2);

    float accumulator[128];
    for (uint column = 0; column < 128; ++column) accumulator[column] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint token = lane; token < shape.context; token += 32) {
        device const uchar * key = keys + token * shape.key_stride;
        device const uchar * value = values + token * shape.value_stride;
        float score = 0.0f;
        for (uint column = 0; column < 256; ++column) score += rotated_query[column] * dequant(key, shape.format, column);
        score *= 0.07216878364870322f;
        float next_maximum = max(maximum, score);
        float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float current_scale = exp(score - next_maximum);
        denominator = denominator * previous_scale + current_scale;
        for (uint column = 0; column < 128; ++column) accumulator[column] = accumulator[column] * previous_scale + current_scale * dequant(value, shape.format, column);
        maximum = next_maximum;
    }

    threadgroup float partial_maximum[32];
    threadgroup float partial_denominator[32];
    threadgroup float partial_output[32 * 128];
    partial_maximum[lane] = maximum;
    partial_denominator[lane] = denominator;
    for (uint column = 0; column < 128; ++column) partial_output[lane * 128 + column] = accumulator[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float merged_maximum = -INFINITY;
        for (uint item = 0; item < 32; ++item) if (partial_denominator[item] > 0.0f) merged_maximum = max(merged_maximum, partial_maximum[item]);
        float merged_denominator = 0.0f;
        float merged[128];
        for (uint column = 0; column < 128; ++column) merged[column] = 0.0f;
        for (uint item = 0; item < 32; ++item) {
            if (partial_denominator[item] == 0.0f) continue;
            float scale = exp(partial_maximum[item] - merged_maximum);
            merged_denominator += partial_denominator[item] * scale;
            for (uint column = 0; column < 128; ++column) merged[column] += partial_output[item * 128 + column] * scale;
        }
        for (uint column = 0; column < 128; ++column) merged[column] /= merged_denominator;
        rotate_inverse(merged, signs1, signs2);
        for (uint column = 0; column < 128; ++column) output_with_guards[column + 1] = merged[column];
    }
}
