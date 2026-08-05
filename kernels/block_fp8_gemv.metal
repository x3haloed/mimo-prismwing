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

struct GemvShape {
    uint rows;
    uint columns;
    uint block_rows;
    uint block_columns;
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
