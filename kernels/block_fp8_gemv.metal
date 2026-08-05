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
    uint block_columns;
};

kernel void block_fp8_gemv(
    device const uchar *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GemvShape &shape [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= shape.rows || shape.block_columns == 0 || shape.columns % shape.block_columns != 0) {
        return;
    }
    float sum = 0.0f;
    const uint row_offset = row * shape.columns;
    for (uint column = 0; column < shape.columns; ++column) {
        const float weight = decode_f8_e4m3fn(weights[row_offset + column]);
        const float scale = scales[column / shape.block_columns];
        sum += weight * scale * input[column];
    }
    output[row] = sum;
}
