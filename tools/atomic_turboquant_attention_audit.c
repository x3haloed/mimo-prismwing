// Compile with ggml/src on the include path from the locked Atomic source.
// Including the implementation deliberately exposes its static WHT oracle so
// this audit tests the pinned code rather than a Prismwing reimplementation.
#include "ggml-turbo-quant.c"

#include <float.h>
#include <stdio.h>

#define CONTEXT 17
#define K_LOGICAL 192
#define K_PADDED 256
#define V_DIM 128

typedef struct {
    const char * name;
    size_t block_bytes;
    double score_relative_l2;
    double score_max_abs;
    double output_relative_l2;
    double output_cosine;
    int deterministic;
    uint64_t bytes_at_max_context;
} audit_result;

static uint64_t fixture_state = 0x91e10da5c79e7b1dULL;

static float fixture_value(void) {
    fixture_state = fixture_state * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t bits = (uint32_t)(fixture_state >> 32);
    return ((float)(bits & 0xffffu) / 32767.5f - 1.0f) * 1.75f;
}

static double dot(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)a[i] * b[i];
    return sum;
}

static void softmax(const double * scores, double * probabilities) {
    double maximum = -DBL_MAX;
    for (int i = 0; i < CONTEXT; ++i) if (scores[i] > maximum) maximum = scores[i];
    double total = 0.0;
    for (int i = 0; i < CONTEXT; ++i) {
        probabilities[i] = exp(scores[i] - maximum);
        total += probabilities[i];
    }
    for (int i = 0; i < CONTEXT; ++i) probabilities[i] /= total;
}

static double relative_l2(const double * actual, const double * expected, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; ++i) {
        const double delta = actual[i] - expected[i];
        error += delta * delta;
        reference += expected[i] * expected[i];
    }
    return sqrt(error / reference);
}

static double cosine(const double * actual, const double * expected, int n) {
    double product = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        product += actual[i] * expected[i];
        actual_norm += actual[i] * actual[i];
        expected_norm += expected[i] * expected[i];
    }
    return product / sqrt(actual_norm * expected_norm);
}

static uint64_t mimo_cache_bytes(size_t block_bytes) {
    const uint64_t maximum_context = 1048576;
    const uint64_t global_per_token = 9u * 4u * 3u * block_bytes;
    const uint64_t sliding_fixed = 39u * 8u * 128u * 3u * block_bytes;
    return global_per_token * maximum_context + sliding_fixed;
}

static audit_result audit_type(
        const char * name,
        int kind,
        const float query_rotated[K_PADDED],
        const float keys[CONTEXT][K_PADDED],
        const float values[CONTEXT][V_DIM],
        const double baseline_scores[CONTEXT],
        const double baseline_output[V_DIM]) {
    size_t block_bytes = kind == 2 ? sizeof(block_turbo2_0)
                       : kind == 3 ? sizeof(block_turbo3_0)
                                   : sizeof(block_turbo4_0);
    const size_t key_bytes = block_bytes * 2;
    const size_t value_bytes = block_bytes;
    unsigned char packed_key[CONTEXT][136];
    unsigned char packed_key_again[CONTEXT][136];
    unsigned char packed_value[CONTEXT][68];
    float key_rotated[CONTEXT][K_PADDED];
    float value_rotated[CONTEXT][V_DIM];
    double scores[CONTEXT];
    double probabilities[CONTEXT];
    double output_rotated[V_DIM] = {0};
    float output_inverse[V_DIM];
    double output[V_DIM];
    int deterministic = 1;

    turbo3_cpu_wht_group_size = 128;
    for (int token = 0; token < CONTEXT; ++token) {
        memset(packed_key[token], 0, sizeof(packed_key[token]));
        memset(packed_key_again[token], 0, sizeof(packed_key_again[token]));
        memset(packed_value[token], 0, sizeof(packed_value[token]));
        if (kind == 2) {
            quantize_row_turbo2_0_ref(keys[token], (block_turbo2_0 *)packed_key[token], K_PADDED);
            quantize_row_turbo2_0_ref(keys[token], (block_turbo2_0 *)packed_key_again[token], K_PADDED);
            quantize_row_turbo2_0_ref(values[token], (block_turbo2_0 *)packed_value[token], V_DIM);
            dequantize_row_turbo2_0((const block_turbo2_0 *)packed_key[token], key_rotated[token], K_PADDED);
            dequantize_row_turbo2_0((const block_turbo2_0 *)packed_value[token], value_rotated[token], V_DIM);
        } else if (kind == 3) {
            quantize_row_turbo3_0_ref(keys[token], (block_turbo3_0 *)packed_key[token], K_PADDED);
            quantize_row_turbo3_0_ref(keys[token], (block_turbo3_0 *)packed_key_again[token], K_PADDED);
            quantize_row_turbo3_0_ref(values[token], (block_turbo3_0 *)packed_value[token], V_DIM);
            dequantize_row_turbo3_0((const block_turbo3_0 *)packed_key[token], key_rotated[token], K_PADDED);
            dequantize_row_turbo3_0((const block_turbo3_0 *)packed_value[token], value_rotated[token], V_DIM);
        } else {
            quantize_row_turbo4_0_ref(keys[token], (block_turbo4_0 *)packed_key[token], K_PADDED);
            quantize_row_turbo4_0_ref(keys[token], (block_turbo4_0 *)packed_key_again[token], K_PADDED);
            quantize_row_turbo4_0_ref(values[token], (block_turbo4_0 *)packed_value[token], V_DIM);
            dequantize_row_turbo4_0((const block_turbo4_0 *)packed_key[token], key_rotated[token], K_PADDED);
            dequantize_row_turbo4_0((const block_turbo4_0 *)packed_value[token], value_rotated[token], V_DIM);
        }
        deterministic &= memcmp(packed_key[token], packed_key_again[token], key_bytes) == 0;
        scores[token] = dot(query_rotated, key_rotated[token], K_PADDED) / sqrt((double)K_LOGICAL);
    }
    softmax(scores, probabilities);
    for (int token = 0; token < CONTEXT; ++token) {
        for (int column = 0; column < V_DIM; ++column) {
            output_rotated[column] += probabilities[token] * value_rotated[token][column];
        }
    }
    for (int column = 0; column < V_DIM; ++column) output_inverse[column] = (float)output_rotated[column];
    turbo_cpu_fwht_inverse(output_inverse, V_DIM);
    for (int column = 0; column < V_DIM; ++column) output[column] = output_inverse[column];

    double maximum_score_error = 0.0;
    for (int token = 0; token < CONTEXT; ++token) {
        double error = fabs(scores[token] - baseline_scores[token]);
        if (error > maximum_score_error) maximum_score_error = error;
    }
    (void)value_bytes;
    audit_result result = {
        name,
        block_bytes,
        relative_l2(scores, baseline_scores, CONTEXT),
        maximum_score_error,
        relative_l2(output, baseline_output, V_DIM),
        cosine(output, baseline_output, V_DIM),
        deterministic,
        mimo_cache_bytes(block_bytes),
    };
    return result;
}

int main(void) {
    float query[K_PADDED] = {0};
    float query_rotated[K_PADDED];
    float keys[CONTEXT][K_PADDED] = {{0}};
    float keys_rotated[CONTEXT][K_PADDED];
    float values[CONTEXT][V_DIM];
    double baseline_scores[CONTEXT];
    double rotated_scores[CONTEXT];
    double probabilities[CONTEXT];
    double baseline_output[V_DIM] = {0};

    for (int column = 0; column < K_LOGICAL; ++column) query[column] = fixture_value();
    for (int token = 0; token < CONTEXT; ++token) {
        for (int column = 0; column < K_LOGICAL; ++column) keys[token][column] = fixture_value();
        for (int column = 0; column < V_DIM; ++column) values[token][column] = fixture_value();
        baseline_scores[token] = dot(query, keys[token], K_LOGICAL) / sqrt((double)K_LOGICAL);
    }
    softmax(baseline_scores, probabilities);
    for (int token = 0; token < CONTEXT; ++token) {
        for (int column = 0; column < V_DIM; ++column) {
            baseline_output[column] += probabilities[token] * values[token][column];
        }
    }

    memcpy(query_rotated, query, sizeof(query));
    turbo_cpu_fwht(query_rotated, 128);
    turbo_cpu_fwht(query_rotated + 128, 128);
    for (int token = 0; token < CONTEXT; ++token) {
        memcpy(keys_rotated[token], keys[token], sizeof(keys[token]));
        turbo_cpu_fwht(keys_rotated[token], 128);
        turbo_cpu_fwht(keys_rotated[token] + 128, 128);
        rotated_scores[token] = dot(query_rotated, keys_rotated[token], K_PADDED) / sqrt((double)K_LOGICAL);
    }
    const double wht_relative_error = relative_l2(rotated_scores, baseline_scores, CONTEXT);

    audit_result results[3] = {
        audit_type("turbo2", 2, query_rotated, keys, values, baseline_scores, baseline_output),
        audit_type("turbo3", 3, query_rotated, keys, values, baseline_scores, baseline_output),
        audit_type("turbo4", 4, query_rotated, keys, values, baseline_scores, baseline_output),
    };
    const uint64_t fp16_bytes = 23040ULL * 1048576ULL + 39ULL * 8ULL * 128ULL * 640ULL;

    printf("{\"schema_version\":1,\"context_tokens\":%d,", CONTEXT);
    printf("\"k_logical\":%d,\"k_padded\":%d,\"v_dim\":%d,", K_LOGICAL, K_PADDED, V_DIM);
    printf("\"wht_score_relative_l2\":%.12g,\"fp16_bytes_at_max_context\":%llu,\"candidates\":[",
           wht_relative_error, (unsigned long long)fp16_bytes);
    for (int index = 0; index < 3; ++index) {
        const audit_result * result = &results[index];
        if (index) printf(",");
        printf("{\"name\":\"%s\",\"block_bytes\":%zu,\"score_relative_l2\":%.12g,",
               result->name, result->block_bytes, result->score_relative_l2);
        printf("\"score_max_abs\":%.12g,\"output_relative_l2\":%.12g,\"output_cosine\":%.12g,",
               result->score_max_abs, result->output_relative_l2, result->output_cosine);
        printf("\"deterministic\":%s,\"bytes_at_max_context\":%llu,\"compression_vs_fp16\":%.12g}",
               result->deterministic ? "true" : "false",
               (unsigned long long)result->bytes_at_max_context,
               (double)fp16_bytes / (double)result->bytes_at_max_context);
    }
    printf("]}\n");

    if (sizeof(block_turbo2_0) != 34 || sizeof(block_turbo3_0) != 50 || sizeof(block_turbo4_0) != 68) return 2;
    if (!(wht_relative_error <= 2e-5)) return 3;
    for (int index = 0; index < 3; ++index) {
        if (!results[index].deterministic || !isfinite(results[index].score_relative_l2) ||
            !isfinite(results[index].output_relative_l2)) return 4;
    }
    return 0;
}
