#include <cmath>
#include <iostream>
#include <vector>

#include "mlx/mlx.h"

int main() {
  using namespace mlx::core;
  constexpr int rows = 4;
  constexpr int columns = 128;
  std::vector<float> weights(rows * columns);
  std::vector<float> input_values(columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      weights[row * columns + column] =
          std::sin(static_cast<float>(row * columns + column) / 29.0f);
    }
  }
  for (int column = 0; column < columns; ++column) {
    input_values[column] = std::cos(static_cast<float>(column) / 17.0f) * 0.01f;
  }

  array weight(weights.begin(), {rows, columns}, float32);
  array input(input_values.begin(), {1, columns}, float32);
  auto quantized = quantize(weight, 128, 4, "affine");
  if (quantized.size() != 3) {
    std::cerr << "unexpected affine quantization result count\n";
    return 1;
  }
  auto output = quantized_matmul(
      input,
      quantized[0],
      quantized[1],
      quantized[2],
      true,
      128,
      4,
      "affine");
  eval(output);
  if (output.shape() != Shape{1, rows}) {
    std::cerr << "unexpected quantized matmul shape\n";
    return 1;
  }
  const float* values = output.data<float>();
  for (int index = 0; index < rows; ++index) {
    if (!std::isfinite(values[index])) {
      std::cerr << "non-finite quantized matmul output\n";
      return 1;
    }
  }
  std::cout << "mlx_cpp_affine_int4_ok rows=" << rows
            << " columns=" << columns << " first=" << values[0] << '\n';
  return 0;
}
