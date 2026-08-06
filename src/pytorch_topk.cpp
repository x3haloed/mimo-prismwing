#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

extern "C" int pw_pytorch_topk_unsorted_f32(const float* values,
                                              std::size_t count,
                                              std::size_t top_k,
                                              std::uint32_t* selected) {
  if (values == nullptr || selected == nullptr || top_k == 0 || top_k >= count) {
    return -1;
  }
  try {
    std::vector<std::pair<float, std::int64_t>> queue(count);
    for (std::size_t index = 0; index < count; ++index) {
      queue[index] = {values[index], static_cast<std::int64_t>(index)};
    }
    std::nth_element(queue.begin(), queue.begin() + top_k - 1, queue.end(),
                     [](const auto& left, const auto& right) {
                       return left.first > right.first;
                     });
    for (std::size_t index = 0; index < top_k; ++index) {
      selected[index] = static_cast<std::uint32_t>(queue[index].second);
    }
  } catch (...) {
    return -2;
  }
  return 0;
}
