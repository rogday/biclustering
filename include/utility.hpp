#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <filesystem>
#include <random>

namespace utility {
enum genes_t {
  SHAKINGS = 0,
  ITERATIONS,
  CLUSTERING_PRINCIPLE,
  SHAKINGS_IN_A_ROW,
  SIZE
};

constexpr std::uint64_t NOT_IN_CLUSTER = 0;
constexpr double eps = 1e-12;

std::string filename_from_path(std::string &path);

static inline std::mt19937 generator{std::random_device{}()};

std::size_t generate_random_value(
    std::size_t lower_bound = std::numeric_limits<std::size_t>::min(),
    std::size_t upper_bound = std::numeric_limits<std::size_t>::max());

template <typename T>
void construct_matrix(std::vector<std::vector<T>> &mat, std::size_t n,
                      std::size_t m) {
  mat.resize(n);
  std::fill(std::begin(mat), std::end(mat), std::vector<T>(m));
}

template <typename T>
void append_clear(std::vector<T> &to, std::vector<T> &from) {
  to.insert(std::end(to), std::begin(from), std::end(from));
  from.clear();
}

template <typename T>
void split(std::vector<T> &from, std::vector<T> &to, std::size_t index) {
  to.insert(std::end(to), std::begin(from) + index, std::end(from));
  from.resize(index);
}

std::vector<std::int64_t> split(std::string const &str);

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig) {
  std::vector<T> ret;
  for (const auto &v : orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

std::vector<std::string> get_files(std::string path);

} // namespace utility
