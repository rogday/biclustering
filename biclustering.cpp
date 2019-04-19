#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <filesystem>

class biclustering_solver_t {

  std::vector<std::int64_t> split(std::string const &str) {
    std::istringstream iss(str);
    return {std::istream_iterator<std::int64_t>{iss},
            std::istream_iterator<std::int64_t>{}};
  }

public:
  struct cell_t {
    std::uint64_t value = 0;
    std::uint64_t cluster_id = 0;
  };

  using entry_t = std::vector<std::string>;
  using two_dim_matrix_t = std::vector<std::vector<cell_t>>;

  biclustering_solver_t() = default;
  biclustering_solver_t(biclustering_solver_t const &) = delete;
  biclustering_solver_t(biclustering_solver_t &&) = delete;

  void parse(std::string const &filename) {
    std::fstream filestream{filename};
    std::string line;

    /** M, P (number of machines and parts) */
    std::getline(filestream, line);
    auto dimensions = split(line);

    matrix.resize(dimensions[0]);
    std::fill(std::begin(matrix), std::end(matrix),
              std::vector<cell_t>(dimensions[1]));

    while (std::getline(filestream, line)) {
      auto vector = split(line);
      ones_overall += vector.size() - 1;
      for (std::size_t i = 1; i < vector.size(); ++i)
        matrix[vector[0] - 1][vector[i] - 1].value = true;
    }

    for (auto &vector : matrix) {
      for (auto &element : vector)
        std::cout << element.value << " ";
      std::cout << std::endl;
    }
  }

  two_dim_matrix_t get_partial_sum() {
    std::size_t n = matrix.size() + 1, m = matrix[0].size() + 1;
    two_dim_matrix_t greedy_matrix(n, std::vector<cell_t>(m));

    for (std::size_t i = 1; i < n; ++i)
      for (std::size_t j = 1; j < m; ++j)
        greedy_matrix[i][j].value =
            matrix[i - 1][j - 1].value + greedy_matrix[i - 1][j].value +
            greedy_matrix[i][j - 1].value - greedy_matrix[i - 1][j - 1].value;

    return greedy_matrix;
  }

  void greedy_clustering() {
    std::size_t n = matrix.size() + 1, m = matrix[0].size() + 1;
    double sum = std::numeric_limits<double>::min();

    two_dim_matrix_t greedy_matrix = get_partial_sum();

    //количество_единичек_в_кластере/(количество_единичек_всего +
    //количество_ноликов_во_всех_кластерах)

    /** ones_in_cluster/zeros_in_cluster -
        (ones_left_in_rows/zeros_left_in_rows +
        ones_left_in_cols/zeros_left_in_cols) */

    std::size_t start_i, start_j, end_i, end_j;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < m; ++j)
        for (std::size_t di = 0; di < n - i; ++di)
          for (std::size_t dj = 0; dj < m - j; ++dj) {
            std::int64_t ones_in_cluster =
                get_sum_inclusive(greedy_matrix, di, dj, di + i, dj + j);

            std::int64_t zeros_in_cluster = i * j - ones_in_cluster;

            std::int64_t ones_left_in_rows =
                get_sum_inclusive(greedy_matrix, 0, dj, n - 1, dj + j) -
                ones_in_cluster;

            std::int64_t zeros_left_in_rows = j * n - ones_left_in_rows;

            std::int64_t ones_left_in_cols =
                get_sum_inclusive(greedy_matrix, di, 0, di + i, m - 1) -
                ones_in_cluster;

            std::int64_t zeros_left_in_cols = i * m - ones_left_in_cols;

            double loss =
                (ones_in_cluster / (zeros_in_cluster + 1.0)) /
                (ones_left_in_rows / (zeros_left_in_rows + 1.0) +
                 ones_left_in_cols / (zeros_left_in_cols + 1.0) + 1.0);

            if (loss > sum) {
              sum = loss;
              start_i = di + 1;
              start_j = dj + 1;
              end_i = di + i;
              end_j = dj + j;
            }
          }
    std::cout << std::endl
              << start_i << ", " << start_j << std::endl
              << end_i << ", " << end_j << std::endl
              << std::endl;

    for (auto i = start_i; i <= end_i; ++i) {
      for (auto j = start_j; j <= end_j; ++j) {
        std::cout << matrix[i - 1][j - 1].value << " ";
      }
      std::cout << std::endl;
    }

    std::cout << std::endl << sum << " " << ones_overall << std::endl;
  }

  std::int64_t get_sum_inclusive(two_dim_matrix_t &mat, std::size_t start_i,
                                 std::size_t start_j, std::size_t end_i,
                                 std::size_t end_j) {
    return mat[end_i][end_j].value - mat[start_i][end_j].value -
           mat[end_i][start_j].value + mat[start_i][start_j].value;
  }

private:
  std::int64_t ones_overall = 0;
  two_dim_matrix_t matrix;
  std::string filename;
};

auto get_files(std::string path) {
  biclustering_solver_t::entry_t files;
  for (auto entry : std::filesystem::directory_iterator(path))
    files.push_back(entry.path().string());
  return files;
}

int main(int argc, char *argv[]) {
  biclustering_solver_t biclustering_solver;
  auto input_files = get_files("../input");

  std::size_t i = ((argc == 1) ? 0 : std::atoi(argv[1])) % input_files.size();
  // for (auto &file : input_files)
  // biclustering_solver.parse(file);
  biclustering_solver.parse(input_files[i]);
  std::cout << std::endl << std::endl;

  auto greedy_matrix = biclustering_solver.get_partial_sum();

  for (auto &vector : greedy_matrix) {
    for (auto &element : vector)
      std::cout << element.value << " ";
    std::cout << std::endl;
  }

  biclustering_solver.greedy_clustering();
}