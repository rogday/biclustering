#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <random>
#include <exception>
#include <stdexcept>

//#define DEBUG

constexpr std::uint64_t NOT_IN_CLUSTER = 0;

class biclustering_solver_t {

  std::vector<std::int64_t> split(std::string const &str) {
    std::istringstream iss(str);
    return {std::istream_iterator<std::int64_t>{iss},
            std::istream_iterator<std::int64_t>{}};
  }

public:
  struct cell_t {
    std::uint64_t value = 0;
    std::uint64_t cluster_id = NOT_IN_CLUSTER;
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
  }

  void full_clear() {
    clusters = 0;
    ones_overall = 0;
    matrix.clear();
  }

  void random_clear() {
    for (auto &vector : matrix)
      for (auto &cell : vector)
        cell.cluster_id = NOT_IN_CLUSTER;
  }

  void initial_random() {
    static constexpr auto ITERATIONS = 7'000;

    two_dim_matrix_t new_matrix;
    std::int64_t new_ones_overall;
    std::size_t new_clusters;

    double max = std::numeric_limits<double>::min();

    for (std::size_t index = 0; index < ITERATIONS; ++index) {
      random_clear();
      random_pass();
      double evaluation = loss();
      if (evaluation > max) {
        new_matrix = matrix;
        new_ones_overall = ones_overall;
        new_clusters = clusters;

        max = evaluation;
        std::cout << index << " " << max << std::endl;
      }
    }

    matrix = new_matrix;
    ones_overall = new_ones_overall;
    clusters = new_clusters;
  }

  void random_pass() {
    static std::vector<std::size_t> machines_clusters(matrix.size());
    static std::vector<std::size_t> parts_clusters(matrix[0].size());

    static std::vector<std::size_t> machines(matrix.size());
    static std::vector<std::size_t> parts(matrix[0].size());

    std::iota(std::begin(machines), std::end(machines), 0);
    std::iota(std::begin(parts), std::end(parts), 0);

    std::mt19937 prng(std::random_device{}());

    std::shuffle(std::begin(machines), std::end(machines), prng);
    std::shuffle(std::begin(parts), std::end(parts), prng);

    std::size_t i = 0; // i - сколько уже сгенерили для машин
    std::size_t j = 0; // j - сколько уже сгенерили для частей

    std::size_t cluster_id = 1;
    clusters = 1 + prng() % machines_clusters.size();

    auto set_clusters = [&cluster_id](auto &outer, auto &inner,
                                      std::size_t start, std::size_t end) {
      for (std::size_t index = start; index < end; ++index)
        outer[inner[index]] = cluster_id;
    };

    for (std::size_t cluster = 1; cluster < clusters; ++cluster, ++cluster_id) {

      std::size_t cluster_size_machines =
          1 + prng() % (machines_clusters.size() - (clusters - cluster) - i);

      std::size_t cluster_size_parts =
          1 + prng() % (parts_clusters.size() - (clusters - cluster) - j);

      set_clusters(machines_clusters, machines, i, i + cluster_size_machines);
      set_clusters(parts_clusters, parts, j, j + cluster_size_parts);

      i += cluster_size_machines;
      j += cluster_size_parts;
    }

    set_clusters(machines_clusters, machines, i, machines_clusters.size());
    set_clusters(parts_clusters, parts, j, parts_clusters.size());

    for (std::size_t i = 0; i < machines_clusters.size(); ++i)
      for (std::size_t j = 0; j < parts_clusters.size(); ++j)
        if (machines_clusters[i] == parts_clusters[j])
          matrix[i][j].cluster_id = parts_clusters[j];
  }

  double loss() {
    std::size_t zeros_in_solution = 0;
    std::size_t ones_in_solution = 0;

    for (auto &vector : matrix)
      for (auto &cell : vector)
        if (cell.cluster_id != NOT_IN_CLUSTER) {
          ones_in_solution += cell.value;
          zeros_in_solution += !cell.value;
        }

    return ones_in_solution / double(ones_overall + zeros_in_solution);
  }

  void print() {
    for (auto &vector : matrix) {
      for (auto &element : vector)
        std::cout << element.value << " ";
      std::cout << std::endl;
    }
  }

public:
  std::int64_t ones_overall = 0;

  two_dim_matrix_t matrix;
  std::size_t clusters;

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
  biclustering_solver.initial_random();
}