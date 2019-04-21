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

//#define DEBUG 0

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

    machines_clusters.resize(dimensions[0]);
    machines_clusters_indices.resize(dimensions[0]);

    parts_clusters.resize(dimensions[1]);
    parts_clusters_indices.resize(dimensions[1]);

    matrix.resize(dimensions[0]);
    std::fill(std::begin(matrix), std::end(matrix),
              std::vector<cell_t>(dimensions[1]));
    while (std::getline(filestream, line)) {
      auto vector = split(line);
      ones_overall += vector.size() - 1;
      for (std::size_t i = 1; i < vector.size(); ++i)
        matrix[vector[0] - 1][vector[i] - 1].value = true;
    }

#ifdef DEBUG
    for (auto &vector : matrix) {
      for (auto &element : vector)
        std::cout << element.value << " ";
      std::cout << std::endl;
    }
#endif
  }

  void clear() {
    clusters = 0;
    ones_overall = 0;

    machines_clusters.clear();
    parts_clusters.clear();

    machines_clusters_indices.clear();
    parts_clusters_indices.clear();

    matrix.clear();
  }

  void initial_random() {

    std::vector<std::size_t> machines(matrix.size());
    std::vector<std::size_t> parts(matrix[0].size());

    std::iota(std::begin(machines), std::end(machines), 0);
    std::iota(std::begin(parts), std::end(parts), 0);

    std::mt19937 prng(std::random_device{}());

    std::shuffle(std::begin(machines), std::end(machines), prng);
    std::shuffle(std::begin(parts), std::end(parts), prng);

    std::size_t i = 0; // i - сколько уже сгенерили для машин
    std::size_t j = 0; // j - сколько уже сгенерили для частей

    std::size_t cluster_id = 1;
    clusters = 1 + prng() % machines_clusters.size();
#ifdef DEBUG
    std::cout << "clusters: " << clusters << std::endl;
#endif
    auto set_clusters = [&cluster_id](auto &outer, auto &inner, auto &_,
                                      std::size_t start, std::size_t end) {
      for (std::size_t index = start; index < end; ++index) {
        outer[inner[index]] = cluster_id;
        _[cluster_id - 1].push_back(index);
      }
    };

    for (std::size_t cluster = 1; cluster < clusters; ++cluster, ++cluster_id) {

      std::size_t cluster_size_machines =
          1 + prng() % (machines_clusters.size() - (clusters - cluster) - i);

      std::size_t cluster_size_parts =
          1 + prng() % (parts_clusters.size() - (clusters - cluster) - j);

      set_clusters(machines_clusters, machines, machines_clusters_indices, i,
                   i + cluster_size_machines);
      set_clusters(parts_clusters, parts, parts_clusters_indices, j,
                   j + cluster_size_parts);

      i += cluster_size_machines;
      j += cluster_size_parts;
    }

    set_clusters(machines_clusters, machines, machines_clusters_indices, i,
                 machines_clusters.size());
    set_clusters(parts_clusters, parts, parts_clusters_indices, j,
                 parts_clusters.size());

#ifdef DEBUG
    std::cout << "machines: " << std::endl;
    for (std::size_t index = 0; index < machines_clusters.size(); ++index) {
      std::cout << machines_clusters[index] << " ";
    }
#endif

#ifdef DEBUG
    std::cout << "\n\nparts: " << std::endl;
    for (std::size_t index = 0; index < parts_clusters.size(); ++index) {
      std::cout << parts_clusters[index] << " ";
    }
#endif
  }

  double print_shit() {
    // 1 в решении/(всего 1 + нулей в решении)
    auto test = matrix;

    std::size_t ones = 0;
    std::size_t zeros_in_solution = 0;
    std::size_t ones_in_solution = 0;

    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
      auto &machines = machines_clusters_indices[cluster];
      auto &parts = parts_clusters_indices[cluster];

      for (std::size_t i : machines)
        for (std::size_t j : parts) {
          ones_in_solution += test[i][j].value;
          zeros_in_solution += (test[i][j].value == 0) ? 1 : 0;
          test[i][j].value = cluster + 2;
        }
    }

    for (std::size_t i = 0; i < matrix.size(); ++i) {
      for (std::size_t j = 0; j < matrix[0].size(); ++j) {
        ones += matrix[i][j].value;
#ifdef DEBUG
        std::cout << test[i][j].value << " ";
#endif
      }
#ifdef DEBUG
      std::cout << std::endl;
#endif
    }
    double f = ones_in_solution / double(ones + zeros_in_solution);
    // std::cout << f << std::endl;
    return f;
  }

public:
  std::int64_t ones_overall = 0;

  std::vector<std::size_t> machines_clusters;
  std::vector<std::size_t> parts_clusters;

  std::vector<std::vector<std::size_t>> machines_clusters_indices;
  std::vector<std::vector<std::size_t>> parts_clusters_indices;

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

  // std::cout << std::endl << std::endl;
  static constexpr auto ITERATIONS = 1'000'000;
  double maximum = std::numeric_limits<double>::min();
  auto matrix = biclustering_solver.matrix;
  for (std::size_t index = 0; index < ITERATIONS; ++index) {
    biclustering_solver.clear();
    biclustering_solver.parse(input_files[i]);
    biclustering_solver.initial_random();
    auto evaluation = biclustering_solver.print_shit();
    if (evaluation > maximum) {
      std::cout << maximum << std::endl;
      maximum = evaluation;
    }
  }
  std::cout << "max: " << maximum;

  biclustering_solver.print_shit();
}