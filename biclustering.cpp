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
    parts_clusters.resize(dimensions[1]);

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
    std::size_t clusters = 1 + prng() % machines_clusters.size();

    std::cout << "clusters: " << clusters << std::endl << std::endl;

    for (std::size_t cluster = 1; cluster < clusters; ++cluster, ++cluster_id) {

      std::cout << (machines_clusters.size() - (clusters - cluster) - i) << " "
                << (parts_clusters.size() - (clusters - cluster) - j)
                << std::endl;

      std::size_t cluster_size_machines =
          1 + prng() % (machines_clusters.size() - (clusters - cluster) - i);

      std::size_t cluster_size_parts =
          1 + prng() % (parts_clusters.size() - (clusters - cluster) - j);

      std::cout << "cluster_size_machines: " << cluster_size_machines
                << ", cluster_size_parts:" << cluster_size_parts << std::endl;

      for (std::size_t x = i; x < i + cluster_size_machines; ++x)
        machines_clusters[machines[x]] = cluster_id;

      for (std::size_t y = j; y < j + cluster_size_parts; ++y)
        parts_clusters[parts[y]] = cluster_id;

      i += cluster_size_machines;
      j += cluster_size_parts;

      std::cout << "i: " << i << ", j: " << j << std::endl;
    }

    for (std::size_t x = i; x < machines_clusters.size(); ++x)
      machines_clusters[machines[x]] = cluster_id;

    for (std::size_t y = j; y < parts_clusters.size(); ++y)
      parts_clusters[parts[y]] = cluster_id;

    std::cout << "machines: " << std::endl;
    for (std::size_t index = 0; index < machines_clusters.size(); ++index) {
      std::cout << machines_clusters[index] << " ";
    }

    std::cout << "\n\nparts: " << std::endl;
    for (std::size_t index = 0; index < parts_clusters.size(); ++index) {
      std::cout << parts_clusters[index] << " ";
    }
  }

private:
  std::int64_t ones_overall = 0;

  std::vector<std::size_t> machines_clusters;
  std::vector<std::size_t> parts_clusters;

  two_dim_matrix_t matrix;
  two_dim_matrix_t greedy_matrix;

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

  biclustering_solver.initial_random();
}