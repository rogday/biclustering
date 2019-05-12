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
#include <iomanip>
#include <functional>

/*
Params to tune:
- number of shakings
- total iter
- clustering principle
- shaking cut distance
- number of shakings sequentially
*/

class genetic_t {
public:
  using gene_t = std::uint64_t;
  using chromosome_t = std::vector<gene_t>;
  using population_t = std::vector<chromosome_t>;

  struct settings_t {
    std::uint64_t n;         // amount of genes
    std::uint64_t mutatable; // mutable genes
    std::uint64_t population_size;

    chromosome_t restrictions;

    double new_population_percentage;
    double mutation_probability;
  };

private:
  std::function<double(chromosome_t &, std::string &)> fitness;
  settings_t settings;
  std::string file;

  static std::mt19937 generator;

public:
  genetic_t(std::function<double(chromosome_t &, std::string &)> fitness,
            settings_t settings, std::string file)
      : fitness(fitness), settings(settings), file(file) {}

  void run(std::size_t iterations) {
    auto population = initial_population();
    for (int i = 0; i < iterations; ++i) {
      std::cout << "degeneration " << i << std::endl;
      derivative_population(population);
    }
  }

private:
  template <typename T> static T random(T lower_bound, T upper_bound) {
    std::uniform_int_distribution<T> distribution(lower_bound, upper_bound);
    return distribution(generator);
  }

  gene_t mutator(std::size_t i) {
    return random<gene_t>(0, settings.restrictions[i]);
  }

  chromosome_t random_chromosome() {
    chromosome_t chromosome(settings.n);
    for (std::size_t i = 0; i < std::size(chromosome); ++i)
      chromosome[i] = mutator(i);
    return chromosome;
  }

  chromosome_t crossover(chromosome_t &mammy, chromosome_t &daddy) {
    std::uint64_t index = random(static_cast<std::uint64_t>(1), settings.n - 1);

    chromosome_t child(mammy);
    for (std::size_t i = index; i < settings.n; ++i)
      std::swap(child[i], daddy[i]);

    return child;
  }

  chromosome_t mutation(chromosome_t &&chromosome) {
    std::vector<std::size_t> indices(settings.n);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::shuffle(std::begin(indices), std::end(indices), generator);

    for (std::size_t i = 0; i < settings.mutatable; ++i)
      chromosome[indices[i]] = mutator(indices[i]);
    return chromosome;
  }

  population_t initial_population() {
    population_t population(settings.population_size);
    for (auto &chromosome : population)
      chromosome = random_chromosome();
    return population;
  }

  std::vector<double> calc_fitness(population_t &population) {
    std::vector<double> ret(std::size(population));
    for (std::size_t i = 0; i < std::size(ret); ++i)
      ret[i] = fitness(population[i], file);
    return ret;
  }

  void derivative_population(population_t &population) {
    auto fitness = calc_fitness(population);

    double sum = std::accumulate(std::begin(fitness), std::end(fitness), 0.0);
    std::for_each(std::begin(fitness), std::end(fitness),
                  [sum](double &value) { return value / sum; });

    auto loss = fitness;
    std::for_each(std::begin(loss), std::end(loss),
                  [sum](double &value) { return 1.0 - value; });

    std::discrete_distribution<std::size_t> fitness_indices(std::begin(fitness),
                                                            std::end(fitness));
    std::discrete_distribution<std::size_t> loss_indices(std::begin(loss),
                                                         std::end(loss));

    std::vector<chromosome_t> parents(
        2 * static_cast<std::size_t>(settings.n *
                                     settings.new_population_percentage));

    for (std::size_t i = 0; i < std::size(parents); ++i)
      parents[i] = population[fitness_indices(generator)];

    for (std::size_t i = 0; i < std::size(parents); i += 2) {
      auto child = mutation(crossover(parents[i], parents[i + 1]));
      population[loss_indices(generator)] = child;
    }
  }
};

std::mt19937 genetic_t::generator{std::random_device{}()};

constexpr std::uint64_t NOT_IN_CLUSTER = 0;
constexpr double eps = 1e-12;

namespace utility {
enum genes_t {
  SHAKINGS = 0,
  ITERATIONS,
  CLUSTERING_PRINCIPLE,
  SHAKINGS_IN_A_ROW,
  SIZE
};

std::string filename_from_path(std::string &path) {
  return std::filesystem::path(path).stem().string();
}

static std::mt19937 generator{std::random_device{}()};

auto generate_random_value(
    std::size_t lower_bound = std::numeric_limits<std::size_t>::min(),
    std::size_t upper_bound = std::numeric_limits<std::size_t>::max()) {

  std::uniform_int_distribution<std::size_t> distribution(lower_bound,
                                                          upper_bound);
  return distribution(generator);
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

std::vector<std::int64_t> split(std::string const &str) {
  std::istringstream iss(str);
  return {std::istream_iterator<std::int64_t>{iss},
          std::istream_iterator<std::int64_t>{}};
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig) {
  std::vector<T> ret;
  for (const auto &v : orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

} // namespace utility

class biclustering_solver_t {
  using entry_t = std::vector<std::string>;
  using indices_t = std::vector<std::size_t>;
  template <typename T> using matrix_t = std::vector<std::vector<T>>;

  template <typename T>
  void construct_matrix(matrix_t<T> &mat, std::size_t n, std::size_t m) {
    mat.resize(n);
    std::fill(std::begin(mat), std::end(mat), std::vector<T>(m));
  }

public:
  biclustering_solver_t(genetic_t::chromosome_t &settings)
      : settings(settings){};

  biclustering_solver_t(biclustering_solver_t const &) = delete;
  biclustering_solver_t(biclustering_solver_t &&) = delete;

  void parse(std::string const &path_) {
    full_clear();
    path = path_;

    std::fstream filestream{path};
    std::string line;

    /** M, P (number of machines and parts) */
    std::getline(filestream, line);
    auto dimensions = utility::split(line);
    n = dimensions[0];
    m = dimensions[1];

    construct_matrix(matrix, n, m);

    while (std::getline(filestream, line)) {
      auto vector = utility::split(line);
      ones_overall += vector.size() - 1;
      for (std::size_t i = 1; i < vector.size(); ++i)
        matrix[vector[0] - 1][vector[i] - 1] = true;
    }
  }

  void full_clear() {
    clusters = ones_overall = n = m = 0;
    machines_clusters.clear();
    parts_clusters.clear();
    matrix.clear();
  }

  void random_clear() {
    for (auto &v : machines_clusters)
      v.clear();

    for (auto &v : parts_clusters)
      v.clear();
  }

  void initial_random() {
    auto ITERATIONS = settings[utility::ITERATIONS];

    std::size_t new_clusters;

    matrix_t<std::size_t> new_machines_clusters;
    matrix_t<std::size_t> new_parts_clusters;

    double max = std::numeric_limits<double>::lowest();

    for (std::size_t index = 0; index < ITERATIONS; ++index) {
      random_clear();
      random_pass();
      double evaluation = loss();
      if (evaluation > max) {
        new_machines_clusters = machines_clusters;
        new_parts_clusters = parts_clusters;

        new_clusters = clusters;

        max = evaluation;
        // std::cout << index << " " << max << std::endl;
      }
    }

    clusters = new_clusters;

    machines_clusters = new_machines_clusters;
    parts_clusters = new_parts_clusters;

    // std::cout << "clusters: " << clusters << std::endl;
  }

  void random_pass() {
    std::size_t i = 0; // i - сколько уже сгенерили для машин
    std::size_t j = 0; // j - сколько уже сгенерили для частей

    std::size_t cluster_id = 1;
    clusters = utility::generate_random_value(1ull, n);

    machines_clusters.resize(clusters);
    parts_clusters.resize(clusters);

    auto set_clusters = [&cluster_id](auto &outer, auto &inner,
                                      std::size_t start, std::size_t end) {
      for (std::size_t index = start; index < end; ++index)
        outer[cluster_id - 1].push_back(inner[index]);
    };

    auto prepare_indices = [](auto &container, std::size_t size) {
      container.resize(size);
      std::iota(std::begin(container), std::end(container), 0);
      std::shuffle(std::begin(container), std::end(container),
                   utility::generator);
    };

    indices_t machines;
    indices_t parts;

    prepare_indices(machines, n);
    prepare_indices(parts, m);

    std::normal_distribution dist_machines(n / double(clusters),
                                           n / double(clusters));

    std::normal_distribution dist_parts(m / double(clusters),
                                        m / double(clusters));

    for (std::size_t cluster = 1; cluster < clusters; ++cluster, ++cluster_id) {

      std::size_t cluster_size_machines = std::clamp(
          std::uint64_t(std::round(dist_machines(utility::generator))), 1ull,
          n - (clusters - cluster) - i);

      std::size_t cluster_size_parts =
          std::clamp(std::uint64_t(std::round(dist_parts(utility::generator))),
                     1ull, m - (clusters - cluster) - j);

      set_clusters(machines_clusters, machines, i, i + cluster_size_machines);
      set_clusters(parts_clusters, parts, j, j + cluster_size_parts);

      i += cluster_size_machines;
      j += cluster_size_parts;
    }

    set_clusters(machines_clusters, machines, i, n);
    set_clusters(parts_clusters, parts, j, m);
  }

  double loss() {
    std::size_t zeros_in_solution = 0;
    std::size_t ones_in_solution = 0;

    for (std::size_t cluster = 0; cluster < clusters; ++cluster)
      for (std::size_t x : machines_clusters[cluster])
        for (std::size_t y : parts_clusters[cluster]) {
          ones_in_solution += matrix[x][y];
          zeros_in_solution += !matrix[x][y];
        }

    return ones_in_solution / double(ones_overall + zeros_in_solution);
  }

  void optimize() {
    auto neighbors = settings[utility::SHAKINGS];

    double max_loss = loss(), last_loss;

    auto old_machines_clusters = machines_clusters;
    auto old_parts_clusters = parts_clusters;

    auto new_machines_clusters = machines_clusters;
    auto new_parts_clusters = parts_clusters;

    auto save_solution = [&](bool flag = false) {
      (flag ? new_machines_clusters : old_machines_clusters) =
          machines_clusters;
      (flag ? new_parts_clusters : old_parts_clusters) = parts_clusters;
    };

    auto set_current = [&](bool flag) {
      machines_clusters =
          (flag ? new_machines_clusters : old_machines_clusters);
      parts_clusters = (flag ? new_parts_clusters : old_parts_clusters);
      clusters = parts_clusters.size();
    };

    auto restore_original = std::bind(set_current, false);

    std::vector<std::function<void()>> shakers;
    shakers.push_back(std::bind(&biclustering_solver_t::split_pass, this));
    shakers.push_back(std::bind(&biclustering_solver_t::merge_pass, this));

    std::size_t iterations = 50;
    bool flag;

    do {
      last_loss = max_loss;
      flag = false;

      save_solution(false);

      for (std::size_t i = 0; i < neighbors; ++i) {
        for (auto &shaker : shakers) {
          shaker();
          local_search();

          double current_loss = loss();
          if (current_loss > max_loss) {
            max_loss = current_loss;
            flag = true;
            save_solution(true);
          }

          restore_original();
        }
      }

      set_current(flag);
    } while (1.0 - last_loss / max_loss > eps || iterations--);

    std::cout << "clusters: " << clusters << ", loss: " << max_loss
              << ", test: " << loss() << std::endl;
  }

  void local_search() {
    double max_loss = std::numeric_limits<double>::lowest(), last_loss;

    matrix_t<std::size_t> *first = &machines_clusters;
    matrix_t<std::size_t> *second = &parts_clusters;

    if (settings[utility::CLUSTERING_PRINCIPLE] != 0)
      std::swap(first, second);

    do {
      last_loss = max_loss;

      swap(*first);
      swap(*second);

      max_loss = std::max(max_loss, loss());
    } while (1.0 - last_loss / max_loss > eps);
  }

  void swap(matrix_t<std::size_t> &clusters_vector) {
    double max_loss = loss(), last_loss, current_loss;

    std::uint64_t *i, *j;
    do {
      last_loss = max_loss;
      i = j = nullptr;

      for (std::size_t cluster1 = 0; cluster1 < clusters; ++cluster1)
        for (std::size_t cluster2 = cluster1 + 1; cluster2 < clusters;
             ++cluster2)
          for (auto &x : clusters_vector[cluster1])
            for (auto &y : clusters_vector[cluster2]) {

              std::swap(x, y);

              current_loss = loss();
              if (current_loss > max_loss) {
                i = &x;
                j = &y;
                max_loss = current_loss;
              }

              std::swap(x, y);
            }

      if (i != nullptr)
        std::swap(*i, *j);

    } while (1.0 - last_loss / max_loss > eps);
  }

  void split_pass() {
    std::int64_t cluster = utility::generate_random_value(0, clusters - 1);

    if (machines_clusters[cluster].size() > 1 &&
        parts_clusters[cluster].size() > 1) {
      std::int64_t x = utility::generate_random_value(
          1, machines_clusters[cluster].size() - 1);

      std::int64_t y =
          utility::generate_random_value(1, parts_clusters[cluster].size() - 1);

      split_apply(cluster, x, y);
    }
  }

  void split_apply(std::size_t cluster, std::size_t x, std::size_t y) {
    split(cluster, x, y);
    ++clusters;
  }

  void merge_pass() {

    if (clusters < 2)
      return;

    std::int64_t cluster1 = utility::generate_random_value(0, clusters - 2);
    std::int64_t cluster2 =
        utility::generate_random_value(cluster1 + 1, clusters - 1);

    merge_apply(cluster1, cluster2);
  }

  void merge_apply(std::size_t cluster1, std::size_t cluster2) {
    merge(cluster1, cluster2);
    std::swap(machines_clusters[cluster2], machines_clusters.back());
    std::swap(parts_clusters[cluster2], parts_clusters.back());
    machines_clusters.pop_back();
    parts_clusters.pop_back();
    --clusters;
  }

  void merge(std::size_t cluster1, std::size_t cluster2) {
    utility::append_clear(machines_clusters[cluster1],
                          machines_clusters[cluster2]);
    utility::append_clear(parts_clusters[cluster1], parts_clusters[cluster2]);
  }

  void split(std::size_t cluster, std::size_t i, std::size_t j) {
    utility::split(machines_clusters[cluster], machines_clusters.emplace_back(),
                   i);
    utility::split(parts_clusters[cluster], parts_clusters.emplace_back(), j);
  }

  void read_solution(std::string &path) {
    std::string answer_file_name =
        "../output/" + utility::filename_from_path(path) + ".sol";
    std::fstream file{answer_file_name, std::ios_base::in};

    full_clear();
    parse(path);

    auto read_vector = [&file](auto &vector) {
      std::string line;
      std::getline(file, line);
      std::istringstream iss(line);

      std::size_t cluster, index = 0;
      while (iss >> cluster) {
        vector.resize(std::max(cluster + 1, vector.size()));
        vector[cluster].push_back(index++);
      }
    };

    read_vector(machines_clusters);
    read_vector(parts_clusters);
    clusters = parts_clusters.size();

    file.close();
  }

  void save_data() {
    std::ofstream fstream{"../output/" + utility::filename_from_path(path) +
                              ".sol",
                          std::ios_base::out | std::ios_base::trunc};

    std::vector<std::size_t> machines(n), parts(m);

    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
      for (auto x : machines_clusters[cluster])
        machines[x] = cluster;
      for (auto y : parts_clusters[cluster])
        parts[y] = cluster;
    }

    for (auto &x : machines)
      fstream << x << " ";

    fstream << std::endl;

    for (auto &y : parts)
      fstream << y << " ";

    fstream.close();
  }

private:
  genetic_t::chromosome_t settings;

  std::int64_t ones_overall = 0;

  matrix_t<bool> matrix;

  std::size_t n;
  std::size_t m;

  std::size_t clusters;
  matrix_t<std::size_t> machines_clusters;
  matrix_t<std::size_t> parts_clusters;

  std::string path;
};

auto get_files(std::string path) {
  std::vector<std::string> files;
  for (auto entry : std::filesystem::directory_iterator(path))
    files.push_back(entry.path().string());
  return files;
}

int main() {
  using namespace utility;

  genetic_t::settings_t settings;
  settings.n = SIZE;
  settings.mutatable = static_cast<std::uint64_t>(settings.n * 0.3);
  settings.population_size = 15;
  settings.new_population_percentage = 0.7;
  settings.mutation_probability = 0.2;

  genetic_t::chromosome_t restrictions(settings.n);
  restrictions[SHAKINGS] = 20;
  restrictions[ITERATIONS] = 50'000;
  restrictions[CLUSTERING_PRINCIPLE] = 1;
  restrictions[SHAKINGS_IN_A_ROW] = 10;

  settings.restrictions = restrictions;

  auto input_files = get_files("../input");

  auto fitness = [](genetic_t::chromosome_t &chromosome, std::string &file) {
    biclustering_solver_t biclustering_solver(chromosome);
    biclustering_solver.parse(file);
    biclustering_solver.initial_random();
    biclustering_solver.optimize();
    return biclustering_solver.loss();
  };

  genetic_t genetic_algo(fitness, settings, input_files[0]);
  genetic_algo.run(20);

  return 0;
}