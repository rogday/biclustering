#include <biclustering.hpp>

template <typename T>
void biclustering_solver_t::construct_matrix(matrix_t<T> &mat, std::size_t n,
                                             std::size_t m) {
  mat.resize(n);
  std::fill(std::begin(mat), std::end(mat), std::vector<T>(m));
}

void biclustering_solver_t::parse(std::string const &path_) {
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

void biclustering_solver_t::full_clear() {
  clusters = ones_overall = n = m = 0;
  machines_clusters.clear();
  parts_clusters.clear();
  matrix.clear();
  cluster_matrix.clear();
}

void biclustering_solver_t::initial_random() {
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

  construct_cluster_matrix();
  // std::cout << "clusters: " << clusters << std::endl;
}

void biclustering_solver_t::optimize() {
  auto neighbors = settings[utility::SHAKINGS];

  double max_loss = loss(), last_loss;

  auto old_machines_clusters = machines_clusters;
  auto old_parts_clusters = parts_clusters;

  auto new_machines_clusters = machines_clusters;
  auto new_parts_clusters = parts_clusters;

  auto save_solution = [&](bool flag = false) {
    (flag ? new_machines_clusters : old_machines_clusters) = machines_clusters;
    (flag ? new_parts_clusters : old_parts_clusters) = parts_clusters;
  };

  auto set_current = [&](bool flag) {
    machines_clusters = (flag ? new_machines_clusters : old_machines_clusters);
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
  } while (1.0 - last_loss / max_loss > utility::eps || iterations--);

  construct_cluster_matrix();
  // std::cout << "clusters: " << clusters << ", loss: " << max_loss <<
  // std::endl;
}

void biclustering_solver_t::random_clear() {
  for (auto &v : machines_clusters)
    v.clear();

  for (auto &v : parts_clusters)
    v.clear();
}

double biclustering_solver_t::loss() {
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

void biclustering_solver_t::read_solution(std::string &path) {
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

  construct_cluster_matrix();
  file.close();
}

void biclustering_solver_t::save_data() {
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

void biclustering_solver_t::random_pass() {
  std::size_t i = 0; // i - сколько уже сгенерили для машин
  std::size_t j = 0; // j - сколько уже сгенерили для частей

  std::size_t cluster_id = 1;
  clusters = utility::generate_random_value(1ull, n);

  machines_clusters.resize(clusters);
  parts_clusters.resize(clusters);

  auto set_clusters = [&cluster_id](auto &outer, auto &inner, std::size_t start,
                                    std::size_t end) {
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

    std::size_t cluster_size_machines =
        std::clamp(std::uint64_t(std::round(dist_machines(utility::generator))),
                   1ull, n - (clusters - cluster) - i);

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

void biclustering_solver_t::local_search() {
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
  } while (1.0 - last_loss / max_loss > utility::eps);
}

void biclustering_solver_t::swap(matrix_t<std::size_t> &clusters_vector) {
  double max_loss = loss(), last_loss, current_loss;

  std::uint64_t *i, *j;
  do {
    last_loss = max_loss;
    i = j = nullptr;

    for (std::size_t cluster1 = 0; cluster1 < clusters; ++cluster1)
      for (std::size_t cluster2 = cluster1 + 1; cluster2 < clusters; ++cluster2)
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

  } while (1.0 - last_loss / max_loss > utility::eps);
}

void biclustering_solver_t::split_pass() {
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

void biclustering_solver_t::split_apply(std::size_t cluster, std::size_t x,
                                        std::size_t y) {
  split(cluster, x, y);
  ++clusters;
}

void biclustering_solver_t::merge_pass() {

  if (clusters < 2)
    return;

  std::int64_t cluster1 = utility::generate_random_value(0, clusters - 2);
  std::int64_t cluster2 =
      utility::generate_random_value(cluster1 + 1, clusters - 1);

  merge_apply(cluster1, cluster2);
}

void biclustering_solver_t::merge_apply(std::size_t cluster1,
                                        std::size_t cluster2) {
  merge(cluster1, cluster2);
  std::swap(machines_clusters[cluster2], machines_clusters.back());
  std::swap(parts_clusters[cluster2], parts_clusters.back());
  machines_clusters.pop_back();
  parts_clusters.pop_back();
  --clusters;
}

void biclustering_solver_t::merge(std::size_t cluster1, std::size_t cluster2) {
  utility::append_clear(machines_clusters[cluster1],
                        machines_clusters[cluster2]);
  utility::append_clear(parts_clusters[cluster1], parts_clusters[cluster2]);
}

void biclustering_solver_t::split(std::size_t cluster, std::size_t i,
                                  std::size_t j) {
  utility::split(machines_clusters[cluster], machines_clusters.emplace_back(),
                 i);
  utility::split(parts_clusters[cluster], parts_clusters.emplace_back(), j);
}

void biclustering_solver_t::construct_cluster_matrix() {
  utility::construct_matrix(cluster_matrix, n, m); // clear

  for (std::size_t cluster = 0; cluster < clusters; ++cluster)
    for (std::size_t x : machines_clusters[cluster])
      for (std::size_t y : parts_clusters[cluster])
        cluster_matrix[x][y] = cluster + 1;

  xs = utility::flatten(machines_clusters);
  ys = utility::flatten(parts_clusters);
}