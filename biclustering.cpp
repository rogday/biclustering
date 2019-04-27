/*#include <iostream>
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

#include <SFML/Graphics.hpp>

constexpr std::uint64_t NOT_IN_CLUSTER = 0;
constexpr double eps = 1e-12;

using args_t = std::tuple<std::int64_t, std::int64_t, std::int64_t>;

namespace utility {
auto to_vector(args_t &args) {
  std::vector<std::int64_t> v;
  v.push_back(std::get<0>(args));
  v.push_back(std::get<1>(args));
  v.push_back(std::get<2>(args));
  return v;
}

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
} // namespace utility

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

; // namespace utility

class biclustering_solver_t {

  using entry_t = std::vector<std::string>;
  using indices_t = std::vector<std::size_t>;
  template <typename T> using matrix_t = std::vector<std::vector<T>>;

  std::vector<std::int64_t> split(std::string const &str) {
    std::istringstream iss(str);
    return {std::istream_iterator<std::int64_t>{iss},
            std::istream_iterator<std::int64_t>{}};
  }

  template <typename T>
  void construct_matrix(matrix_t<T> &mat, std::size_t n, std::size_t m) {
    mat.resize(n);
    std::fill(std::begin(mat), std::end(mat), std::vector<T>(m));
  }

public:
  biclustering_solver_t(sf::RenderWindow &window) : window(window){};
  biclustering_solver_t(biclustering_solver_t const &) = delete;
  biclustering_solver_t(biclustering_solver_t &&) = delete;

  void parse(std::string const &filename) {
    full_clear();

    std::fstream filestream{filename};
    std::string line;

    // M, P (number of machines and parts)
    std::getline(filestream, line);
    auto dimensions = split(line);
    n = dimensions[0];
    m = dimensions[1];

    construct_matrix(matrix, n, m);
    construct_matrix(cluster_matrix, n, m);

    while (std::getline(filestream, line)) {
      auto vector = split(line);
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
    cluster_matrix.clear();
  }

  void random_clear() {
    for (auto &v : machines_clusters)
      v.clear();

    for (auto &v : parts_clusters)
      v.clear();
  }

  void initial_random() {
    static constexpr auto ITERATIONS = 10'000;

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
        std::cout << index << " " << max << std::endl;
      }
    }

    clusters = new_clusters;

    machines_clusters = new_machines_clusters;
    parts_clusters = new_parts_clusters;

    construct_cluster_matrix();

    std::cout << "clusters: " << clusters << std::endl;
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

  void construct_cluster_matrix() {
    construct_matrix(cluster_matrix, n, m); // clear
    for (std::size_t cluster = 0; cluster < clusters; ++cluster)
      for (std::size_t x : machines_clusters[cluster])
        for (std::size_t y : parts_clusters[cluster])
          cluster_matrix[x][y] = cluster + 1;
  }

  double loss() { return loss(clusters); }

  double loss(std::size_t clusters_) {
    std::size_t zeros_in_solution = 0;
    std::size_t ones_in_solution = 0;

    for (std::size_t cluster = 0; cluster < clusters_; ++cluster)
      for (std::size_t x : machines_clusters[cluster])
        for (std::size_t y : parts_clusters[cluster]) {
          ones_in_solution += matrix[x][y];
          zeros_in_solution += !matrix[x][y];
        }

    return ones_in_solution / double(ones_overall + zeros_in_solution);
  }

  void optimize() {
    constexpr std::size_t neighbors = 10;

    double max_loss = loss(), last_loss;

    auto new_machines_clusters = machines_clusters;
    auto new_parts_clusters = parts_clusters;

    matrix_t<std::int64_t> solutions;
    std::vector<double> losses;

    auto save_solution = [this, &solutions](std::vector<std::int64_t> v = {}) {
      solutions.emplace_back(v);
    };

    auto set_current = [this, &solutions, &new_machines_clusters,
                        &new_parts_clusters](std::size_t i) {
      switch (solutions[i].size()) {
      case 0: // original
        machines_clusters = new_machines_clusters;
        parts_clusters = new_parts_clusters;
        break;
      case 2: // merge
        merge_pass({solutions[i][0], solutions[i][1], -1});
        break;
      case 3: // split
        split_pass({solutions[i][0], solutions[i][1], solutions[i][2]});
        break;

      default:
        break;
      }

      new_machines_clusters = machines_clusters;
      new_parts_clusters = parts_clusters;
      clusters = machines_clusters.size();
    };

    auto restore_original = std::bind(set_current, 0);

    auto clear = [&solutions]() { solutions.clear(); };

    std::vector<std::function<std::vector<std::int64_t>(
        std::tuple<std::int64_t, std::int64_t, std::int64_t>)>>
        shakers;
    shakers.push_back(std::bind(&biclustering_solver_t::split_pass, this,
                                std::placeholders::_1));
    shakers.push_back(std::bind(&biclustering_solver_t::merge_pass, this,
                                std::placeholders::_1));

    std::size_t index, iterations = 5'000;

    do {
      last_loss = max_loss;
      index = 0;

      save_solution();

      for (std::size_t i = 0; i < neighbors; ++i) {
        for (auto &shaker : shakers) {
          auto v = shaker({-1, -1, -1});
          local_search();
          save_solution(v);
          restore_original();
        }
      }

      index = std::distance(losses.begin(),
                            std::max_element(losses.begin(), losses.end())) %
              losses.size();

      max_loss = losses[index];
      set_current(index);

      clear();

    } while (iterations--);
    // while (1.0 - last_loss / max_loss > eps);

    construct_cluster_matrix();
    std::cout << "clusters: " << clusters << ", loss: " << max_loss
              << ", test: " << loss() << std::endl;
  }

  void local_search() {
    double max_loss = std::numeric_limits<double>::lowest(), last_loss;

    do {
      last_loss = max_loss;

      swap<false>();
      swap<true>();

      max_loss = std::max(max_loss, loss());
    } while (1.0 - last_loss / max_loss > eps);
  }

  template <bool Machines> void swap() {
    double max_loss = loss(), last_loss, current_loss;

    std::int64_t new_cluster1, new_cluster2, i, j;

    std::reference_wrapper<matrix_t<std::size_t>> clusters_vector =
        machines_clusters;

    if constexpr (!Machines)
      clusters_vector = parts_clusters;

    do {
      last_loss = max_loss;
      i = j = new_cluster1 = new_cluster2 = -1;

      for (std::size_t cluster1 = 0; cluster1 < clusters; ++cluster1) {

        if (clusters_vector.get()[cluster1].size() == 1)
          continue;

        for (std::size_t cluster2 = cluster1 + 1; cluster2 < clusters;
             ++cluster2) {

          if (clusters_vector.get()[cluster2].size() == 1)
            continue;

          for (std::size_t x = 0; x < clusters_vector.get()[cluster1].size();
               ++x)
            for (std::size_t y = 0; y < clusters_vector.get()[cluster2].size();
                 ++y) {

              std::swap(clusters_vector.get()[cluster1][x],
                        clusters_vector.get()[cluster2][y]);

              current_loss = loss();
              if (current_loss > max_loss) {
                i = x;
                j = y;
                new_cluster1 = cluster1;
                new_cluster2 = cluster2;

                max_loss = current_loss;
              }

              std::swap(clusters_vector.get()[cluster1][x],
                        clusters_vector.get()[cluster2][y]);
            }
        }
      }

      if (i != -1)
        std::swap(clusters_vector.get()[new_cluster1][i],
                  clusters_vector.get()[new_cluster2][j]);

    } while (1.0 - last_loss / max_loss > eps);
  }

  std::vector<std::int64_t>
  split_pass(std::tuple<std::int64_t, std::int64_t, std::int64_t> args) {
    auto [cluster, x, y] = args;

    if (cluster == -1 || x == -1 || y == -1) {
      cluster = utility::generate_random_value(0, clusters - 1);

      if (machines_clusters[cluster].size() > 1 &&
          parts_clusters[cluster].size() > 1) {
        x = utility::generate_random_value(
            1, machines_clusters[cluster].size() - 1);

        y = utility::generate_random_value(1,
                                           parts_clusters[cluster].size() - 1);
      }
      return {};
    }

    split_apply(cluster, x, y);
    return {cluster, x, y};
  }

  void split_apply(std::size_t cluster, std::size_t x, std::size_t y) {
    split(cluster, x, y);
    ++clusters;
  }

  std::vector<std::int64_t>
  merge_pass(std::tuple<std::int64_t, std::int64_t, std::int64_t> args) {
    auto [cluster1, cluster2, _] = args;

    if (clusters < 2)
      return {};

    if (cluster1 == -1 || cluster2 == -1) {
      cluster1 = utility::generate_random_value(0, clusters - 2);
      cluster2 = utility::generate_random_value(cluster1 + 1, clusters - 1);
    }

    merge_apply(cluster1, cluster2);
    return {cluster1, cluster2};
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

  void read_solution(std::string &file) {}

  void check_solutions(std::vector<std::string> &files) {}

  void save_data() {}

  void append_loss() {}

  void draw(bool new_colors = false) {
    static std::random_device rd;
    static unsigned int seed = rd();

    seed = (new_colors) ? rd() : seed;

    std::mt19937 prng(seed);

    float size_i = window.getSize().x / float(m);
    float size_k = window.getSize().y / float(n);

    window.clear(sf::Color::Black);

    float radius = 2.5;
    std::vector<sf::Color> colors(clusters + 1);
    colors[0] = sf::Color::White;

    for (std::size_t i = 1; i <= clusters; ++i) {
      int r = prng() % 256, g = prng() % 256, b = prng() % 256;
      colors[i] = sf::Color(r, g, b);
    }

    for (int i = 0; i < n; ++i)
      for (int k = 0; k < m; ++k) {
        std::size_t cluster_id = cluster_matrix[i][k];

        sf::RectangleShape rect(sf::Vector2f(size_i, size_k));

        rect.setPosition(size_i * k, size_k * i);

        rect.setOutlineColor(sf::Color::Black);
        rect.setOutlineThickness(1);

        rect.setFillColor(colors[cluster_id]);
        window.draw(rect);

        if (matrix[i][k]) {
          sf::CircleShape circle;

          circle.setFillColor(sf::Color::Black);
          circle.setRadius(radius);
          circle.setPosition(size_i * k - radius + size_i / 2,
                             size_k * i - radius + size_k / 2);
          window.draw(circle);
        }
      }

    window.display();
  }

private:
  std::int64_t ones_overall = 0;

  matrix_t<bool> matrix;
  matrix_t<std::size_t> cluster_matrix;

  std::size_t n;
  std::size_t m;

  std::size_t clusters; //ору
  matrix_t<std::size_t> machines_clusters;
  matrix_t<std::size_t> parts_clusters;

  sf::RenderWindow &window;

  std::string filename;
};

auto get_files(std::string path) {
  std::vector<std::string> files;
  for (auto entry : std::filesystem::directory_iterator(path))
    files.push_back(entry.path().string());
  return files;
}

void print_choice(std::vector<std::string> &map) {
  std::size_t i = -1;
  std::cout << std::endl;
  while (++i != map.size())
    std::cout << "#" << i << ": " << map[i] << std::endl;
  std::cout << std::endl;
}

sf::RenderWindow &init_window(double size) {
  sf::VideoMode vm = sf::VideoMode::getDesktopMode();
  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;

  static sf::RenderWindow window(
      sf::VideoMode(int(vm.width / size), int(vm.height / size)),
      "Biclustering",
      sf::Style::Titlebar | sf::Style::Close | sf::Style::Resize, settings);

  window.setPosition(sf::Vector2i(vm.width / 2 - vm.width / (size * 2),
                                  vm.height / 2 - vm.height / (size * 2)));

  window.setKeyRepeatEnabled(true);
  window.setVerticalSyncEnabled(true);

  return window;
}

int main(int argc, char *argv[]) {
  auto &window = init_window(1.5);

  biclustering_solver_t biclustering_solver(window);
  auto input_files = get_files("../input");

  biclustering_solver.parse(input_files[0]);
  biclustering_solver.initial_random();

  bool show_solution = false;

  sf::Event event;
  while (window.isOpen()) {
    biclustering_solver.draw();

    while (window.pollEvent(event)) {
      switch (event.type) {

      case sf::Event::KeyPressed:
        if (event.key.code == sf::Keyboard::Escape) { //  close
          window.close();
        }
        if (event.key.code == sf::Keyboard::Space) { // change colors
          biclustering_solver.draw(true);
        } else if (event.key.code == sf::Keyboard::O) { // optimize
          biclustering_solver.optimize();
          std::cout << "done" << std::endl;
        } else if (event.key.code == sf::Keyboard::P) { // print
          print_choice(input_files);
        } else if (event.key.code == sf::Keyboard::H) { // filp show solution
          show_solution ^= true;
        } else if (event.key.code == sf::Keyboard::C) { // check sols
          biclustering_solver.check_solutions(input_files);
        } else if (event.key.code == sf::Keyboard::S) { // save
          biclustering_solver.save_data();
          biclustering_solver.append_loss();
          std::cout << "saved" << std::endl;
        } else if (event.key.code >= sf::Keyboard::Num0 &&
                   event.key.code <= sf::Keyboard::Num9) { // choose file

          std::size_t n = event.key.code - sf::Keyboard::Num0;
          if (n >= input_files.size())
            break;

          std::string path = input_files[n];
          std::string state = "optimized";

          if (show_solution)
            biclustering_solver.read_solution(path);
          else {
            state = "random";
            biclustering_solver.parse(path);
            biclustering_solver.initial_random();
          }

          std::string name = utility::filename_from_path(path) + ": " +
                             std::to_string(biclustering_solver.loss());

          window.setTitle(name);
          std::cout << "#" << n << " " << state << " " << name << std::endl;
        }

        break;

      case sf::Event::Closed:
        window.close();
        break;

      case sf::Event::Resized:
        sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
        window.setView(sf::View(visibleArea));
        break;
      }
    }
  }
}*/

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

#include <SFML/Graphics.hpp>

constexpr std::uint64_t NOT_IN_CLUSTER = 0;
constexpr double eps = 1e-12;

namespace utility {
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
} // namespace utility

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

; // namespace utility

class biclustering_solver_t {

  using entry_t = std::vector<std::string>;
  using indices_t = std::vector<std::size_t>;
  template <typename T> using matrix_t = std::vector<std::vector<T>>;

  std::vector<std::int64_t> split(std::string const &str) {
    std::istringstream iss(str);
    return {std::istream_iterator<std::int64_t>{iss},
            std::istream_iterator<std::int64_t>{}};
  }

  template <typename T>
  void construct_matrix(matrix_t<T> &mat, std::size_t n, std::size_t m) {
    mat.resize(n);
    std::fill(std::begin(mat), std::end(mat), std::vector<T>(m));
  }

public:
  biclustering_solver_t(sf::RenderWindow &window) : window(window){};
  biclustering_solver_t(biclustering_solver_t const &) = delete;
  biclustering_solver_t(biclustering_solver_t &&) = delete;

  void parse(std::string const &filename) {
    full_clear();

    std::fstream filestream{filename};
    std::string line;

    /** M, P (number of machines and parts) */
    std::getline(filestream, line);
    auto dimensions = split(line);
    n = dimensions[0];
    m = dimensions[1];

    construct_matrix(matrix, n, m);
    construct_matrix(cluster_matrix, n, m);

    while (std::getline(filestream, line)) {
      auto vector = split(line);
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
    cluster_matrix.clear();
  }

  void random_clear() {
    for (auto &v : machines_clusters)
      v.clear();

    for (auto &v : parts_clusters)
      v.clear();
  }

  void initial_random() {
    static constexpr auto ITERATIONS = 10'000;

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
        std::cout << index << " " << max << std::endl;
      }
    }

    clusters = new_clusters;

    machines_clusters = new_machines_clusters;
    parts_clusters = new_parts_clusters;

    construct_cluster_matrix();

    std::cout << "clusters: " << clusters << std::endl;
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

  void construct_cluster_matrix() {
    construct_matrix(cluster_matrix, n, m); // clear
    for (std::size_t cluster = 0; cluster < clusters; ++cluster)
      for (std::size_t x : machines_clusters[cluster])
        for (std::size_t y : parts_clusters[cluster])
          cluster_matrix[x][y] = cluster + 1;
  }

  double loss() { return loss(clusters); }

  double loss(std::size_t clusters_) {
    std::size_t zeros_in_solution = 0;
    std::size_t ones_in_solution = 0;

    for (std::size_t cluster = 0; cluster < clusters_; ++cluster)
      for (std::size_t x : machines_clusters[cluster])
        for (std::size_t y : parts_clusters[cluster]) {
          ones_in_solution += matrix[x][y];
          zeros_in_solution += !matrix[x][y];
        }

    return ones_in_solution / double(ones_overall + zeros_in_solution);
  }

  void optimize() {
    constexpr std::size_t neighbors = 50;

    double max_loss = loss(), last_loss;

    matrix_t<std::int64_t> solutions;

    std::vector<decltype(machines_clusters)> new_machines_clusters;
    std::vector<decltype(machines_clusters)> new_parts_clusters;

    auto save_solution = [this, &new_machines_clusters, &new_parts_clusters]() {
      new_machines_clusters.push_back(machines_clusters);
      new_parts_clusters.push_back(parts_clusters);
    };

    auto set_current = [this, &new_machines_clusters,
                        &new_parts_clusters](std::size_t i) {
      machines_clusters = new_machines_clusters[i];
      parts_clusters = new_parts_clusters[i];
      clusters = new_machines_clusters[i].size();
    };

    auto restore_original = std::bind(set_current, 0);

    auto clear = [&new_machines_clusters, &new_parts_clusters]() {
      new_machines_clusters.clear();
      new_parts_clusters.clear();
    };

    std::vector<std::function<void()>> shakers;
    shakers.push_back(std::bind(&biclustering_solver_t::split_pass, this));
    shakers.push_back(std::bind(&biclustering_solver_t::merge_pass, this));

    std::size_t index, iterations = 50;

    do {
      last_loss = max_loss;
      index = 0;

      save_solution();

      for (std::size_t i = 0; i < neighbors; ++i) {
        for (auto &shaker : shakers) {
          shaker();
          local_search();
          save_solution();
          restore_original();
        }
      }

      for (std::size_t i = 1; i < new_machines_clusters.size(); ++i) {
        machines_clusters = new_machines_clusters[i];
        parts_clusters = new_parts_clusters[i];
        clusters = new_parts_clusters[i].size();

        double current_loss = loss();

        if (current_loss > max_loss) {
          max_loss = current_loss;
          index = i;
        }

        restore_original();
      }

      set_current(index);

      clear();

    } while (1.0 - last_loss / max_loss > eps || iterations--);

    construct_cluster_matrix();
    std::cout << "clusters: " << clusters << ", loss: " << max_loss
              << ", test: " << loss() << std::endl;
  }

  void local_search() {
    double max_loss = std::numeric_limits<double>::lowest(), last_loss;

    do {
      last_loss = max_loss;

      swap<false>();
      swap<true>();

      max_loss = std::max(max_loss, loss());
    } while (1.0 - last_loss / max_loss > eps);
  }

  template <bool Machines> void swap() {
    double max_loss = loss(), last_loss, current_loss;

    std::int64_t new_cluster1, new_cluster2, i, j;

    std::reference_wrapper<matrix_t<std::size_t>> clusters_vector =
        machines_clusters;

    if constexpr (!Machines)
      clusters_vector = parts_clusters;

    do {
      last_loss = max_loss;
      i = j = new_cluster1 = new_cluster2 = -1;

      for (std::size_t cluster1 = 0; cluster1 < clusters; ++cluster1) {

        if (clusters_vector.get()[cluster1].size() == 1)
          continue;

        for (std::size_t cluster2 = cluster1 + 1; cluster2 < clusters;
             ++cluster2) {

          if (clusters_vector.get()[cluster2].size() == 1)
            continue;

          for (std::size_t x = 0; x < clusters_vector.get()[cluster1].size();
               ++x)
            for (std::size_t y = 0; y < clusters_vector.get()[cluster2].size();
                 ++y) {

              std::swap(clusters_vector.get()[cluster1][x],
                        clusters_vector.get()[cluster2][y]);

              current_loss = loss();
              if (current_loss > max_loss) {
                i = x;
                j = y;
                new_cluster1 = cluster1;
                new_cluster2 = cluster2;

                max_loss = current_loss;
              }

              std::swap(clusters_vector.get()[cluster1][x],
                        clusters_vector.get()[cluster2][y]);
            }
        }
      }

      if (i != -1)
        std::swap(clusters_vector.get()[new_cluster1][i],
                  clusters_vector.get()[new_cluster2][j]);

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

      // split(cluster, x, y);

      // merge(cluster, clusters);

      //  machines_clusters.pop_back();
      // parts_clusters.pop_back();

      // return cluster, x, y
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

    // std::size_t size_machines = machines_clusters[cluster1].size();
    // std::size_t size_parts = parts_clusters[cluster1].size();

    // merge(cluster1, cluster2);

    //  utility::split(machines_clusters[cluster1], machines_clusters[cluster2],
    //                 size_machines);
    //  utility::split(parts_clusters[cluster1], parts_clusters[cluster2],
    //               size_parts);

    // return cluster1, cluster2
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

  void read_solution(std::string &file) {}

  void check_solutions(std::vector<std::string> &files) {}

  void save_data() {}

  void append_loss() {}

  void draw(bool new_colors = false) {
    static std::random_device rd;
    static unsigned int seed = rd();

    seed = (new_colors) ? rd() : seed;

    std::mt19937 prng(seed);

    float size_i = window.getSize().x / float(m);
    float size_k = window.getSize().y / float(n);

    window.clear(sf::Color::Black);

    float radius = 2.5;
    std::vector<sf::Color> colors(clusters + 1);
    colors[0] = sf::Color::White;

    for (std::size_t i = 1; i <= clusters; ++i) {
      int r = prng() % 256, g = prng() % 256, b = prng() % 256;
      colors[i] = sf::Color(r, g, b);
    }

    for (int i = 0; i < n; ++i)
      for (int k = 0; k < m; ++k) {
        std::size_t cluster_id = cluster_matrix[i][k];

        sf::RectangleShape rect(sf::Vector2f(size_i, size_k));

        rect.setPosition(size_i * k, size_k * i);

        rect.setOutlineColor(sf::Color::Black);
        rect.setOutlineThickness(1);

        rect.setFillColor(colors[cluster_id]);
        window.draw(rect);

        if (matrix[i][k]) {
          sf::CircleShape circle;

          circle.setFillColor(sf::Color::Black);
          circle.setRadius(radius);
          circle.setPosition(size_i * k - radius + size_i / 2,
                             size_k * i - radius + size_k / 2);
          window.draw(circle);
        }
      }

    window.display();
  }

private:
  std::int64_t ones_overall = 0;

  matrix_t<bool> matrix;
  matrix_t<std::size_t> cluster_matrix;

  std::size_t n;
  std::size_t m;

  std::size_t clusters; //ору
  matrix_t<std::size_t> machines_clusters;
  matrix_t<std::size_t> parts_clusters;

  sf::RenderWindow &window;

  std::string filename;
};

auto get_files(std::string path) {
  std::vector<std::string> files;
  for (auto entry : std::filesystem::directory_iterator(path))
    files.push_back(entry.path().string());
  return files;
}

void print_choice(std::vector<std::string> &map) {
  std::size_t i = -1;
  std::cout << std::endl;
  while (++i != map.size())
    std::cout << "#" << i << ": " << map[i] << std::endl;
  std::cout << std::endl;
}

sf::RenderWindow &init_window(double size) {
  sf::VideoMode vm = sf::VideoMode::getDesktopMode();
  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;

  static sf::RenderWindow window(
      sf::VideoMode(int(vm.width / size), int(vm.height / size)),
      "Biclustering",
      sf::Style::Titlebar | sf::Style::Close | sf::Style::Resize, settings);

  window.setPosition(sf::Vector2i(vm.width / 2 - vm.width / (size * 2),
                                  vm.height / 2 - vm.height / (size * 2)));

  window.setKeyRepeatEnabled(true);
  window.setVerticalSyncEnabled(true);

  return window;
}

int main(int argc, char *argv[]) {
  auto &window = init_window(1.5);

  biclustering_solver_t biclustering_solver(window);
  auto input_files = get_files("../input");

  biclustering_solver.parse(input_files[0]);
  biclustering_solver.initial_random();

  bool show_solution = false;

  sf::Event event;
  while (window.isOpen()) {
    biclustering_solver.draw();

    while (window.pollEvent(event)) {
      switch (event.type) {

      case sf::Event::KeyPressed:
        if (event.key.code == sf::Keyboard::Escape) { //  close
          window.close();
        }
        if (event.key.code == sf::Keyboard::Space) { // change colors
          biclustering_solver.draw(true);
        } else if (event.key.code == sf::Keyboard::O) { // optimize
          biclustering_solver.optimize();
          std::cout << "done" << std::endl;
        } else if (event.key.code == sf::Keyboard::P) { // print
          print_choice(input_files);
        } else if (event.key.code == sf::Keyboard::H) { // filp show solution
          show_solution ^= true;
        } else if (event.key.code == sf::Keyboard::C) { // check sols
          biclustering_solver.check_solutions(input_files);
        } else if (event.key.code == sf::Keyboard::S) { // save
          biclustering_solver.save_data();
          biclustering_solver.append_loss();
          std::cout << "saved" << std::endl;
        } else if (event.key.code >= sf::Keyboard::Num0 &&
                   event.key.code <= sf::Keyboard::Num9) { // choose file

          std::size_t n = event.key.code - sf::Keyboard::Num0;
          if (n >= input_files.size())
            break;

          std::string path = input_files[n];
          std::string state = "optimized";

          if (show_solution)
            biclustering_solver.read_solution(path);
          else {
            state = "random";
            biclustering_solver.parse(path);
            biclustering_solver.initial_random();
          }

          std::string name = utility::filename_from_path(path) + ": " +
                             std::to_string(biclustering_solver.loss());

          window.setTitle(name);
          std::cout << "#" << n << " " << state << " " << name << std::endl;
        }

        break;

      case sf::Event::Closed:
        window.close();
        break;

      case sf::Event::Resized:
        sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
        window.setView(sf::View(visibleArea));
        break;
      }
    }
  }
}