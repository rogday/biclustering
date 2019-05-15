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

#include <utility.hpp>
#include <biclustering.hpp>

#include <SFML/Graphics.hpp>

namespace drawer {
void draw(sf::RenderWindow &window, biclustering_solver_t &solver,
          bool new_colors = false) {
  static std::random_device rd;
  static unsigned int seed = rd();

  seed = (new_colors) ? rd() : seed;

  std::mt19937 prng(seed);

  std::size_t n = solver.get_n(), m = solver.get_m(),
              clusters = solver.get_clusters();

  auto &cluster_matrix = solver.get_cluster_matrix();
  auto &matrix = solver.get_matrix();

  auto &xs = solver.get_xs();
  auto &ys = solver.get_ys();

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

      std::size_t cluster_id = cluster_matrix[xs[i]][ys[k]];

      sf::RectangleShape rect(sf::Vector2f(size_i, size_k));

      rect.setPosition(size_i * k, size_k * i);

      rect.setOutlineColor(sf::Color::Black);
      rect.setOutlineThickness(1);

      rect.setFillColor(colors[cluster_id]);
      window.draw(rect);

      if (matrix[xs[i]][ys[k]]) {
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
} // namespace drawer

int main(int argc, char *argv[]) {
  using namespace utility;

  auto &window = drawer::init_window(1.5);

  biclustering_solver_t::settings_t settings(SIZE);
  settings[SHAKINGS] = 20;
  settings[ITERATIONS] = 50'000;
  settings[CLUSTERING_PRINCIPLE] = 1;
  settings[SHAKINGS_IN_A_ROW] = 10;

  biclustering_solver_t biclustering_solver(settings);
  auto input_files = utility::get_files("../input");

  biclustering_solver.parse(input_files[0]);
  biclustering_solver.initial_random();

  bool show_solution = false;

  sf::Event event;
  while (window.isOpen()) {
    drawer::draw(window, biclustering_solver);

    while (window.pollEvent(event)) {
      switch (event.type) {

      case sf::Event::KeyPressed:
        if (event.key.code == sf::Keyboard::Escape) { //  close
          window.close();
        }
        if (event.key.code == sf::Keyboard::Space) { // change colors
          drawer::draw(window, biclustering_solver, true);
        } else if (event.key.code == sf::Keyboard::O) { // optimize
          biclustering_solver.optimize();
          std::cout << "done" << std::endl;
        } else if (event.key.code == sf::Keyboard::P) { // print
          drawer::print_choice(input_files);
        } else if (event.key.code == sf::Keyboard::H) { // filp show solution
          show_solution ^= true;
        } else if (event.key.code == sf::Keyboard::S) { // save
          biclustering_solver.save_data();
          // biclustering_solver.append_loss();
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