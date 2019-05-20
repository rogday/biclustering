#include <iostream>

#include <genetic.hpp>
#include <utility.hpp>
#include <biclustering.hpp>

int main() {
  using namespace utility;
  using chromosome_t = biclustering_solver_t::settings_t;

  genetic_t<chromosome_t>::settings_t settings;
  settings.n = SIZE;
  settings.mutatable = 1.0 / SIZE;
  settings.population_size = 15;
  settings.new_population_percentage = 0.7;
  settings.mutation_probability = 0.2;

  chromosome_t restrictions(settings.n);
  restrictions[SHAKINGS] = 50;
  restrictions[ITERATIONS] = 100'000;
  restrictions[CLUSTERING_PRINCIPLE] = 1;

  settings.restrictions = restrictions;

  auto input_files = get_files("../input");

  auto fitness = [](std::string &filename, chromosome_t &settings) {
    biclustering_solver_t biclustering_solver(settings);
    biclustering_solver.parse(filename);
    biclustering_solver.initial_random();
    biclustering_solver.optimize();

    std::cout << "parameters: ";
    for (auto &val : settings)
      std::cout << val << " ";
    std::cout << std::endl;

    return biclustering_solver.loss();
  };

  for (auto &filename : input_files) {
    std::cout << filename << std::endl;
    genetic_t<chromosome_t> genetic_algo(
        std::bind(fitness, filename, std::placeholders::_1), settings);
    genetic_algo.run(20);
  }

  return 0;
}