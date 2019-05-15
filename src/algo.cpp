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
  restrictions[SHAKINGS] = 20;
  restrictions[ITERATIONS] = 50'000;
  restrictions[CLUSTERING_PRINCIPLE] = 1;
  restrictions[SHAKINGS_IN_A_ROW] = 10;

  settings.restrictions = restrictions;

  auto input_files = get_files("../input");

  auto fitness = [&input_files](chromosome_t &settings) {
    biclustering_solver_t biclustering_solver(settings);
    biclustering_solver.parse(input_files[0]);
    biclustering_solver.initial_random();
    biclustering_solver.optimize();

    for (auto &val : settings)
      std::cout << val << " ";
    std::cout << std::endl;

    return biclustering_solver.loss();
  };

  genetic_t<chromosome_t> genetic_algo(fitness, settings);
  genetic_algo.run(20);

  return 0;
}