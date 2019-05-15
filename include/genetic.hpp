#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <valarray>
#include <type_traits>
#include <utility>

/*
Params to tune:
- number of shakings
- total iter
- clustering principle
- shaking cut distance
- number of shakings sequentially
*/

template <typename chromosome_t> class genetic_t {
public:
  using gene_t = typename chromosome_t::value_type;
  // using gene_t = std::uint64_t;
  // using chromosome_t = std::valarray<gene_t>;
  using population_t = std::vector<chromosome_t>;

  struct settings_t {
    std::uint64_t n;  // amount of genes
    double mutatable; // mutable genes probability
    std::uint64_t population_size;

    chromosome_t restrictions;

    double new_population_percentage;
    double mutation_probability;
  };

private:
  std::function<double(chromosome_t &)> fitness;
  settings_t settings;

  static inline std::mt19937 generator{std::random_device{}()};

public:
  genetic_t(std::function<double(chromosome_t &)> fitness, settings_t settings)
      : fitness(fitness), settings(settings) {}

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
    std::discrete_distribution<std::size_t> distr(
        {1.0 - settings.mutatable, settings.mutatable});

    for (std::size_t i = 0; i < std::size(chromosome); ++i)
      if (distr(generator))
        chromosome[i] = mutator(i);

    return chromosome;
  }

  population_t initial_population() {
    population_t population(settings.population_size);
    std::generate(std::begin(population), std::end(population),
                  std::bind(&genetic_t::random_chromosome, this));
    return population;
  }

  std::valarray<double> calc_fitness(population_t &population) {
    std::valarray<double> ret(std::size(population));
    for (std::size_t i = 0; i < std::size(ret); ++i)
      ret[i] = fitness(population[i]);
    std::cout << ret.sum() << std::endl;
    return ret;
  }

  void derivative_population(population_t &population) {
    auto fitness = calc_fitness(population);
    fitness /= fitness.sum();

    auto loss = 1.0 - fitness;

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