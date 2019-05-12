#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <functional>

/*
Params to tune:
- number of shakings
- total iter
- clustering principle
- shaking cut distance
- number of shakings sequentially
*/

namespace drochevo {

using gene_t = std::uint64_t;
using chromosome_t = std::vector<gene_t>;
using population_t = std::vector<chromosome_t>;

std::uint64_t n = 10;            // amount of genes
std::uint64_t mutatable = n / 2; // mutatable genes
std::uint64_t population_size = 100;

double new_population_percentage = 0.7;
double mutation_probability = 0.2;

std::mt19937 generator{std::random_device{}()};

// chromosome_t restrictions(n);
chromosome_t restrictions = {2, 1, 3, 4, 6, 7, 9, 5, 8, 10};

template <typename T> T random(T lower_bound, T upper_bound) {
  std::uniform_int_distribution<T> distribution(lower_bound, upper_bound);

  return distribution(generator);
}

auto mutator = [](std::size_t i) { return random<gene_t>(0, restrictions[i]); };

chromosome_t random_chromosome() {
  chromosome_t chromosome(n);
  for (std::size_t i = 0; i < std::size(chromosome); ++i)
    chromosome[i] = mutator(i);
  return chromosome;
}

chromosome_t crossover(chromosome_t &mammy, chromosome_t &daddy) {
  std::uint64_t index = random(static_cast<std::uint64_t>(1), n - 1);

  chromosome_t child(mammy);
  for (std::size_t i = index; i < n; ++i)
    std::swap(child[i], daddy[i]);

  return child;
}

chromosome_t mutation(chromosome_t &chromosome) {
  std::vector<std::size_t> indices(n);
  std::iota(std::begin(indices), std::end(indices), 0);
  std::shuffle(std::begin(indices), std::end(indices), generator);

  for (std::size_t i = 0; i < mutatable; ++i)
    chromosome[indices[i]] = mutator(indices[i]);
  return chromosome;
}

population_t initial_population() {
  population_t population(population_size);
  for (auto &chromosome : population)
    chromosome = random_chromosome();
  return population;
}

// extern double fitness(chromosome_t &);
double fitness(chromosome_t &) { return 1.0; };

std::vector<double> calc_fitness(population_t &population) {
  std::vector<double> ret(std::size(population));
  for (std::size_t i = 0; i < std::size(ret); ++i)
    ret[i] = fitness(population[i]);
  return ret;
}

void derivative_population(population_t &population) {
  auto fitness = calc_fitness(population);

  double sum = std::accumulate(std::begin(fitness), std::end(fitness), 0.0);

  std::for_each(std::begin(fitness), std::end(fitness),
                [sum](double &value) { return value / sum; });

  std::discrete_distribution<std::size_t> indices(std::begin(fitness),
                                                  std::end(fitness));

  std::vector<chromosome_t> parents(
      2 * static_cast<std::size_t>(n * new_population_percentage));

  for (std::size_t i = 0; i < std::size(parents); ++i)
    parents[i] = population[indices(generator)];

  for (std::size_t i = 0; i < std::size(parents); i += 2) {
    auto child = mutation(crossover(parents[i], parents[i + 1]));
    population[indices(generator)] = child;
  }
}

} // namespace drochevo

int main() { return 0; }