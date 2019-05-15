#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <valarray>
#include <functional>
#include <numeric>

#include <utility.hpp>

class biclustering_solver_t {
  using entry_t = std::vector<std::string>;
  using indices_t = std::vector<std::size_t>;
  template <typename T> using matrix_t = std::vector<std::vector<T>>;

  template <typename T>
  void construct_matrix(matrix_t<T> &, std::size_t, std::size_t);

public:
  using settings_t = std::valarray<size_t>;

  biclustering_solver_t(settings_t &settings) : settings(settings){};

  biclustering_solver_t(biclustering_solver_t const &) = delete;
  biclustering_solver_t(biclustering_solver_t &&) = delete;

  void parse(std::string const &);

  void full_clear();

  void initial_random();

  void optimize();

  void random_clear();

  double loss();

  void read_solution(std::string &);
  void save_data();

  // getters
  std::size_t get_n() { return n; }
  std::size_t get_m() { return m; }
  std::size_t get_clusters() { return clusters; }

  matrix_t<std::size_t> &get_cluster_matrix() { return cluster_matrix; }
  matrix_t<bool> &get_matrix() { return matrix; }

  indices_t &get_xs() { return xs; }
  indices_t &get_ys() { return ys; }

private:
  void random_pass();

  void local_search();

  void swap(matrix_t<std::size_t> &);

  void split_pass();

  void split_apply(std::size_t, std::size_t, std::size_t);

  void merge_pass();

  void merge_apply(std::size_t, std::size_t);

  void merge(std::size_t, std::size_t);

  void split(std::size_t, std::size_t i, std::size_t j);

  void construct_cluster_matrix();

private:
  settings_t settings;

  std::int64_t ones_overall = 0;

  matrix_t<bool> matrix;
  matrix_t<std::size_t> cluster_matrix;

  std::size_t n;
  std::size_t m;

  std::size_t clusters;

  matrix_t<std::size_t> machines_clusters;
  matrix_t<std::size_t> parts_clusters;

  indices_t xs;
  indices_t ys;

  std::string path;
};