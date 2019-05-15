#include <utility.hpp>

std::string utility::filename_from_path(std::string &path) {
  return std::filesystem::path(path).stem().string();
}

std::size_t utility::generate_random_value(std::size_t lower_bound,
                                           std::size_t upper_bound) {

  std::uniform_int_distribution<std::size_t> distribution(lower_bound,
                                                          upper_bound);
  return distribution(generator);
}

std::vector<std::int64_t> utility::split(std::string const &str) {
  std::istringstream iss(str);
  return {std::istream_iterator<std::int64_t>{iss},
          std::istream_iterator<std::int64_t>{}};
}

std::vector<std::string> utility::get_files(std::string path) {
  std::vector<std::string> files;
  for (auto entry : std::filesystem::directory_iterator(path))
    files.push_back(entry.path().string());
  return files;
}
