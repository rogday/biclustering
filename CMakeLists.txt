cmake_minimum_required (VERSION 3.10)

project (algorithms)

set(CMAKE_CXX_STANDARD 17)

include_directories("include")
include_directories("library/include")

#SFML
include_directories("thirdparty/SFML/include")

set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_DIR "thirdparty/SFML/lib/cmake/SFML")
find_package(SFML COMPONENTS graphics window system REQUIRED)

#utility
set(UTILITY src/utility.cpp)

#Biclustering library
set(LIBRARY library/src/biclustering.cpp)

#Genetic algo
set(GENETIC src/algo.cpp)
add_executable(genetic ${GENETIC} ${UTILITY} ${LIBRARY})

#Visualizer
set(DRAWER src/drawer.cpp)
add_executable(drawer ${DRAWER} ${UTILITY} ${LIBRARY})
target_link_libraries(drawer sfml-graphics sfml-window sfml-system)