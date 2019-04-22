cmake_minimum_required (VERSION 3.10)

project (biclustering)

set(CMAKE_CXX_STANDARD 17)

include_directories("thirdparty/SFML/include")

set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_DIR "thirdparty/SFML/lib/cmake/SFML")
find_package(SFML COMPONENTS graphics window system REQUIRED)

set(TARGET_SOURCES biclustering.cpp)
add_executable(biclustering ${TARGET_SOURCES})
target_link_libraries(biclustering sfml-graphics sfml-window sfml-system)