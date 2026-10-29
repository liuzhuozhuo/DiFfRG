cmake_minimum_required(VERSION 3.27.0)

project(ONxOMfiniteT)

# ##############################################################################
# Setup Application
# ##############################################################################

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

add_subdirectory(flows)

# ##############################################################################
# Kurganov-Tadmor finite volume on the polar or Cartesian grid of the two invariants, and the 1D reference
# ##############################################################################

add_executable(ONxOM ONxOM.cc)
target_link_libraries(ONxOM PRIVATE DiFfRG::DiFfRG ONxOMFlows)

# ##############################################################################
# Symlink the parameter file into the build directory
# ##############################################################################

execute_process(
  COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
          "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
