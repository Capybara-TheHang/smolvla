# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-src")
  file(MAKE_DIRECTORY "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-src")
endif()
file(MAKE_DIRECTORY
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-build"
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix"
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/tmp"
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/src/sanitizers-cmake-populate-stamp"
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/src"
  "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/src/sanitizers-cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/src/sanitizers-cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/lightwheel/workspace/smolvla/isaacteleop/build/_deps/sanitizers-cmake-subbuild/sanitizers-cmake-populate-prefix/src/sanitizers-cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
