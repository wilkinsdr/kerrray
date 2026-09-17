cmake_minimum_required (VERSION 3.10)
project (raytrace_cpu)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/obj)

if(EXISTS ${CMAKE_SOURCE_DIR}/cmake.local)
include(${CMAKE_SOURCE_DIR}/cmake.local)
endif ()

find_package(GSL)
include_directories(${GSL_INCLUDE_DIRS})

# CFITSIO
find_library(CFITSIO_LIBRARIES NAMES cfitsio HINTS ${CFITSIO_PATH}/lib)
find_path(CFITSIO_INCLUDE_DIR fitsio.h PATH_SUFFIXES libcfitsio3 libcfitsio0 cfitsio HINTS ${CFITSIO_PATH}/include)
set(cfitsio_libs ${CFITSIO_LIBRARIES})
include_directories(${CFITSIO_INCLUDE_DIR})

# HDF5
# HDF5_ROOT and HDF5_INCLUDE_DIRS can be specified in CMakeLocal.txt if cmake cannot find them automatically
#set(HDF5_USE_STATIC_LIBRARIES ON)
#find_package(HDF5 REQUIRED CXX)
find_package(HDF5 REQUIRED CXX)
find_library(hdf5 hdf5)
find_library(hdf5_cpp hdf5_cpp)
# set(hdf5 hdf5)
# set(hdf5_cpp hdf5_cpp)
# set(hdf5_libs hdf5_cpp hdf5)
message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
message(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")

find_package(OpenMP)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()

include_directories("${CMAKE_SOURCE_DIR}/src")
add_subdirectory (src)
