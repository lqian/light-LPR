cmake_minimum_required(VERSION 3.0.0)

project(mnn-lpdr  C CXX)

set(_MAIJOR_VERION 0)
set(_MINOR_VERION 1)
set(_PATCH_VERION 0)

# c++11 required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

## redefine executable and library output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "build examples"  ON)
option(BUILD_SHARED_LIBS "build shared labraries, else build archive"  ON)
option(BUILD_JAVA_API "build API for java language" ON)
option(BUILD_UTILS "build extra utils" OFF)

## required package
find_package(OpenMP REQUIRED)
find_package(OpenCV 3.0.0 REQUIRED COMPONENTS core imgcodecs imgproc)

if (OPENMP_FOUND)
	add_definitions("-D_OPENMP")
endif()

include_directories(
	include 
)

if(DEFINED LIGHT_LPR_ARCH)
if (${LIGHT_LPR_ARCH} STREQUAL "arm") 	
	link_directories(
		lib-arm
	)
else()
	link_directories(
		lib
	)
endif()
else()
	link_directories(
		lib
	)
endif()

add_subdirectory(src/mlpdr)

if(BUILD_EXAMPLES)
	 add_subdirectory(src/examples)
endif()
if (BUILD_UTILS)
	add_subdirectory(src/util)
endif()	
