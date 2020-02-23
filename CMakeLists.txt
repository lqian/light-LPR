cmake_minimum_required(VERSION 3.0.0)

project(Light-LPR  C CXX)

set(_MAIJOR_VERION 1)
set(_MINOR_VERION 1)
set(_PATCH_VERION 0)

## redefine executable and library output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "build examples"  ON)
option(BUILD_SHARED_LIBS "build shared labraries, else build archive"  ON)
option(BUILD_JAVA_API "build API for java language" ON)


# need the real opencv install path if build on windows
set(OpenCV_DIR  "C:\\Users\\xinhuan\\Desktop\\workspace\\opencv\\build-win32\\install")

## required package
find_package(OpenMP REQUIRED)
find_package(OpenCV 3.0.0 REQUIRED COMPONENTS core imgcodecs imgproc highgui)

if (OPENMP_FOUND)
	add_definitions("-D_OPENMP")
endif()

include_directories(
	include 
)

if (WIN32 OR MSVC OR MSYS OR MINGW)
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
	link_directories(
		lib-win32
	)
else()
	# c++11 required
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
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
endif()


add_subdirectory(src/mlpdr)

if(BUILD_EXAMPLES)
	 add_subdirectory(src/examples)
endif()
