file(GLOB_RECURSE mlpdr_srcs *.cpp)
file(GLOB mlpdr_api_srcs *_api.cpp)

if(NOT BUILD_JAVA_API)
	list(REMOVE_ITEM mlpdr_srcs ${mlpdr_api_srcs} )
endif()

add_library(mlpdr ${mlpdr_srcs})
target_link_libraries(mlpdr MNN ${OpenMP_CXX} ${OpenCV_LIBRARIES})