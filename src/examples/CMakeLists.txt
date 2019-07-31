find_package(OpenCV 3.4.0 REQUIRED COMPONENTS core imgcodecs imgproc highgui)

file(GLOB_RECURSE test_srcs *.cpp)
 
foreach(source_file ${test_srcs})
 	# get file name
  	get_filename_component(name ${source_file} NAME_WE)
	# get folder name
  	get_filename_component(path ${source_file} PATH)
  	get_filename_component(folder ${path} NAME_WE)
    
  	add_executable(${name} ${source_file})
	target_link_libraries(${name} mlpdr ${OpenCV_LIBRARIES} ${OpenMP_CXX})
  

  	# set back RUNTIME_OUTPUT_DIRECTORY
  	set_target_properties(${name} PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${folder}")

endforeach()
