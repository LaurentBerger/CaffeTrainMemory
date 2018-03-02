cmake_minimum_required(VERSION 2.8)

PROJECT (LeNetDisplayStruct)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

find_package(glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(Caffe  REQUIRED)

file(GLOB LeNetDisplayStruct_SRCS
    "*.h"
    "*.cpp")

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
    
    
ADD_EXECUTABLE (LeNetDisplayStruct ${LeNetDisplayStruct_SRCS} )


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS}  ${GFLAGS_INCLUDE_DIR} ${Caffe_INCLUDE_DIRS})
	target_link_libraries( LeNetDisplayStruct ${BOOSTLIBS} ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${Caffe_LIBRARIES})
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)
