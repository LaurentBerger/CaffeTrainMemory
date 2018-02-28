cmake_minimum_required(VERSION 2.8)

PROJECT (LeNetAndShape)


file(GLOB LeNetAndShape_SRCS
    "*.h"
    "*.cpp")

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
    
    
ADD_EXECUTABLE (LeNetAndShape ${LeNetAndShape_SRCS} )
if (glog_FOUND)
     message ("glog ${GLOG_INCLUDE_DIRS} ${GLOG_LIBRARIES}xxx")
else(glog_FOUND)
     message ("noglog")
endif (glog_FOUND)
     
message ("gflags ${GFLAGS_INCLUDE_DIR}  xxx")
message ("opencv ${OpenCV_INCLUDE_DIRS} xxx")


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS}  ${GFLAGS_INCLUDE_DIR} ${Caffe_INCLUDE_DIRS})
	target_link_libraries( LeNetAndShape ${BOOSTLIBS} ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${Caffe_LIBRARIES})
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)
