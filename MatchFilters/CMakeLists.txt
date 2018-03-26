cmake_minimum_required(VERSION 2.8)

PROJECT (MatchFilters)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
#set(BOOST_LIB_DIR "G:/Lib/install/boost/lib64-msvc-14.1/")
#set(BOOSTLIBS "${BOOST_LIB_DIR}libboost_date_time-vc141-mt-gd-x64-1_66.lib;${BOOST_LIB_DIR}libboost_filesystem-vc141-mt-gd-x64-1_66.lib;${BOOST_LIB_DIR}libboost_system-vc141-mt-gd-x64-1_66.lib")

find_package(glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(Caffe  REQUIRED)

file(GLOB MatchFilters_SRCS
    "*.h"
    "*.cpp")

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
    
    
ADD_EXECUTABLE (MatchFilters ${MatchFilters_SRCS} )
message("${BOOSTLIBS}")
if (glog_FOUND)
     message ("glog ${GLOG_INCLUDE_DIRS} ${GLOG_LIBRARIES}xxx")
else(glog_FOUND)
     message ("noglog")
endif (glog_FOUND)
     
message ("gflags ${GFLAGS_INCLUDE_DIR}  xxx")
message ("opencv ${OpenCV_INCLUDE_DIRS} xxx")


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS}  ${GFLAGS_INCLUDE_DIR} ${Caffe_INCLUDE_DIRS})
	target_link_libraries( MatchFilters ${BOOSTLIBS} ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${Caffe_LIBRARIES})
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)
#if (glog_FOUND)
#	include_directories( g:/Lib/install/glog/include/  g:/Lib/install/leveldb/include )
#	target_link_libraries( MnistCaffe ${BOOSTLIBS} G:/Lib/install/glog/lib/glogd.lib G:/Lib/install/glog/lib/glogd.lib G:/Lib/build/leveldb/Debug/leveldbd.lib )
#endif (glog_FOUND)
