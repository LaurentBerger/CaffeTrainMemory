cmake_minimum_required(VERSION 2.8)

PROJECT (BasicCaffeExample)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(Caffe  REQUIRED)

FOREACH( mesdossiers 
       LeNet
       LeNetAndShape
	 LeNetDisplayStruct
    )

    ADD_SUBDIRECTORY(${mesdossiers})

ENDFOREACH()
