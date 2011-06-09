cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
PROJECT(rbfnetwork2)

SET(CMAKE_VERBOSE_MAKEFILE OFF)

SET(CMAKE_BUILD_TYPE distribution)
# SET(CMAKE_BUILD_TYPE debug)


SET(CMAKE_CXX_FLAGS_DISTRIBUTION " -Wall -O3 -D_POSIX_PTHREAD_SEMANTICS -D_REENTRANT -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG " -Wall -g3  -D_POSIX_PTHREAD_SEMANTICS -D_REENTRANT -DBZ_DEBUG")

#boost bits
find_package( Boost COMPONENTS python serialization REQUIRED)
find_package( PythonLibs REQUIRED)

# find_package(clapack REQUIRED)

# message(STATUS "Boost_LIBRARIES=${Boost_LIBRARIES}")
# message(STATUS "Python_Includes=${PYTHON_INCLUDE_DIRS}")

ADD_DEFINITIONS(-Wall -Wno-deprecated)
include_directories(${CMAKE_SOURCE_DIR}/. ${Boost_INCLUDES} ${PYTHON_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}/numpy)

# ADD_EXECUTABLE(rbfnetwork2 main.cpp rbfnetwork.cpp logreader.cpp readlog_train.cpp test_rbfn.cpp rbfn_analysis.cpp create_training_set.cpp rbfset.cpp)

# ADD_LIBRARY(pyrbfnet SHARED rbfnetwork.cpp bindings.cpp rbfset.cpp)
ADD_LIBRARY(rbfnetwork SHARED rbfnetwork.cpp)
ADD_LIBRARY(pyrbfnet SHARED rbfnetwork.cpp rbfn_bindings.cpp)

# TARGET_LINK_LIBRARIES(rbfnetwork2 lapack-3 boost_serialization)
TARGET_LINK_LIBRARIES(pyrbfnet ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} lapack)
