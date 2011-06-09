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
include_directories(${CMAKE_SOURCE_DIR}/src ${Boost_INCLUDES} ${PYTHON_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}/numpy)


ADD_LIBRARY(rbfnetwork SHARED src/rbfnetwork.cpp)
ADD_LIBRARY(pyrbfnet SHARED src/rbfnetwork.cpp src/rbfn_bindings.cpp)

TARGET_LINK_LIBRARIES(rbfnetwork ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} lapack)
TARGET_LINK_LIBRARIES(pyrbfnet ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} lapack)
