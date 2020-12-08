cmake_minimum_required(VERSION 3.10)
project(Pycgal)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


find_package(pybind11 REQUIRED)

find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

find_package(Eigen3 3.1.0 REQUIRED)
#find_package(Boost REQUIRED system filesystem date_time thread)

include(CGAL_Eigen_support)

if(NOT TARGET CGAL::Eigen_support)
  message(STATUS "This project requires the Eigen library, and will not be compiled.")
  return()
endif()



pybind11_add_module(pycgal src/python_binding.cpp)
set_target_properties(pycgal PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

foreach(target pycgal)
  target_link_libraries(${target} PUBLIC CGAL::Eigen_support)# PRIVATE ${Boost_LIBRARIES})
endforeach()