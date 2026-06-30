cmake_minimum_required(VERSION 4.1.2)

project(
  LightSLAM
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME libonnxruntime)

file(GLOB SOURCES "src/*.cc")

add_library(${PROJECT_NAME}_lib SHARED ${SOURCES})

target_include_directories(${PROJECT_NAME}_lib PUBLIC 
	include 
	${OpenCV_INCLUDE_DIRS} 
	${Eigen3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
	${ONNXRUNTIME_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}_lib PUBLIC 
	${OpenCV_LIBS} 
	Eigen3::Eigen 
	${Boost_libraries}
	${ONNXRUNTIME_LIBRARIES})

# demos
add_executable(xfeat_lighterglue demo/XFeat_LighterGlue.cc)
target_include_directories(xfeat_lighterglue PRIVATE include)
target_link_libraries(xfeat_lighterglue PRIVATE ${PROJECT_NAME}_lib)
