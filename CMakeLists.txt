cmake_minimum_required(VERSION 2.8)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
find_package( OpenCV 3.1 REQUIRED )

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package (Threads REQUIRED)

SET(CMAKE_CXX_FLAGS "-g -std=c++11")
include_directories(  ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} include)

foreach( lp RANGE 2 2)
    add_executable( problem${lp} problem${lp}.cpp  RobustMatcher.cpp)
    set_target_properties( problem${lp} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../Problem${lp}")
     target_link_libraries( problem${lp} ${EIGEN3_LIBS} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} )
endforeach(lp)
