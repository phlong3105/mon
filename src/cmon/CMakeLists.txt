cmake_minimum_required(VERSION 3.27)
project(mon)

# set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR "C:/Program Files (x86)/opencv/mingw_build/install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FILE(GLOB_RECURSE SOURCES *.cpp *.hpp)

add_library(${PROJECT_NAME} SHARED
        core/builtin.cpp
        core/builtin.hpp
        core/path.cpp
        core/path.hpp
        global.cpp
        global.hpp
        mon.cpp
        mon.hpp
)

install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
