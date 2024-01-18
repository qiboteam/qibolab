cmake_minimum_required (VERSION 3.0.2...3.28.1)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

project(libqibolab)

set(VERSION "\"0.0.1\"")

find_package(Python3 COMPONENTS Interpreter Development)

# running the cffi builder
if (NOT EXISTS ${PROJECT_SOURCE_DIR/src/cqibolab.cc})
  execute_process(COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/build.py WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
endif()

include_directories(${Python3_INCLUDE_DIRS})
include_directories(src)
add_library(qibolab SHARED ${PROJECT_SOURCE_DIR}/src/cqibolab.c)

# pkg-config
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "${prefix}")
set(includedir "${prefix}/include")
set(extraincludirs "-I${Python3_INCLUDE_DIR}")
set(libdir "${prefix}/lib")
set(pythonlibs "${Python3_LIBRARIES}")

configure_file(
  "${PROJECT_SOURCE_DIR}/src/qibolab.pc.in"
  "${PROJECT_SOURCE_DIR}/src/qibolab.pc"
  )

install(FILES ${PROJECT_SOURCE_DIR}/src/qibolab.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/qibolab DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS qibolab LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
