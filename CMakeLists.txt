cmake_minimum_required (VERSION 2.6)

option(ENABLE_CUDA "enable cuda usage" OFF)
option(CUDA_COMPILER_WRAPPER "CUDA compiler wrapper (ex: 'macos_nvcc')" OFF)

if (CUDA_COMPILER_WRAPPER)
SET(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_WRAPPER})
else()
SET(CMAKE_CUDA_COMPILER nvcc)
endif()

if (ENABLE_CUDA)
    project (fractalexplorer LANGUAGES C)# CUDA)
else()
    project (fractalexplorer LANGUAGES C)
endif()


cmake_policy(SET CMP0017 OLD) 


set (FRACTALEXPLORER_VERSION_MAJOR 0)
set (FRACTALEXPLORER_VERSION_MINOR 1)


set (CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(SDL2 REQUIRED)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})


if (ENABLE_CUDA)
    find_package(CUDA)

    if (CUDA_FOUND)
        message(STATUS "using cuda")
        add_definitions(-DHAVE_CUDA)
    else()
        message(ERROR "Wanted CUDA but couldn't find it")
    endif()
endif()


find_package(OpenCL)


if (ENABLE_OpenCL)
    find_package(CUDA)

    if (OpenCL_FOUND)
        message(STATUS "using cuda")
        add_definitions(-DHAVE_OpenCL)
        include_directories(${OpenCL_INCLUDE_DIRS})
    else()
        message(ERROR "Wanted OpenCL but couldn't find it")
    endif()
endif()



#optimized
#-Ofast -ffast-math -finline-functions
set (CMAKE_C_FLAGS "-Ofast -ffast-math -finline-functions -g -Wall -pedantic -std=c99 ${CMAKE_C_FLAGS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")



set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")


configure_file (
  "${PROJECT_SOURCE_DIR}/fractalexplorerconfig.h.in"
  "${PROJECT_BINARY_DIR}/fractalexplorerconfig.h"
)

if (CMAKE_BUILD_TYPE MATCHES RELEASE)
    message("release mode")
else()
    message("dev mode")
    add_definitions(-DFRACTALEXPLORER_DEV)
endif()

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/src")

add_subdirectory (src) 


#add_subdirectory (examples) 


install (FILES "${PROJECT_BINARY_DIR}/fractalexplorerconfig.h" DESTINATION include)

