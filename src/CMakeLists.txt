

set(extra_files "")
set(extra_libs "")

add_executable(fractalexplorer fractalexplorer.h log.h        fractalexplorer.c log.c)

if (OpenCL_FOUND)
    set(extra_libs ${extra_libs} ${OpenCL_LIBRARIES})
endif()

if (CUDA_FOUND)
    set(extra_libs ${extra_libs} ${CUDA_LIBRARIES} stdc++) 
endif()

target_link_libraries(fractalexplorer m pthread ${extra_libs} ${CMAKE_DL_LIBS})

install (FILES fractalexplorer.h DESTINATION include)
