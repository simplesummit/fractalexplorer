

set(extra_files "")
set(extra_libs "")

add_executable(fractalexplorer fractalexplorer.h log.h        fractalexplorer.c log.c visuals.c util.c color.c compute_C.c)

if (OpenCL_FOUND)
    set(extra_libs ${extra_libs} ${OpenCL_LIBRARIES})
endif()

if (CUDA_FOUND)
    set(extra_libs ${extra_libs} ${CUDA_LIBRARIES} stdc++) 
endif()

target_link_libraries(fractalexplorer m pthread ${SDL2_LIBRARIES} ${MPI_LIBRARIES} ${extra_libs} ${CMAKE_DL_LIBS})




if(MPI_COMPILE_FLAGS)
  set_target_properties(fractalexplorer PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(fractalexplorer PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

install (FILES fractalexplorer.h DESTINATION include)
