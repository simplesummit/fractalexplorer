###  src/Makefile.am -- Makefile for automake compiling main executables
#
#    This file is part of the small-summit-fractal project
#
#    small-summit-fractal source code, as well as any other resources in this
#  project are free software; you are free to redistribute it and/or modify
#  them under the terms of the GNU General Public License; either version 3 of
#  thelicense, or any later version.
#
#    These programs are hopefully useful and reliable, but it is understood
#  that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
#  <cade@cade.site> for more info on this.
#
#    Here is a copy of the GPL v3, which this software is licensed under. You
#  can also find a copy at http://www.gnu.org/licenses/.
###

# what flags you want to pass to the C compiler & linker
AM_CFLAGS = --pedantic -Wall -std=c99 -O2
AM_LDFLAGS =


bin_PROGRAMS = fractalexplorer

fractalexplorer_SOURCES = fractalexplorer.c fractalexplorer.h fr.h log.c log.h engine_c.c engine_c.h control_loop.c control_loop.h visuals.c visuals.h #color.c color.h calc_c.c calc_c.h  fractalexplorer_calc_cuda.h util.c util.hrender.h render.c

fractalexplorer_SOURCES += FontCache/SDL_FontCache.c FontCache/SDL_FontCache.h

fractalexplorer_LDADD = @SDL_LDFLAGS@ -lSDL2_ttf
fractalexplorer_CFLAGS = -fopenmp @SDL_CFLAGS@
fractalexplorer_LDFLAGS =

if HAVE_CUDA
fractalexplorer_LDADD += engine_cuda.o
fractalexplorer_LDFLAGS += -lcuda -lcudart -lcudadevrt
fractalexplorer_LDADD += -lcuda -lcudart -lcudadevrt
endif


# TODO: test if nvcc itself supports -Ofast
# for debugging, add '-G -g' and run binary with cuda-memcheck
NVCC_CFLAGS = -Xcompiler -Ofast -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_62,code=compute_62 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_62,code=compute_62
CUDA_CFLAGS = @CUDA_CFLAGS@ $(foreach arg,$(shell mpicc --showme:compile), -Xcompiler $(arg))
CUDA_LDFLAGS = @CUDA_LDFLAGS@  $(foreach arg,$(shell mpicc --showme:link), -Xlinker $(arg))

engine_cuda.o: engine_cuda.cu engine_cuda.h fr.h
	$(NVCC) $(CUDA_CFLAGS) $(NVCC_CFLAGS) -Xptxas -allow-expensive-optimizations=true -use_fast_math -Xcompiler -Ofast -Xcompiler -ffast-math -Xcompiler -fPIC -I. -I.. -DHAVE_CONFIG_H -m64 $(CUDA_LDFLAGS) -lib -o $@ -c $< 

# the -ccbin is special for macos and is just a thing sometimes you have to do
