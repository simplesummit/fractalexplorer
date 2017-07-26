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

fractalexplorer_SOURCES = fractalexplorer.c fractalexplorer.h fr.h color.c color.h log.c log.h calc_c.c calc_c.h  fractalexplorer_calc_cuda.h util.c util.hrender.h render.c
fractalexplorer_LDADD = @SDL_LDFLAGS@ -lSDL2_ttf
fractalexplorer_CFLAGS = @SDL_CFLAGS@
fractalexplorer_LDFLAGS =

if HAVE_CUDA
fractalexplorer_LDADD += calc_cuda.o
fractalexplorer_LDFLAGS += -lcuda -lcudart -lcudadevrt
fractalexplorer_LDADD += -lcuda -lcudart -lcudadevrt
endif


# TODO: test if nvcc itself supports -Ofast
NVCC_CFLAGS = -Xcompiler -Ofast -gencode arch=compute_62,code=sm_62 #-gencode arch=compute_62,code=compute_62 #-gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_62,code=compute_62
CUDA_CFLAGS = @CUDA_CFLAGS@
CUDA_LDFLAGS = @CUDA_LDFLAGS@

calc_cuda.o: calc_cuda.cu fr.h
	$(NVCC) $(CUDA_CFLAGS) $(NVCC_CFLAGS) -Xptxas -allow-expensive-optimizations=true -use_fast_math -Xcompiler -Ofast -Xcompiler -ffast-math -Xcompiler -fPIC -I. -I.. -DHAVE_CONFIG_H -m64 $(CUDA_LDFLAGS) -lib -o $@ -c $<
