## Makefile

CC=mpicc
NVCC=/usr/local/cuda/bin/nvcc

USE_CUDA = true

NVCC_CFLAGS=-I. -I/usr/local/cuda/include/ -Xcompiler -O3 -Xcompiler -fPIC -m64 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_62,code=compute_62
NVCC_LDFLAGS=

CFLAGS=-std=c99 -Werror -Wall -ffast-math -O3 -I. $(shell sdl2-config --cflags)
LDFLAGS=-L/usr/local/cuda/lib64 -lmpi -lSDL2 -lm $(shell sdl2-config --ldflags)

mandelbrot_src = mandelbrot.o mandelbrot_render.o mandelbrot_calc_c.o color.o log.o 

ifeq ($(USE_CUDA), true)
CFLAGS += -DUSE_CUDA
LDFLAGS += -lcuda -lcudart -lcudadevrt
mandelbrot_src += mandelbrot_calc_cuda.cuo
endif



default: mandelbrot
	@

%.cuo: %.cu %.h
	$(NVCC) $(NVCC_CFLAGS) -lib -c -o $@ $< $(NVCC_LDFLAGS)


%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

mandelbrot: $(mandelbrot_src)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


clean:
	rm -f *.o


