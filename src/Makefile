## Makefile

CC=mpicc
CFLAGS=-std=c99 -Werror -Wall -ffast-math -O3 -I.
LDFLAGS=-lmpi -lglut -lGL -lGLU -lm


mandelbrot_src = mandelbrot.o mandelbrot_render.o mandelbrot_util.o mandelbrot_calc_c.o log.o


default: mandelbrot
	@

%.o: %.c %.h
	$(CC) -c -o $@ $<

mandelbrot: $(mandelbrot_src)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


clean:
	rm -f *.o


