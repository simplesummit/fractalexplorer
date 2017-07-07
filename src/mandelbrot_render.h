//



#ifndef __MANDELBROT_RENDER_H__
#define __MANDELBROT_RENDER_H__

void mandlebrot_render(int * argc, char ** argv);


void draw();

void idle_handler();

void key_handler(unsigned char k, int x, int y);

#endif

