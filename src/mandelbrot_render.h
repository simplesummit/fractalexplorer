//



#ifndef __MANDELBROT_RENDER_H__
#define __MANDELBROT_RENDER_H__

void mandelbrot_render(int * argc, char ** argv);

void mouse_handler(int button, int state, int x, int y);

void motion_handler(int x, int y);

void draw();

void idle_handler();

void key_handler(unsigned char k, int x, int y);

#endif

