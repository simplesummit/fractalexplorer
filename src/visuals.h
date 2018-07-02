
/*

This is the generic header file.

These three functions are implemented in "visuals_sdl.c" and "visuals_glfw.c"

But they are switched by visuals.c

*/

#ifndef __VISUALS_H__
#define __VISUALS_H__


#define VISUALS_USE_SDL 0x0001
#define VISUALS_USE_GLFW 0x0002

#define HAVE_GLFW


int visuals_flag;

void visuals_init();

void visuals_update(unsigned char * fractal_pixels);

void visuals_finish();



#endif
