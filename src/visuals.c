
/*

This is the generic header file.

These three functions are implemented in "visuals_sdl.c" and "visuals_glfw.c"

But they are switched by visuals.c

*/


#include "visuals.h"

#include "visuals_sdl.h"
#include "visuals_glfw.h"

int visuals_flag = VISUALS_USE_GLFW;

void visuals_init() {
    if (visuals_flag == VISUALS_USE_SDL) {
        visuals_sdl_init();
    } else if (visuals_flag == VISUALS_USE_GLFW) {
#ifdef HAVE_GLFW
        visuals_glfw_init();
#endif
    }
}

void visuals_update(unsigned char * fractal_pixels) {
    if (visuals_flag == VISUALS_USE_SDL) {
        visuals_sdl_update(fractal_pixels);
    } else if (visuals_flag == VISUALS_USE_GLFW) {
#ifdef HAVE_GLFW
        visuals_glfw_update(fractal_pixels);
#endif
    }
}

void visuals_finish() {
    if (visuals_flag == VISUALS_USE_SDL) {
        visuals_sdl_finish();
    } else if (visuals_flag == VISUALS_USE_GLFW) {
#ifdef HAVE_GLFW
        visuals_glfw_finish();
#endif
    }
}

