

#ifndef __VISUALS_H__
#define __VISUALS_H__

#include "SDL.h"

SDL_Window * window;
SDL_Renderer * renderer;
SDL_Surface * screen;
SDL_Texture * texture;


void visuals_init();

void visuals_update(unsigned char * fractal_pixels);

void visuals_finish();



#endif
