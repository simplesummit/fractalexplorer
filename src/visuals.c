#include "visuals.h"
#include "fractalexplorer.h"
#include "log.h"
#include "SDL.h"

SDL_Window * window;
SDL_Renderer * renderer;
SDL_Texture * texture;
unsigned char * texture_raw;

void visuals_init() {
    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO) != 0) {
        log_fatal("Fail on SDL_Init(): %s", SDL_GetError());
        M_EXIT(1);
    }

    atexit(SDL_Quit);

    int window_flags = 0;

    window = SDL_CreateWindow("fractalexplorer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, fractal_params.width, fractal_params.height, window_flags);

    if (window == NULL) {
        log_fatal("Fail on SDL_CreateWindow(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_GetWindowSize(window, &fractal_params.width, &fractal_params.height);
    SDL_ShowCursor(SDL_DISABLE);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (renderer == NULL) {
        log_fatal("Fail on SDL_CreateRenderer(): %s", SDL_GetError());
        M_EXIT(1);
    }


    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, fractal_params.width, fractal_params.height);
    if (texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }


    texture_raw = malloc(3 * fractal_params.width * fractal_params.height);
}

typedef struct RGB_t {
    unsigned char rgb[3];
} RGB_t;

void visuals_update(unsigned char * fractal_pixels) {
    // fractal_pixels contains column major order in RGB
    // needs to be converted into "texture" into row major RGB

    int i, j;
    int from_xy, to_xy;
    for (i = 0; i < fractal_params.width; ++i) {
        for (j = 0; j < fractal_params.height; ++j) {
            // column major
            //from_xy = 3 * (fractal_params.height * i + j);
            // row major
            //to_xy = 3 * (fractal_params.width * j + i);
            
            /*
            texture_raw[to_xy + 0] = 255 * i / fractal_params.width;
            texture_raw[to_xy + 1] = 255 * j / fractal_params.height;
            texture_raw[to_xy + 2] = 255;
            */

           // printf("%d,%d,%d\n", fractal_pixels[from_xy + 0], fractal_pixels[from_xy + 1], fractal_pixels[from_xy + 2]);
            
            //texture_raw[to_xy + 0] = fractal_pixels[from_xy + 0];
            //texture_raw[to_xy + 1] = fractal_pixels[from_xy + 1];
            //texture_raw[to_xy + 2] = fractal_pixels[from_xy + 2];
            ((RGB_t*)texture_raw)[j * fractal_params.width + i] = ((RGB_t*)fractal_pixels)[i * fractal_params.height + j];
        }
    }

    SDL_RenderClear(renderer);

    SDL_UpdateTexture(texture, NULL, texture_raw, 3 * fractal_params.width);
    SDL_RenderCopy(renderer, texture, NULL, NULL);


    // do graphs here

    SDL_RenderPresent(renderer);

}

void visuals_finish() {


    SDL_DestroyWindow(window);

    SDL_Quit();

}
