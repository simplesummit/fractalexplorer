
#include "fractalexplorer.h"

// most stuff here
SDL_Window * window;
SDL_Renderer * renderer;

int win_width, win_height;


SDL_Texture * texture_fractal;


void visuals_init(int _win_width, int _win_height) {
    win_width = _win_width;
    win_height = _win_height;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(win_width, win_height, 0, &window, &renderer);

    texture_fractal = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, win_height, win_height);
}

bool visuals_update(hq_color_t * fractal_pixels) {
    // fractal is always a square
    int fractal_dim = win_height;
    fractal.width = fractal_dim;
    fractal.height = fractal_dim;

    
    SDL_UpdateTexture(texture_fractal, NULL, fractal_pixels, sizeof(hq_color_t) * fractal_dim);

    SDL_Rect fractal_rect;
    fractal_rect.x = 0;
    fractal_rect.y = 0;
    fractal_rect.w = fractal_dim;
    fractal_rect.h = fractal_dim;

    SDL_RenderCopy(renderer, texture_fractal, NULL, &fractal_rect);

    SDL_RenderPresent(renderer);

    SDL_Event event;

    bool keep_going = true;

    while (SDL_PollEvent(&event) && keep_going) {
        switch (event.type) {
            case SDL_QUIT: {
                keep_going = false;
                break;
            }
        }
    }

    return keep_going;
}

void visuals_finish() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    SDL_Quit();
}

