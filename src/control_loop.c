
#include "control_loop.h"
#include "fractalexplorer.h"
#include "SDL.h"


control_update_t control_update_loop() {
    control_update_t result;

    result.updated = false;
    result.quit = false;
    

    SDL_Event event;
    while(SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            result.quit = true;
        }
    }
    
    return result;
}

