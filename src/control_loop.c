
#include "control_loop.h"
#include "fractalexplorer.h"
#include "SDL.h"
#include "log.h"

#define SDL_SCANCODE_CACHE_SIZE 284

#define WHENPRESS(scn) (key_state[scn] && !last_key_state[scn])
#define ISPRESS(scn) (key_state[scn])


unsigned char * key_state = NULL;
unsigned char * last_key_state = NULL;

// used to scale pixel values to complex numbers
// take pixel_coef * pixels/(sqrt(width*height)*zoom) to get the value
double pixel_coef = 1.75f;

SDL_Event event;


SDL_Joystick * joystick = NULL;


int last_update_ticks = 0;



// whether or not to pan using the mouse motion
bool pan_with_mouse = false;

void control_update_init() {
    // this only needs to be called once
    key_state = (unsigned char *)SDL_GetKeyboardState(NULL);
    last_key_state = (unsigned char *)malloc(SDL_SCANCODE_CACHE_SIZE);

    memcpy(last_key_state, key_state, SDL_SCANCODE_CACHE_SIZE);

    last_update_ticks = SDL_GetTicks();

}

control_update_t control_update_loop() {
    control_update_t result;

    result.updated = false;
    result.quit = false;


    bool keep_going = true;

    float time_mul = SDL_GetTicks() - last_update_ticks;

    // so it isn't too jittery, cap it at 2 fps movement
    if (time_mul > 500) {
        time_mul = 500;
    }

    last_update_ticks = SDL_GetTicks();

    //SDL_PumpEvents();



    while (SDL_PollEvent(&event) && keep_going) {
        switch (event.type) {
            case SDL_QUIT: {
                result.quit = true;
                keep_going = false;
                break;
            }
            case SDL_MOUSEMOTION: {
                if (pan_with_mouse) {
                    double re_change = -pixel_coef * time_mul * event.motion.xrel / (80 * fractal_params.width * fractal_params.zoom);
                    
                    double im_change = pixel_coef * time_mul * event.motion.yrel / (80 * fractal_params.height * fractal_params.zoom);


                    fractal_params.center_r += re_change;
                    fractal_params.center_i += im_change;

                    result.updated = true;
                }
                break;
            }
            case SDL_MOUSEBUTTONUP:
            case SDL_MOUSEBUTTONDOWN: {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    pan_with_mouse = event.button.state == SDL_PRESSED;
                    result.updated = true;
                }
                break;
            }
        }
    }




    // these two sections make you able to pan using the keyboard
    // adjust center imaginary
    if (ISPRESS(SDL_SCANCODE_UP) || ISPRESS(SDL_SCANCODE_W)) {
        fractal_params.center_i += time_mul / (1000 * fractal_params.zoom);
        result.updated = true;
    }

    if (ISPRESS(SDL_SCANCODE_DOWN) || ISPRESS(SDL_SCANCODE_S)) {
        fractal_params.center_i -= time_mul / (1000 * fractal_params.zoom);
        result.updated = true;        
    }

    // adjust center real
    if (ISPRESS(SDL_SCANCODE_RIGHT) || ISPRESS(SDL_SCANCODE_D)) {
        fractal_params.center_r += time_mul / (1000 * fractal_params.zoom);
        result.updated = true;
    }
    if (ISPRESS(SDL_SCANCODE_LEFT) || ISPRESS(SDL_SCANCODE_A)) {
        fractal_params.center_r -= time_mul / (1000 * fractal_params.zoom);
        result.updated = true;
    }

    // zoom in
    if (ISPRESS(SDL_SCANCODE_SPACE)) {
        fractal_params.zoom *= pow(2.5, time_mul / 1000.0);
        result.updated = true;
    }
    if (ISPRESS(SDL_SCANCODE_LSHIFT) || ISPRESS(SDL_SCANCODE_RSHIFT)) {
        fractal_params.zoom /= pow(2.5, time_mul / 1000.0);
        result.updated = true;
    }


    // change fractal
    if (WHENPRESS(SDL_SCANCODE_M)) {
        // cycles:
        // mandelbrot, multibrot

        fractal_type_idx = (fractal_type_idx + 1) % NUM_FRACTAL_TYPES;
        fractal_params.type = fractal_types[fractal_type_idx].flag;

        result.updated = true;
    }

    // quit
    if (WHENPRESS(SDL_SCANCODE_ESCAPE)) {
        result.quit = true;
    }

    memcpy(last_key_state, key_state, SDL_SCANCODE_CACHE_SIZE);


    return result;
}
