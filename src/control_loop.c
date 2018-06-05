
#include "control_loop.h"
#include "fractalexplorer.h"
#include "SDL.h"


unsigned char * key_state = NULL;

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

    last_update_ticks = SDL_GetTicks();

}

control_update_t control_update_loop() {
    control_update_t result;

    result.updated = false;
    result.quit = false;


    bool keep_going = true;

    float time_mul = SDL_GetTicks() - last_update_ticks;
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
                }
                break;
            }
        }
    }

    // these two sections make you able to pan using the keyboard
    // adjust center imaginary
    if (key_state[SDL_SCANCODE_UP]) {
        fractal_params.center_i += time_mul / (1000 * fractal_params.zoom);
    }

    if (key_state[SDL_SCANCODE_DOWN]) {
        fractal_params.center_i -= time_mul / (1000 * fractal_params.zoom);
    }

    // adjust center real
    if (key_state[SDL_SCANCODE_RIGHT]) {
        fractal_params.center_r += time_mul / (1000 * fractal_params.zoom);
    }
    if (key_state[SDL_SCANCODE_LEFT]) {
        fractal_params.center_r -= time_mul / (1000 * fractal_params.zoom);
    }

    // zoom in
    if (key_state[SDL_SCANCODE_SPACE]) {
        fractal_params.zoom *= pow(2.5, time_mul / 1000.0);
    }
    if (key_state[SDL_SCANCODE_LSHIFT] || key_state[SDL_SCANCODE_RSHIFT]) {
        fractal_params.zoom /= pow(2.5, time_mul / 1000.0);
    }

    return result;
}

