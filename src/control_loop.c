
#include "control_loop.h"
#include "fractalexplorer.h"
#include "SDL.h"
#include "log.h"

#include "visuals.h"

// 
#include "visuals_glfw.h"

#define SDL_SCANCODE_CACHE_SIZE 284

#define WHENPRESS(scn) (key_state[scn] && !last_key_state[scn])
#define ISPRESS(scn) (key_state[scn])


unsigned char * key_state = NULL;
unsigned char * last_key_state = NULL;

// used to scale pixel values to complex numbers
// take pixel_coef * pixels/(sqrt(width*height)*zoom) to get the value
double pixel_coef = 1.75f;

float last_glfwtime;

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

// which controller scheme to use
#include "controllers/logitech_f310.h"

int _joy_i = -1;

control_update_t control_update_loop_sdl() {
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
    SDL_JoystickUpdate();
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
            case SDL_JOYDEVICEADDED:
                if (event.jdevice.which >= 0) {
                    _joy_i = event.jdevice.which;
                    printf("Joystick added: %d\n", _joy_i);
                    joystick = SDL_JoystickOpen(_joy_i);

                } break;
            case SDL_JOYDEVICEREMOVED:
                if (event.jdevice.which >= 0) {
                    printf("Joystick removed: %d\n", event.jdevice.which);
                    if (event.jdevice.which == _joy_i) {
                        joystick = NULL; printf("removed active controller!\n");
                        _joy_i = -1;
                    }

                } break;
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

#define SMASH(x) (fabs(x) > 0.05f) ? (x) : 0.0f;
#define JOYSCALE(x) SMASH(((float)(x))/(32768.0f))
    if (joystick != NULL) {
        float horiz = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_HORIZONTAL_AXIS));
        fractal_params.center_r += time_mul * horiz / (1500.0 * fractal_params.zoom);
        float vertical = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_VERTICAL_AXIS));
        fractal_params.center_i -= time_mul * vertical / (1500.0 * fractal_params.zoom);
        float qr = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_QR_AXIS));
        float qi = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_QI_AXIS));
        fractal_params.q_r += time_mul * qr / (5000.0 * fractal_params.zoom);
        fractal_params.q_i += time_mul * qi / (5000.0 * fractal_params.zoom);

/*
        float zoom_in = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_ZOOM_POS_AXIS));
        float zoom_out = JOYSCALE(SDL_JoystickGetAxis(joystick, CONTROLLER_ZOOM_NEG_AXIS));

        float total_zoom = zoom_in - zoom_out;
        fractal_params.zoom *= pow(2.5, time_mul * total_zoom / 1000.0);
*/

        result.updated = true;
    }

    // quit
    if (WHENPRESS(SDL_SCANCODE_ESCAPE)) {
        result.quit = true;
    }

    memcpy(last_key_state, key_state, SDL_SCANCODE_CACHE_SIZE);


    return result;
}

#define GLFW_ISPRESS(x) (glfwGetKey(window, x) == GLFW_PRESS)

control_update_t control_update_loop_glfw() {
    control_update_t result;

    glfwPollEvents();

    result.updated = false;
    result.quit = false;


    bool keep_going = true;

    float time_mul = glfwGetTime() - last_glfwtime;

    // so it isn't too jittery, cap it at 2 fps movement
    if (time_mul > 0.5) {
        time_mul = 0.5;
    }

    last_glfwtime = glfwGetTime();

    //SDL_PumpEvents();

    // these two sections make you able to pan using the keyboard
    // adjust center imaginary
    if (GLFW_ISPRESS(GLFW_KEY_W) || GLFW_ISPRESS(GLFW_KEY_UP)) {
        fractal_params.center_i -= time_mul / fractal_params.zoom;
        result.updated = true;
    }

    if (GLFW_ISPRESS(GLFW_KEY_S) || GLFW_ISPRESS(GLFW_KEY_DOWN)) {
        fractal_params.center_i += time_mul / fractal_params.zoom;
        result.updated = true;
    }

    if (GLFW_ISPRESS(GLFW_KEY_A) || GLFW_ISPRESS(GLFW_KEY_LEFT)) {
        fractal_params.center_r -= time_mul / fractal_params.zoom;
        result.updated = true;
    }

    if (GLFW_ISPRESS(GLFW_KEY_D) || GLFW_ISPRESS(GLFW_KEY_RIGHT)) {
        fractal_params.center_r += time_mul / fractal_params.zoom;
        result.updated = true;
    }


    if (GLFW_ISPRESS(GLFW_KEY_SPACE)) {
        fractal_params.zoom *= pow(2.5, time_mul);
        result.updated = true;
    }

    if (GLFW_ISPRESS(GLFW_KEY_LEFT_SHIFT) || GLFW_ISPRESS(GLFW_KEY_RIGHT_SHIFT)) {
        fractal_params.zoom /= pow(2.5, time_mul);
        result.updated = true;
    }

    return result;
}


control_update_t control_update_loop() {
    if (visuals_flag == VISUALS_USE_SDL) {
        return control_update_loop_sdl();
    } else {
        return control_update_loop_glfw();
    }
}


