// for controlling the fractal parameters

#ifndef __CONTROL_LOOP_H__
#define __CONTROL_LOOP_H__

#include <stdbool.h>
#include <SDL.h>

typedef struct control_update_t {
    // tells what updated

    bool updated;

    // if this is set, quit
    bool quit;

} control_update_t;

// states of the keyboard
unsigned char * key_state;

SDL_Joystick * joystick;

int last_update_ticks;


void control_update_init();

control_update_t control_update_loop();

#endif

