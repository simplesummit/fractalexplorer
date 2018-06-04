// for controlling the fractal parameters

#ifndef __CONTROL_LOOP_H__
#define __CONTROL_LOOP_H__

#include <stdbool.h>

typedef struct control_update_t {
    // tells what updated

    bool updated;

    // if this is set, quit
    bool quit;

} control_update_t;


control_update_t control_update_loop();

#endif

