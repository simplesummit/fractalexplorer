/* render.c -- this is ran on the head node, and is responsible for
            -- communication and screen rendering

  This file is part of the fractalexplorer project.

  fractalexplorer source code, as well as any other resources in this
project are free software; you are free to redistribute it and/or modify them
under the terms of the GNU General Public License; either version 3 of the
license, or any later version.

  These programs are hopefully useful and reliable, but it is understood
that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
<cade@cade.site> for more info on this.

  Here is a copy of the GPL v3, which this software is licensed under. You
can also find a copy at http://www.gnu.org/licenses/.

*/

#include "fractalexplorer.h"
#include "calc_c.h"
#include "render.h"

#include <lz4.h>

#include <mpi.h>

#include <math.h>

#include <SDL.h>
#include <SDL_ttf.h>


// the normalization value of SDL axis. If a joystick is all the way forward,
// its value will be +AXIS_MAX, if it is all the way back, it will be -AXIS_MAX
// thus, to linearly scale between -1.0 and 1.0, divide the input by AXIS_MAX
#define AXIS_MAX (32717.0f)


// SDL error handling macro. if the SDL function returns an error, it prints
// out the integer code, and prints the SDL error string for last error.
// sdl_hndl_res is the variable used
int sdl_hndl_res;
#define SDL_HNDL(x) sdl_hndl_res = x; if (sdl_hndl_res < 0) {  \
  log_error("While executing %s, SDL error (code %d): %s", #x, sdl_hndl_res, SDL_GetError()); exit(0); \
}

// this 'smashes' x down to 0 if abs(x)<(e), or just evaluates to x
#define SMASH(x, e) ((int)floor(((x) <= (e) && (x) >= -(e)) ? (0) : (x)))

// the font size for SDL rendering. Eventually (possibly) this should be
// relative to window size
#define FONT_SIZE           30


// the last full-cycle FPS (what the user sees)
double last_fps = 0.0;

// stores the ratio of compressed data to expanded data
double compress_rate = 0.0;

// to store timing and performance data (see fr.h for this type)
tperf_t tperf_render;

// an RGBA array of pixels to read in from compute nodes
unsigned char * pixels;

// a global hash, so we know whether to update the fractal
unsigned int hash;

// the previous x and y coordinates
double last_x, last_y;

// whether or not we are currently using a joystick
#define USE_JOYSTICK ((joystick) != NULL)

// event to loop through with SDL_PollEvent
SDL_Event cevent;

// a pointer to a joystick
SDL_Joystick *joystick = NULL;


// joystick axis numbers, should be found out using joytest or similar programs
// these were found for the Logitech DualAction
#define CONTROLLER_HORIZONTAL_AXIS      0
#define CONTROLLER_VERTICAL_AXIS        1
#define CONTROLLER_ZOOM_AXIS            3

// typed axis ids
SDL_GameControllerAxis horiz = CONTROLLER_HORIZONTAL_AXIS,
                       vert = CONTROLLER_VERTICAL_AXIS,
                       zaxis = CONTROLLER_ZOOM_AXIS;


// pointer to SDL render structures to use
SDL_Window *window;
SDL_Surface *surface;
SDL_Surface *screen;
TTF_Font *font;
SDL_Surface * tsurface;
SDL_Rect offset;

// this is the color of the info text at the top left of the screen
SDL_Color text_color = { 255, 255, 255 };

// the message lengths for strings
#define MAX_ONSCREEN_MESSAGE   (100 + 10 * 4)
#define NUM_ONSCREEN_MESSAGE   (4)

// pointer to onscreen message strings
char ** onscreen_message = NULL;


// hash function to determine whether fr has changed
unsigned int hash_fr(fr_t fr) {
    return (int)floor( fr.Z + fr.w * (fr.h + fr.Z) - fr.cX - fr.cY + sin(fr.cX + fr.Z * fr.cY / fr.w) + fr.h);
}

// requests picture from compute nodes
void gather_picture() {
    // timing structures
    tperf_t tp_bc, tp_rv;
    // pe is how much should be computed by each worker
    int i, pe = fr.h / fr.num_workers;
    // naddr never points to its own memory, just as an offset to the global
    // pixels array. Thus, free() should never be called with naddr
    unsigned char * naddr,
    // cmp_bytes
                  * cmp_bytes = (unsigned char *)malloc(LZ4_compressBound(fr.mem_w * pe));
    int nbytes, cmp_nbytes;

    // we use this for printing out statistics
    double total_bytes = 0, total_compressed_bytes = 0;

    C_TIME(tp_bc,
    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
    )

    C_TIME(tp_rv,
    // loop through all workers
    for (i = 1; i <= fr.num_workers; ++i) {
        // get the offset into the final
        naddr = pixels + pe * fr.mem_w * (i - 1);
        nbytes = fr.mem_w * pe;

        MPI_Recv(&cmp_nbytes, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        log_trace("recv from %d (compressed), size: %d", i, cmp_nbytes);
        MPI_Recv(cmp_bytes, cmp_nbytes, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        LZ4_decompress_safe((char *)cmp_bytes, (char*)naddr, cmp_nbytes, nbytes);
        log_trace("%%%lf of final size (worker %d)", 100.0 * cmp_nbytes / nbytes, i);
        total_bytes += nbytes;
        total_compressed_bytes += cmp_nbytes;
    }
    )
    compress_rate = total_compressed_bytes / total_bytes;
    memcpy(surface->pixels, pixels, fr.mem_w * fr.h);
    //log_trace("MPI_Bcast(fr) fps: %lf", 1.0 / tp_bc.elapsed_s);
    // print out debug info
    log_trace("MPI_Recv(pixels) fps: %lf", 1.0 / tp_rv.elapsed_s);
}

// refreshes the whole window, recalculating if needed
void window_refresh() {
    if (hash == hash_fr(fr)) {
        return;
    }
    tperf_t tp_wr;

    offset = (SDL_Rect){0, 0, 0, 0};

    C_TIME(tp_wr,
    if (pixels == NULL) {
        log_debug("malloc'ing render pixels");
        if (pixels != NULL) {
            //free(pixels);
        }
        pixels = (unsigned char *)malloc(fr.mem_w * fr.h);
        memset(pixels, 0, fr.mem_w * fr.h);
    }


    // GET PIXEL DATA HERE
    gather_picture();
    //mand_c(fr.w, fr.h, fr.cX, fr.cY, fr.Z, fr.max_iter, pixels);

    draw();

    // first time through, allocated enough messages
    if (onscreen_message == NULL) {
        onscreen_message = malloc(NUM_ONSCREEN_MESSAGE * sizeof(char *));
        int i;
        for (i = 0; i < NUM_ONSCREEN_MESSAGE; ++i) {
            onscreen_message[i] = malloc(MAX_ONSCREEN_MESSAGE);
        }
    }
    // onscreen messages
    sprintf(onscreen_message[0], "fps: %2.1lf", last_fps);
    sprintf(onscreen_message[1], "compression rate: %1.2lf", compress_rate);

    tsurface = TTF_RenderText_Solid(font, onscreen_message[0], text_color);
    SDL_HNDL(SDL_BlitSurface(tsurface, NULL, screen, &offset));
    offset.y += FONT_SIZE;
    tsurface = TTF_RenderText_Solid(font, onscreen_message[1], text_color);

    // upload the pixels to the surface
    SDL_HNDL(SDL_BlitSurface(tsurface, NULL, screen, &offset));

    // do a screen refresh
    SDL_HNDL(SDL_UpdateWindowSurface(window));
    )
    // print out debug info
    log_info("window_refresh() fps: %lf", 1.0 / tp_wr.elapsed_s);
    last_fps = 1.0 / tp_wr.elapsed_s;
    //draw();
}


// our main method
void fractalexplorer_render(int * argc, char ** argv) {

    // SDL initialization
    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO)) {
        log_fatal("Could not initialize SDL: %s", SDL_GetError());
        exit(1);
    }

    // make sure to quit and shut down libraries
    atexit(SDL_Quit);

    int __res = 0;
    if ((__res = TTF_Init()) < 0) {
        log_fatal("Could not initialize SDL ttf: %d", __res);
        exit(1);
    }

    // make sure to quit and shut down TTF libraries
    atexit(TTF_Quit);

    // try to open our default font
    font = TTF_OpenFont("UbuntuMono-R.ttf", FONT_SIZE);
    if (font == NULL) {
        log_debug("TTF_OpenFont(UbuntuMono-R.ttf) Failed: %s", TTF_GetError());
        font = TTF_OpenFont("Ubuntu-R.ttf", FONT_SIZE);
        if (font == NULL) {
            log_debug("TTF_OpenFont(Ubuntu-R.ttf) Failed: %s", TTF_GetError());
            font = TTF_OpenFont("UbuntuMono.ttf", FONT_SIZE);
            if (font == NULL) {
                log_debug("TTF_OpenFont(UbuntuMono.ttf) Failed: %s", TTF_GetError());
                log_fatal("Could not find a suitable font!");
                exit(1);
            }
        }
    }

    log_debug("%i joysticks were found, using joystick[0]\n\n", SDL_NumJoysticks());
    //log_debug("The names of the joysticks are:\n");
    if (SDL_NumJoysticks() > 0) {
        //controller = SDL_GameControllerOpen(0);
        joystick = SDL_JoystickOpen(0);
        log_debug("Controller name: %s\n", SDL_JoystickName(joystick));
        if (joystick == NULL) {
            log_error("opening joystick 0 failed: %s", SDL_GetError());
        }
    }

    if (USE_JOYSTICK) {
        log_info("using joystick");
    } else {
        log_info("not using joystick");
    }


    window = SDL_CreateWindow("Mandelbrot Render", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, fr.w, fr.h, 0);
    if (fr.w == 0 || fr.h == 0 || use_fullscreen) {
        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP); // SDL_WINDOW_FULLSCREEN_DESKTOP, or SDL_WINDOW_FULLSCREEN
    }
    // in case fullscreen changes it
    SDL_GetWindowSize(window, &fr.w, &fr.h);

    screen = SDL_GetWindowSurface(window);

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE, fr.w, fr.h, 32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000);


    if (surface == NULL) {
        log_error("SDL failed to create surface: %s", SDL_GetError());
    }

    fr.mem_w = surface->pitch;

    window_refresh();

    pixels = NULL;

    bool keep_going = true;
    bool inner_keep_going = true;
    bool update = true;

    bool r_down = false, l_down = false;
    bool s_down = false;
    int horiz_v = 0, vert_v = 0, zoom_v = 0;

    int last_ticks = SDL_GetTicks();

    while (keep_going == true) {
        //log_trace("outer loop");
        update = false;
        inner_keep_going = true;
        if (USE_JOYSTICK) {
            update = horiz_v != 0 || vert_v != 0 || zoom_v != 0;
            if (update) {
                double scale_allinput = (double)(SDL_GetTicks() - last_ticks) / 1000.0;
                fr.cX += scale_allinput * horiz_v / (AXIS_MAX * fr.Z);
                fr.cY -= scale_allinput * vert_v / (AXIS_MAX * fr.Z);
                double zfact = 1.0 + scale_allinput * abs(zoom_v) / AXIS_MAX;
                if (zoom_v > 0) {
                    fr.Z /= zfact;
                } else if (zoom_v < 0) {
                    fr.Z *= zfact;
                }
                last_ticks = SDL_GetTicks();
            }
            /*
            if (update) {
                log_trace("diff: %d,%d,%d", horiz_v, vert_v, zoom_v);
            }
            */
        }

        while (SDL_PollEvent(&cevent)) {
            if (inner_keep_going) {
                switch (cevent.type) {
                    case SDL_JOYAXISMOTION:
                        log_trace("joystick axis");
                        if (cevent.jaxis.axis == horiz) {
                            horiz_v = SMASH(cevent.jaxis.value, AXIS_MAX * .05f);
                        }
                        if (cevent.jaxis.axis == vert) {
                            vert_v = SMASH(cevent.jaxis.value, AXIS_MAX * .05f);
                        }
                        if (cevent.jaxis.axis == zaxis) {
                            zoom_v = SMASH(cevent.jaxis.value, AXIS_MAX * .05f);
                        }
                        break;
                    case SDL_QUIT:
                        log_info("SDL_Quit event");
                        keep_going = false;
                        inner_keep_going = false;
                        break;
                    case SDL_KEYUP:
                        if (cevent.key.keysym.sym == SDLK_LSHIFT || cevent.key.keysym.sym == SDLK_RSHIFT) {
                            s_down = false;
                        }
                        break;
                    case SDL_KEYDOWN:
                        if (cevent.key.keysym.sym == ' ') {
                            if (s_down) {
                                fr.Z /= 1.5;
                            } else {
                                fr.Z *= 1.5;
                            }
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_LSHIFT || cevent.key.keysym.sym == SDLK_RSHIFT) {
                            s_down = true;
                            //fr.Z /= 1.5;
                        } else if (cevent.key.keysym.sym == SDLK_LEFT) {
                            fr.cX -= .15 / fr.Z;
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_RIGHT) {
                            fr.cX += .15 / fr.Z;
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_UP) {
                            fr.cY += .15 / fr.Z;
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_DOWN) {
                            fr.cY -= .15 / fr.Z;
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_ESCAPE) {
                            keep_going = false;
                            inner_keep_going = false;
                        } else if (cevent.key.keysym.sym == 'p') {
                            fr.max_iter += 1;
                            update = true;
                        } else if (cevent.key.keysym.sym == 'o') {
                            if (fr.max_iter > 0) {
                                fr.max_iter -= 1;
                                update = true;
                            }
                        } else if (cevent.key.keysym.sym == 'k' && cevent.key.repeat == 0) {
                            if (fr.num_workers < compute_size) {
			                          fr.num_workers++;
                                update = true;
                            }
                        } else if (cevent.key.keysym.sym == 'j' && cevent.key.repeat == 0) {
                            if (fr.num_workers > 1) {
			                    fr.num_workers--;
                                update = true;
                            }
                        } else if (cevent.key.keysym.sym == 'm' && cevent.key.repeat == 0) {
                            fractal_types_idx = (fractal_types_idx + 1) % FR_FRACTAL_NUM;
                            fr.fractal_type = fractal_types[fractal_types_idx];
                            update = true;
                        } else if (cevent.key.keysym.sym == 'n' && cevent.key.repeat == 0) {
                            fractal_types_idx = (fractal_types_idx - 1 + FR_FRACTAL_NUM) % FR_FRACTAL_NUM;
                            fr.fractal_type = fractal_types[fractal_types_idx];
                            update = true;
                        }
                        break;
                    case SDL_MOUSEMOTION:
                        if (l_down) {
                            fr.cX -= cevent.motion.xrel / (fr.Z * fr.w);
                            fr.cY += cevent.motion.yrel / (fr.Z * fr.w);
                            update = true;
                        }
                        if (r_down) {
                            double zoomin = 1 + fabs(2.0 * (cevent.motion.yrel + cevent.motion.xrel) / (fr.w + fr.h));
                            if (!s_down) {
                                fr.Z *= zoomin;
                            } else {
                                fr.Z /= zoomin;
                            }
                            update = true;
                        }
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        if (cevent.button.button == SDL_BUTTON_LEFT) {
                            l_down = true;
                        } else if (cevent.button.button == SDL_BUTTON_RIGHT) {
                            r_down = true;
                        }
                        break;
                    case SDL_MOUSEBUTTONUP:
                        if (cevent.button.button == SDL_BUTTON_LEFT) {
                            l_down = false;
                        } else if (cevent.button.button == SDL_BUTTON_RIGHT) {
                            r_down = false;
                        }
                        break;
                    default:
                        //
                        break;
                }
            }
            SDL_PumpEvents();
            //inner_do = false;
        }
        if (!keep_going) {
            break;
        }
        if (update) {
            log_trace("recomputing fractal");
            window_refresh();
            //_fr_interactive_sdl_recompute(fr, fr_engine);
        }
        //do_update = false;
    }

    log_info("quitting now");

    exit(0);
}

void draw() {
    screen = SDL_GetWindowSurface(window);

    SDL_HNDL(SDL_BlitSurface(surface, NULL, screen, NULL));
}
