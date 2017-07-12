//

#include "mandelbrot.h"
#include "mandelbrot_calc_c.h"
#include "mandelbrot_render.h"

#include <lz4.h>

#include <mpi.h>

#include <math.h>

#include <SDL.h>


#define AXIS_MAX 32717.0f
#define SDL_HNDL(x) res = x; if (res < 0) { log_fatal("While executing %s, SDL error: %s", #x, SDL_GetError()); }

// smashes down x if abs(x) <= e
#define SMASH(x, e) ((int)floor(((x) <= (e) && (x) >= -(e)) ? (0) : (x)))


tperf_t tperf_render;

//unsigned int prog, texture;

unsigned char * pixels;

unsigned int hash;

int res;

float last_x, last_y;


#define USE_JOYSTICK ((joystick) != NULL)

SDL_Event cevent;

SDL_Joystick *joystick = NULL;


SDL_GameControllerAxis horiz = 0, vert = 1, zaxis = 3;

SDL_Window *window;
SDL_Surface *surface;
SDL_Surface *screen;

//int last_bt;

//void async_draw(int);

unsigned int hash_fr(fr_t fr) {
    return (int)floor( fr.Z + fr.w * (fr.h + fr.Z) - fr.cX - fr.cY + sin(fr.cX + fr.Z * fr.cY / fr.w) + fr.h);
}

void gather_picture() {
    tperf_t tp_bc, tp_rv;
    int i, pe = fr.h / fr.num_workers;
    unsigned char * naddr, * cmp_bytes = (unsigned char *)malloc(LZ4_compressBound(fr.mem_w * pe)); 
    int nbytes, cmp_nbytes;

    C_TIME(tp_bc,
    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
    )
    C_TIME(tp_rv,
    for (i = 1; i <= fr.num_workers; ++i) {
        naddr = pixels + pe * fr.mem_w * (i - 1);
        nbytes = fr.mem_w * pe;

        MPI_Recv(&cmp_nbytes, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        log_trace("recv from %d (compressed), size: %d", i, cmp_nbytes);
        cmp_bytes = (unsigned char *)malloc(cmp_nbytes);
        MPI_Recv(cmp_bytes, cmp_nbytes, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        LZ4_decompress_safe((char *)cmp_bytes, (char*)naddr, cmp_nbytes, nbytes);        
        log_trace("%%%lf of final size (worker %d)", 100.0 * cmp_nbytes / nbytes, i);
    }
    )
    memcpy(surface->pixels, pixels, fr.mem_w * fr.h);
    //log_trace("MPI_Bcast(fr) fps: %lf", 1.0 / tp_bc.elapsed_s);
    log_trace("MPI_Recv(pixels) fps: %lf", 1.0 / tp_rv.elapsed_s);
}


void window_refresh() {
    if (hash == hash_fr(fr)) {
        return;
    }
    tperf_t tp_wr;
    C_TIME(tp_wr,
    //if (w != fr.w || h != fr.h || pixels == NULL) {
        log_debug("remallocing render pixels");
        if (pixels != NULL) {
            //free(pixels);
        }
        pixels = (unsigned char *)malloc(fr.mem_w * fr.h);
    //}


    // GET PIXEL DATA HERE
    gather_picture();
    //mand_c(fr.w, fr.h, fr.cX, fr.cY, fr.Z, fr.max_iter, pixels);

    draw();

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA, fr.w, fr.h, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, pixels);
    //glutSwapBuffers();

    )
    log_info("window_refresh() fps: %lf", 1.0 / tp_wr.elapsed_s);
    //draw();

}


void mandelbrot_render(int * argc, char ** argv) {

    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO)) {
        log_fatal("Could not initialize SDL: %s", SDL_GetError());
        exit(1);
    }


    log_debug("%i joysticks were found, using joystick[0]\n\n", SDL_NumJoysticks() );
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

    atexit(SDL_Quit);

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
                            fr.Z *= 1.5;
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
                        } else if (cevent.key.keysym.sym == 'k') {
                            if (fr.num_workers < compute_size) {
			        fr.num_workers++;
                                update = true;
                            }
                        } else if (cevent.key.keysym.sym == 'j') {
                            if (fr.num_workers > 1) {
			        fr.num_workers--;
                                update = true;
                            }
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

    //window_refresh(fr.w, fr.h);
  //  glutMainLoop();
}

void draw() {

    screen = SDL_GetWindowSurface(window);

    SDL_HNDL(SDL_BlitScaled(surface, NULL, screen, NULL));
    //SDL_BlitSurface(surface, NULL, screen, NULL);

    SDL_HNDL(SDL_UpdateWindowSurface(window));

}


