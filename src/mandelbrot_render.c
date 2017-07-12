//

#include "mandelbrot.h"
#include "mandelbrot_calc_c.h"
#include "mandelbrot_render.h"

#include <mpi.h>

#include <math.h>

#include <SDL.h>


#define AXIS_MAX 32717.0f
#define SDL_HNDL(x) res = x; if (res < 0) { log_fatal("While executing %s, SDL error: %s", #x, SDL_GetError()); }


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
    int i, pe = fr.h / compute_size;
    unsigned char * naddr;
    int nbytes;
    C_TIME(tp_bc,
    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
    )
    C_TIME(tp_rv,
    for (i = 1; i <= compute_size; ++i) {
        naddr = pixels + pe * fr.mem_w * (i - 1);
        nbytes = fr.mem_w * pe;
        printf("%d\n", fr.mem_w * pe);
        log_trace("recv from %d, naddr-pixels: %d, size: %d", i, (int)(naddr-pixels), nbytes);
        MPI_Recv(naddr, nbytes, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    while (keep_going == true) {
        //log_trace("outer loop");
        update = false;
        inner_keep_going = true;
        while (SDL_PollEvent(&cevent)) {
            if (inner_keep_going) {

                switch (cevent.type) {
                    case SDL_JOYAXISMOTION:
                        if (USE_JOYSTICK) {
                            log_trace("joystick axis");
                            if (cevent.jaxis.axis == horiz) {
                                log_trace("horiz");
                                fr.cX += .05 * cevent.jaxis.value / (AXIS_MAX * fr.Z);
                                update = true;
                            }
                            if (cevent.jaxis.axis == vert) {
                                fr.cY -= .05 * cevent.jaxis.value / (AXIS_MAX * fr.Z);
                                update = true;
                            }
                            if (cevent.jaxis.axis == zaxis) {
                                double zfact = 1.0 + .15 * fabs(cevent.jaxis.value) / AXIS_MAX;
                                if (cevent.jaxis.value > 0) {
                                    fr.Z /= zfact;
                                } else {
                                    fr.Z *= zfact;
                                }
                                update = true;
                            }
                             
                        }
                        break;
                    case SDL_QUIT:
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

    //window_refresh(fr.w, fr.h);
  //  glutMainLoop();
}

void draw() {

    screen = SDL_GetWindowSurface(window);

    SDL_HNDL(SDL_BlitScaled(surface, NULL, screen, NULL));
    //SDL_BlitSurface(surface, NULL, screen, NULL);

    SDL_HNDL(SDL_UpdateWindowSurface(window));
    
    /*
    C_TIME(tperf_render, 
    glDrawPixels(fr.w, fr.h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, pixels);
    glutSwapBuffers();
    );
    double fps = 1.0 / tperf_render.elapsed_s;
    if (fps < 15) {
        log_warn("draw() fps is low: %lf", fps);
    }*/
    //log_trace("draw() fps: %lf", fps);
}
/*
void async_draw(int te) {
    glutPostRedisplay();

    glutTimerFunc( 10, async_draw, 1);
}

void idle_handler() {
    glutPostRedisplay();
}

void key_handler(unsigned char key, int x, int y) {
    log_trace("key '%c' pressed at %d,%d", key, x, y);
}*/

/*
void motion_handler(int x, int y) {
    bool do_refresh = true;
    if (last_bt == GLUT_RIGHT_BUTTON) {
	double zoomin = 1 + fabs(2.0 * ((y - last_y) + (x - last_x)) / (fr.w + fr.h));
	if (GLUT_ACTIVE_CTRL) {
            fr.Z *= zoomin;
	} else {
            fr.Z *= zoomin;
	}  
    } else if(last_bt == GLUT_LEFT_BUTTON) {
        fr.cX = fr.cX - (x - last_x) / (fr.Z * fr.w);
        fr.cY = fr.cY - (y - last_y) / (fr.Z * fr.h);
    } else {
        do_refresh = false;
    }
    if (do_refresh) {
        window_refresh(fr.w, fr.h);
    }
    last_x = x; last_y = y;

}

void mouse_handler(int button, int state, int x, int y) {
    last_bt = button;
    last_x = x; last_y = y;
}

void reshape_handler(GLint w, GLint h) {
    window_refresh(w, h);

}

*/