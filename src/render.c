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

// signum function
#define sgn(x) ((x) > 0 ? 1 : ((x) < 0) ? -1 : 0)


// SDL error handling macro. if the SDL function returns an error, it prints
// out the integer code, and prints the SDL error string for last error.
// sdl_hndl_res is the variable used
int sdl_hndl_res;
#define SDL_HNDL(x) sdl_hndl_res = x; if (sdl_hndl_res < 0) {  \
  log_error("While executing %s, SDL error (code %d): %s", #x, sdl_hndl_res, SDL_GetError()); exit(0); \
}

// this 'smashes' x down to 0 if abs(x)<(e), or just evaluates to x
// useful for joystick inputs so no slight drift keeps forcing recomputation
#define SMASH(x, e) ((int)floor(((x) <= (e) && (x) >= -(e)) ? (0) : (x)))

// the font size for SDL rendering. Eventually (possibly) this should be
// relative to window size
#define MIN(a, b) ((a) > (b) ? (a) : (b) )
#define FONT_SIZE (MIN(12, fr.h/34))

// graph width and height (bottom right)
#define GRAPH_W ((int)floor(12 * (FONT_SIZE)))
#define GRAPH_H ((int)floor(GRAPH_W / 2.0))

#define LEGEND_W GRAPH_W
#define LEGEND_H FONT_SIZE


// the last full-cycle FPS (what the user sees)
double last_fps = 0.0, last_draw_fps = 0.0, last_compute_fps = 0.0,
       last_transfer_fps = 0.0, last_decompress_fps = 0.0,
       last_last_fps = 0.0;

double last_graph_scale = 0.0;

// array of graph scales per pixel
double * graph_scale_array = NULL;
int graph_scale_array_idx = 0;

bool has_graphed = false;
double graph_scale = 0.0;


// stores the ratio of compressed data to expanded data
double compress_rate = 0.0;

// to store timing and performance data (see fr.h for this type)
tperf_t tperf_render;

// an RGBA array of pixels to read in from compute nodes
unsigned char * pixels = NULL;

// a global hash, so we know whether to update the fractal
unsigned int hash;

// whether or not to draw the information
bool show_text_info = true;
bool show_graph = true;
bool show_extra = true;

// the previous x and y coordinates
double last_x, last_y;

// whether or not we are currently using a joystick
#define USE_JOYSTICK ((joystick) != NULL)

// event to loop through with SDL_PollEvent
SDL_Event cevent;

// a pointer to a joystick
SDL_Joystick *joystick = NULL;

// timeout to demo
int timeout = 10;

// current animation
int curr_anim = 0;

typedef struct anim_param_t {

    int fractal_id;

    // parameters to pan and zoom to
    double cX, cY, Z;
    double u, v;

    // the animation duration
    int ticks;
} anim_param_t;

int num_animations;
anim_param_t * animations;


// joystick axis numbers, should be found out using joytest or similar programs
// these were found for the Logitech DualAction
#define CONTROLLER_HORIZONTAL_AXIS      0
#define CONTROLLER_VERTICAL_AXIS        1
#define CONTROLLER_ZOOM_AXIS            3

// tweaks fr.u parameter
#define CONTROLLER_U_AXIS               2



// typed axis ids
SDL_GameControllerAxis horiz = CONTROLLER_HORIZONTAL_AXIS,
                       vert = CONTROLLER_VERTICAL_AXIS,
                       zaxis = CONTROLLER_ZOOM_AXIS,
                       uaxis = CONTROLLER_U_AXIS;


// pointer to SDL render structures to use
SDL_Window *window;
SDL_Surface *surface;
SDL_Surface *screen;
SDL_Texture *texture;
SDL_Texture *graph_texture;
TTF_Font *font;
SDL_Surface * tsurface;
SDL_Rect offset;
SDL_Renderer * renderer;
SDL_Texture * message_texture;
SDL_Texture * legend_texture;

// texture memory
unsigned char * graph_texture_pixels = NULL;
unsigned char * legend_texture_pixels = NULL;


// current pixel on the graph
int graph_cpixel = 0;



int * recv_nbytes = NULL;
unsigned char ** recv_bytes = NULL, ** recv_compressed_bytes = NULL;
MPI_Request * recv_requests = NULL;
MPI_Status * recv_status = NULL;


// this is the color of the info text at the top left of the screen
SDL_Color text_color = { 0, 0, 0 },
          text_box_color = {255, 255, 255 };

// the message lengths for strings
#define MAX_ONSCREEN_MESSAGE   (100 + 30 * 5)
#define NUM_ONSCREEN_MESSAGE   (10)

// pointer to onscreen message strings
char ** onscreen_message = NULL;

// requests picture from compute nodes
void gather_picture() {
    // pe is how much should be computed by each worker
    int i, j;
    // naddr never points to its own memory, just as an offset to the global
    // pixels array. Thus, free() should never be called with naddr
    unsigned char * naddr;
    if (recv_nbytes == NULL) {
        log_debug("initializing buffers for storing compressed/computed pixels");
        recv_nbytes = malloc(sizeof(int *) * compute_size);
        recv_compressed_bytes = malloc(sizeof(unsigned char *) * compute_size);
        recv_bytes = malloc(sizeof(unsigned char *) * compute_size);
        recv_requests = malloc(sizeof(MPI_Request) * compute_size);
        recv_status = malloc(sizeof(MPI_Status) * compute_size);
        for (i = 0; i < compute_size; ++i) {
            // this should be the maximum ever needed
            recv_bytes[i] = malloc(4 * fr.w * fr.h);
            recv_compressed_bytes[i] = malloc(LZ4_compressBound(4 * fr.w * fr.h));
        }
    }

    int bytes_per_compute = 4 * fr.w * fr.h / fr.num_workers;

    double total_compressed_bytes = 0;

    tperf_t tp_compute;
    tperf_t tp_recv, tp_decompress;

    C_TIME(tp_compute,
        MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    )

    // if getting lingering artifacts, make sure to clear the buffer
    //memset(pixels, 0, 4 * fr.w * fr.h);

    C_TIME(tp_recv,
        // loop through all workers
        for (i = 0; i < fr.num_workers; ++i) {
            MPI_Recv(&recv_nbytes[i], 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Irecv(recv_compressed_bytes[i], abs(recv_nbytes[i]), MPI_UNSIGNED_CHAR, i + 1, 0, MPI_COMM_WORLD, &recv_requests[i]);
        }

        for (i = 0; i < fr.num_workers; ++i) {
            MPI_Wait(&recv_requests[i], &recv_status[i]);
        }
    )

    C_TIME(tp_decompress,
        for (i = 0; i < fr.num_workers; ++i) {
            if (recv_nbytes[i] > 0) {
                int decompress_err = LZ4_decompress_safe((char *)recv_compressed_bytes[i], (char*)recv_bytes[i], abs(recv_nbytes[i]), bytes_per_compute);
                if (decompress_err < 0) {
                    log_error("decompression error: %d", decompress_err);
                    exit(3);
                }
            } else {
                memcpy(recv_bytes[i], recv_compressed_bytes[i], bytes_per_compute);
            }

            for (j = i; j < fr.h; j += fr.num_workers) {
                memcpy(pixels + 4 * (j * fr.w), recv_bytes[i] + 4 * (fr.w * ((j-i) / fr.num_workers)), 4 * fr.w);
            }
            total_compressed_bytes += abs(recv_nbytes[i]);
        }
    )

    MPI_Barrier(MPI_COMM_WORLD);
    last_transfer_fps = 1.0 / tp_recv.elapsed_s;
    last_decompress_fps = 1.0 / tp_decompress.elapsed_s;
    last_compute_fps = 1.0 / tp_compute.elapsed_s;
    compress_rate = total_compressed_bytes / (4 * fr.w * fr.h);
    log_info("Mb/s: %.2lf", total_compressed_bytes / (1e6 * tp_recv.elapsed_s));
    log_debug("compute tfps: %.2lf, recv tfps: %.2lf, decompress tfps: %.2lf, compression ratio: %.2lf", last_compute_fps, last_transfer_fps, last_decompress_fps, compress_rate);
}

// refreshes the whole window, recalculating if needed
void window_refresh() {
    tperf_t tp_wr, tp_gp, tp_draw, tp_textdraw;
    SDL_Rect text_box_offset = (SDL_Rect){0, 0, 0, 0};
    offset = (SDL_Rect){0, 0, 0, 0};
    offset.w = fr.w / 6;
    offset.h = fr.h / 6;


    SDL_Color compute_color = { 255, 0, 0 };
    SDL_Color transfer_color = { 0, 255, 0 };
    SDL_Color decompress_color = { 255, 0, 255 };
    SDL_Color draw_color = { 0, 255, 255 };

    SDL_Rect graph_offset = (SDL_Rect){0, 0, 0, 0};
    graph_offset.w = GRAPH_W;
    graph_offset.h = GRAPH_H;
    graph_offset.x = fr.w - graph_offset.w;
    graph_offset.y = fr.h - graph_offset.h;

    SDL_Rect legend_offset = (SDL_Rect){0, 0, 0, 0};
    legend_offset.w = LEGEND_W;
    legend_offset.h = LEGEND_H;
    legend_offset.x = fr.w - legend_offset.w;
    legend_offset.y = fr.h - graph_offset.h - LEGEND_H;


    int i, j, ri, ri_s, ri_d;

    // get the window surface again, just in case something changed
    //screen = SDL_GetWindowSurface(window);

    if (pixels == NULL) {
        log_trace("malloc'ing render pixels");
        pixels = (unsigned char *)malloc(4 * fr.w * fr.h);
        memset(pixels, 0, 4 * fr.w * fr.h);
    }


    // time how long it takes to: ask for an image, let compute nodes
    // run, transfer back compressed/uncompressed data, and then
    // combine it into the global pixels array
    C_TIME(tp_gp,
        gather_picture();
    )

    C_TIME(tp_draw,
        // start rendering, we need to clear the render instance
        SDL_RenderClear(renderer);
        //bool scale_wholegraph = false;
        if (show_graph) {
        double total_time = 1.0 / last_fps;
        if (isinf(total_time)) total_time = 0.03;
        double ltotal_time = 1.0 / last_last_fps;

        graph_scale = total_time;

        for (i = 0; i < GRAPH_W; ++i) {
             if (graph_scale_array[i] > graph_scale && graph_scale_array[i] < 10.0) {
                 graph_scale = graph_scale_array[i];
             }
        }

        if (!has_graphed) {
            graph_scale = 0.03;
            last_graph_scale = 0.03;
        }

        graph_scale_array[graph_scale_array_idx] = total_time;

//        log_trace("graph scale: %lf, lgraph: %lf", graph_scale, lgraph_scale);

        graph_scale_array_idx = (graph_scale_array_idx + 1) % GRAPH_W;
        has_graphed = true;

        //strncpy(graph_texture_pixels, graph_texture_pixels + (4 * GRAPH_H), 4 * (GRAPH_H-1)*GRAPH_W);
        for (i = 1; i < GRAPH_W; ++i) {
            for (j = 0; j < GRAPH_H; ++j) {
                ri_d = 4 * ((GRAPH_H - j - 1) * GRAPH_W + i - 1);
                if (GRAPH_H - (int)floor(j * graph_scale / last_graph_scale) - 1 < 0) {
                    graph_texture_pixels[ri_d + 0] = 0;
                    graph_texture_pixels[ri_d + 1] = 0;
                    graph_texture_pixels[ri_d + 2] = 0;
                    graph_texture_pixels[ri_d + 3] = 200;
                } else {
                    ri_s = 4 * ((GRAPH_H - (int)floor(j * graph_scale / last_graph_scale) - 1) * GRAPH_W + i);
                    graph_texture_pixels[ri_d + 0] = graph_texture_pixels[ri_s + 0];
                    graph_texture_pixels[ri_d + 1] = graph_texture_pixels[ri_s + 1];
                    graph_texture_pixels[ri_d + 2] = graph_texture_pixels[ri_s + 2];
                    graph_texture_pixels[ri_d + 3] = graph_texture_pixels[ri_s + 3];
                }
            }
        }
        // so far graphed
        for (i = GRAPH_W - 1; i < GRAPH_W; ++i) {
            double sfg = graph_scale;/*
            for (j = GRAPH_H - 1; j >= GRAPH_H * (sfg - total_time) / graph_scale && j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // this (compute time) shows up in blue
                graph_texture_pixels[ri + 0] = 0;
                graph_texture_pixels[ri + 1] = 0;
                graph_texture_pixels[ri + 2] = 0;
                graph_texture_pixels[ri + 3] = 200;
            }
            sfg -= total_time;*/
            for (; j >= GRAPH_H * (sfg - 1.0 / last_compute_fps) / graph_scale && j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // this (compute time) shows up in blue
                graph_texture_pixels[ri + 0] = compute_color.b;
                graph_texture_pixels[ri + 1] = compute_color.g;
                graph_texture_pixels[ri + 2] = compute_color.r;
                graph_texture_pixels[ri + 3] = compute_color.a;
            }
            sfg -= 1.0 / last_compute_fps;
            for (; j >= GRAPH_H * (sfg - 1.0 / last_transfer_fps) / graph_scale && j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // transfer time shows up in green
                graph_texture_pixels[ri + 0] = transfer_color.b;
                graph_texture_pixels[ri + 1] = transfer_color.g;
                graph_texture_pixels[ri + 2] = transfer_color.r;
                graph_texture_pixels[ri + 3] = transfer_color.a;
            }
            sfg -= 1.0 / last_transfer_fps;
            for (; j >= GRAPH_H * (sfg - 1.0 / last_decompress_fps) / graph_scale && j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // decompression time shows up in red
                graph_texture_pixels[ri + 0] = decompress_color.b;
                graph_texture_pixels[ri + 1] = decompress_color.g;
                graph_texture_pixels[ri + 2] = decompress_color.r;
                graph_texture_pixels[ri + 3] = decompress_color.a;
            }
            sfg -= 1.0 / last_decompress_fps;
            for (; j >= GRAPH_H * (sfg - 1.0 / last_draw_fps) / graph_scale && j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // draw time shows up in white
                graph_texture_pixels[ri + 0] = draw_color.b;
                graph_texture_pixels[ri + 1] = draw_color.g;
                graph_texture_pixels[ri + 2] = draw_color.r;
                graph_texture_pixels[ri + 3] = draw_color.a;
            }
            sfg -= 1.0 / last_draw_fps;
            for (; j >= 0; j--) {
                ri = 4 * (j * GRAPH_W + i);
                // misc shows up in black
                graph_texture_pixels[ri + 0] = 0;
                graph_texture_pixels[ri + 1] = 0;
                graph_texture_pixels[ri + 2] = 0;
                graph_texture_pixels[ri + 3] = 255;
            }
        }
        //memset(graph_texture_pixels + 4 * (GRAPH_H-1) * GRAPH_W, 0, 4 * GRAPH_W);
        //graph_cpixel++;
        SDL_UpdateTexture(graph_texture, NULL, graph_texture_pixels, 4 * GRAPH_W);

        //SDL_RenderCopy(renderer, legend_texture, NULL, &graph_offset);


        //memset(legend_texture_pixels, 0, 4 * LEGEND_W * LEGEND_H);
        }

        SDL_UpdateTexture(texture, NULL, pixels, 4 * fr.w);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        if (show_extra && show_graph) {
           SDL_RenderCopy(renderer, graph_texture, NULL, &graph_offset);
           graph_offset.w = LEGEND_W;
           graph_offset.h = LEGEND_H;
           graph_offset.x = fr.w - GRAPH_W;
           graph_offset.y = fr.h - GRAPH_H;

           memset(legend_texture_pixels, 0, 4 * LEGEND_W * LEGEND_H);

           SDL_UpdateTexture(legend_texture, NULL, legend_texture_pixels, 4 * LEGEND_W);
           SDL_RenderCopy(renderer, legend_texture, NULL, &legend_offset);

           legend_offset.x += FONT_SIZE / 2;

#define ADD_TO_LEGEND(str, col) tsurface = TTF_RenderText_Solid(font, str, col); \
           message_texture = SDL_CreateTextureFromSurface(renderer, tsurface); \
           legend_offset.w = strlen(str) * FONT_SIZE / 2; \
           SDL_RenderCopy(renderer, message_texture, NULL, &legend_offset); \
           legend_offset.x += legend_offset.w + FONT_SIZE / 2; \
           SDL_FreeSurface(tsurface); \
           SDL_DestroyTexture(message_texture);


           ADD_TO_LEGEND("compute", compute_color);
           ADD_TO_LEGEND("io", transfer_color);
           ADD_TO_LEGEND("decomp", decompress_color);
           ADD_TO_LEGEND("draw", draw_color);
        }
 //   	SDL_RenderPresent(renderer);
    )
    last_draw_fps = 1.0 / tp_draw.elapsed_s;

    last_graph_scale = graph_scale;
    if (show_extra && show_text_info) {
        log_trace("showing text info");

        // first time through, allocated enough messages
        if (onscreen_message == NULL) {
            onscreen_message = malloc(NUM_ONSCREEN_MESSAGE * sizeof(char *));
            for (i = 0; i < NUM_ONSCREEN_MESSAGE; ++i) {
                onscreen_message[i] = malloc(MAX_ONSCREEN_MESSAGE);
            }
        }
        for (i = 0; i < NUM_ONSCREEN_MESSAGE; ++i) {
            sprintf(onscreen_message[i], "%s", "");
        }
        sprintf(onscreen_message[0], "%s", fractal_types_names[fractal_types_idx]);
        sprintf(onscreen_message[1], "fps: %.2lf", 1.0 / (tp_gp.elapsed_s + tp_draw.elapsed_s));
        sprintf(onscreen_message[2], "re(center): %+.14lf", fr.cX);
        sprintf(onscreen_message[3], "im(center): %+.14lf", fr.cY);
        sprintf(onscreen_message[4], "re(q) %+.14lf", fr.u);
        sprintf(onscreen_message[5], "im(q) %+.14lf", fr.v);
        sprintf(onscreen_message[6], "zoom: %.2e", fr.Z);
        sprintf(onscreen_message[7], "iter: %d", fr.max_iter);
        switch (fr.engine) {
            case FR_E_C:
                sprintf(onscreen_message[8], "engine: c, workers: %d", fr.num_workers);
                break;
            case FR_E_CUDA:
                sprintf(onscreen_message[8], "engine: cuda, workers: %d", fr.num_workers);
                break;
            default:
                break;
        }
        //sprintf(onscreen_message[6], "compute fps: %2.1lf", 1.0 / (tp_gp.elapsed_s));
        //sprintf(onscreen_message[7], "draw fps: %2.1lf", 1.0 / (tp_draw.elapsed_s + tp_textdraw.elapsed_s));
        C_TIME(tp_textdraw,
            int max_w = 0;
            for (i = 0; i < NUM_ONSCREEN_MESSAGE; ++i) {
                if (strlen(onscreen_message[i]) > 0) {
                    tsurface = TTF_RenderText_Solid(font, onscreen_message[i], text_color);
                    if (tsurface->w > max_w) max_w = tsurface->w;
                    text_box_offset.h += tsurface->h;
                    SDL_FreeSurface(tsurface);
                }
            }
            text_box_offset.w = max_w + FONT_SIZE / 2;
            text_box_offset.h += FONT_SIZE / 2;
            SDL_RenderFillRect(renderer, &text_box_offset);
            for (i = 0; i < NUM_ONSCREEN_MESSAGE; ++i) {
                if (strlen(onscreen_message[i]) > 0) {
                    tsurface = TTF_RenderText_Solid(font, onscreen_message[i], text_color);
                    offset.w = tsurface->w;
                    offset.h = tsurface->h;
                    message_texture = SDL_CreateTextureFromSurface(renderer, tsurface);
                    SDL_RenderCopy(renderer, message_texture, NULL, &offset);
                    offset.y += tsurface->h;
                    SDL_FreeSurface(tsurface);
                    SDL_DestroyTexture(message_texture);
                }
            }
        )

    }

    SDL_RenderPresent(renderer);

    last_last_fps = last_fps;
    last_fps = 1.0 / (tp_gp.elapsed_s + tp_draw.elapsed_s);

    // logging basic info
    log_info("fps: %.2lf, gather_picture() fps: %.2lf", last_fps, 1.0 / tp_gp.elapsed_s);
    log_debug("draw fps: %.2lf, text draw fps: %.2lf", last_draw_fps, 1.0 / tp_textdraw.elapsed_s);


}

#define FONT_ROOT "/usr/local/share/fonts/"
#define FONT_ROOT_2 "/usr/share/fonts/truetype/"

#define OFONTS(x) font = TTF_OpenFont(x, FONT_SIZE);
#define OFONT(x) OFONTS(x); if (font == NULL) { OFONTS(FONT_ROOT x) } if (font == NULL) { OFONTS(FONT_ROOT_2 x) }


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

    int window_flags = 0;
    if (fr.w == 0 || fr.h == 0 || use_fullscreen) {
        window_flags = window_flags | SDL_WINDOW_FULLSCREEN_DESKTOP;
    }

    window = SDL_CreateWindow("Mandelbrot Render", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, fr.w, fr.h, window_flags);
    
    // in case fullscreen changes it
    SDL_GetWindowSize(window, &fr.w, &fr.h);

    if (!show_cursor) SDL_ShowCursor(SDL_DISABLE);

    // try to open our default font

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


    OFONT("UbuntuMono.ttf");
    if (font == NULL) {
        OFONT("ubuntu-font-family/Ubuntu-R.ttf");
    }
    if (font == NULL) {
        log_error("couldn't find UbuntuMono font");
        exit(3);
    }

//    screen = SDL_GetWindowSurface(window);

   // surface = SDL_CreateRGBSurface(SDL_SWSURFACE, fr.w, fr.h, 32, 0xFF, 0xFF00, 0xFF0000, 0xFF000000);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 120);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, fr.w, fr.h);

    graph_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, GRAPH_W, GRAPH_H);
    legend_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, LEGEND_W, LEGEND_H);

    legend_texture_pixels = malloc(4 * LEGEND_W * LEGEND_H);
    memset(legend_texture_pixels, 0, 4 * LEGEND_W * LEGEND_H);

    FILE * path_fp = fopen(fractal_path_file, "r");

    if (path_fp == NULL) {
        log_info("not using path file, couldn't open %s", fractal_path_file);
        num_animations = 0;
    } else {
        #define FSF_CHK(n) if (fsf_res != n) { log_warn("parsing file failed, not using animations"); num_animations = 0; }
        log_debug("using path file: %s", fractal_path_file);
        int fsf_res = 0;
        anim_param_t ta;
        int k = 0;
        while (7 == (fsf_res = fscanf(path_fp, "%d,%lf,%lf,%lf,%lf,%lf,%d\n", &ta.fractal_id, &ta.cX, &ta.cY, &ta.Z, &ta.u, &ta.v, &ta.ticks))) {
               k++;
        }
        fclose(path_fp);
        path_fp = fopen(fractal_path_file, "r");
        num_animations = k;
        log_info("using %d animations", num_animations);
        animations = malloc(sizeof (anim_param_t) * num_animations);
        for (k=0;k<num_animations;k++) {
          fsf_res = fscanf(path_fp, "%d,%lf,%lf,%lf,%lf,%lf,%d\n", &animations[k].fractal_id, &animations[k].cX, &animations[k].cY, &animations[k].Z, &animations[k].u, &animations[k].v, &animations[k].ticks);
          log_debug("Animation point %d: %d,%lf,%lf,%lf,%lf,%lf,%d\n", k, animations[k].fractal_id, animations[k].cX, animations[k].cY, animations[k].Z, animations[k].u, animations[k].v, animations[k].ticks);
        }
        fclose(path_fp);
    }


    graph_scale_array = malloc(GRAPH_W * sizeof(double));
    int i;
    for (i = 0; i < GRAPH_W; ++i) {
        graph_scale_array[i] = 0;
    }
    graph_texture_pixels = malloc(4 * GRAPH_W * GRAPH_H);
    memset(graph_texture_pixels, 0, 4 * GRAPH_W * GRAPH_H);

    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);


    window_refresh();


    bool keep_going = true;
    bool inner_keep_going = true;
    bool update = true;
    bool reset_fr = false;
    bool update_anim = true;

    int num_areas = 2;
    char *area_names[] = { "Elephant Valley", "Tail End" };
    int area_num = -1, last_area_num = -1;

    char * notification_cmd = malloc(1000);

    bool r_down = false, l_down = false;
    bool s_down = false;
    double horiz_v = 0, vert_v = 0, zoom_v = 0, utweak_v = 0;

    int last_ticks = SDL_GetTicks();
    int last_update_ticks = SDL_GetTicks();
    int last_anim_switch_ticks = SDL_GetTicks();

    double anim_start_cX = fr.cX;
    double anim_start_cY = fr.cY;
    double anim_start_Z = fr.Z;
    double anim_start_u = fr.u, anim_start_v = fr.v;
    bool is_anim = false;
    while (keep_going == true) {
        //log_trace("outer loop");
        // set this to true to fore compute each time
        update = false;
        update_anim = false;
        reset_fr = false;
        inner_keep_going = true;
        if (USE_JOYSTICK) {
            update = horiz_v != 0 || vert_v != 0 || zoom_v != 0 || utweak_v != 0;
            if (update) {
                update_anim = true;
                double scale_allinput = (double)(SDL_GetTicks() - last_ticks) / 1000.0;
                fr.cX += 1.0 * scale_allinput * horiz_v / fr.Z;
                fr.cY -= 1.0 * scale_allinput * vert_v / fr.Z;
                fr.u += 1.0 * scale_allinput * utweak_v / fr.Z;
                double zfact = 1.0 + 1.0 * scale_allinput * abs(zoom_v);
                if (zoom_v > 0) {
                    fr.Z /= zfact;
                } else if (zoom_v < 0) {
                    fr.Z *= zfact;
                }
            }
            /*
            if (update) {
                log_trace("diff: %d,%d,%d", horiz_v, vert_v, zoom_v);
            }
            */
        }

        last_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&cevent)) {
            if (inner_keep_going) {
                switch (cevent.type) {
                    case SDL_JOYAXISMOTION:
                        log_trace("joystick axis");
                        if (cevent.jaxis.axis == horiz) {
                            horiz_v = SMASH(cevent.jaxis.value, 100) / AXIS_MAX;
                            //horiz_v = sgn(horiz_v) * pow(fabs(horiz_v), .333);
                        }
                        if (cevent.jaxis.axis == vert) {
                            vert_v = SMASH(cevent.jaxis.value, 100) / AXIS_MAX;
                            //vert_v = sgn(vert_v) * pow(fabs(vert_v), .333);
                        }
                        if (cevent.jaxis.axis == zaxis) {
                            zoom_v = SMASH(cevent.jaxis.value, 100) / AXIS_MAX;
                            //zoom_v = sgn(zoom_v) * pow(fabs(zoom_v), .333);
                        }
                        if (cevent.jaxis.axis == uaxis) {
                            utweak_v = SMASH(cevent.jaxis.value, 100) / AXIS_MAX;
                            //zoom_v = sgn(zoom_v) * pow(fabs(zoom_v), .333);
                        }
                        break;
                    case SDL_JOYDEVICEREMOVED:
                        log_info("joystick index removed: %d", cevent.jdevice.which);
                        joystick = NULL;
                        break;
                    case SDL_JOYDEVICEADDED:
                        log_info("joystick index added: %d", cevent.jdevice.which);
                        if (joystick == NULL) {
                            log_info("using joystick: %d", cevent.jdevice.which);
                        }
                        joystick = SDL_JoystickOpen(cevent.jdevice.which);
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
                        // quit
                        if (cevent.key.keysym.sym == 'q' && cevent.key.repeat == 0) {
                            MPI_Abort(MPI_COMM_WORLD, 0);
                        } else if (cevent.key.keysym.sym == 'e' && cevent.key.repeat == 0) {
                            // hard exit, non-zero status. Some scripts may choose to restart if non-zero
                            MPI_Abort(MPI_COMM_WORLD, 123);
                        } else if (cevent.key.keysym.sym == ' ' && cevent.key.repeat == 0) {
                            if (s_down) {
                                fr.Z /= 1.5;
                            } else {
                                fr.Z *= 1.5;
                            }
                            update = true; update_anim = true;
                        } else if (cevent.key.keysym.sym == SDLK_LSHIFT || cevent.key.keysym.sym == SDLK_RSHIFT && cevent.key.repeat == 0) {
                            s_down = true;
                            //fr.Z /= 1.5;
                        } else if (cevent.key.keysym.sym == SDLK_LEFT && cevent.key.repeat == 0) {
                            fr.cX -= .15 / fr.Z;
                            update = true; update_anim = true;
                        } else if (cevent.key.keysym.sym == SDLK_RIGHT && cevent.key.repeat == 0) {
                            fr.cX += .15 / fr.Z;
                            update = true;
                        } else if (cevent.key.keysym.sym == SDLK_UP && cevent.key.repeat == 0) {
                            fr.cY += .15 / fr.Z;
                            update = true; update_anim = true;
                        } else if (cevent.key.keysym.sym == SDLK_DOWN && cevent.key.repeat == 0) {
                            fr.cY -= .15 / fr.Z;
                            update = true; update_anim = true;
                        } else if (cevent.key.keysym.sym == SDLK_ESCAPE && cevent.key.repeat == 0) {
                            keep_going = false;
                            inner_keep_going = false;
                            log_info("escaping program");
                            MPI_Abort(MPI_COMM_WORLD, 0);
                        } else if (cevent.key.keysym.sym == 'p') {
                            fr.max_iter += 1;
                            update = true;// update_anim = true;
                        } else if (cevent.key.keysym.sym == 'z' && cevent.key.repeat == 0) {
                            // toggle simple coloring
                            fr.fractal_flags ^= FRF_SIMPLE;
                            update = true;
                        } else if (cevent.key.keysym.sym == 'x' && cevent.key.repeat == 0) {
                            // toggle re() binary decomposition
                            fr.fractal_flags ^= FRF_BINARYDECOMP_REAL;
                            update = true;
                        } else if (cevent.key.keysym.sym == 'c' && cevent.key.repeat == 0) {
                            // toggle im() binary decomposition
                            fr.fractal_flags ^= FRF_BINARYDECOMP_IMAG;
                            update = true;
                        } else if (cevent.key.keysym.sym == 'o') {
                            if (fr.max_iter > 0) {
                                fr.max_iter -= 1;
                                update = true;
                            }
                        } else if (cevent.key.keysym.sym == 'a' && cevent.key.repeat == 0) {
                            if (fr.engine == FR_E_C) {
			        fr.engine = FR_E_CUDA;
                            } else if (fr.engine == FR_E_CUDA) {
                                fr.engine = FR_E_C;
                            }
                            update = true;
                        } else if (cevent.key.keysym.sym == 's' && cevent.key.repeat == 0) {
                            if (fr.engine == FR_E_C) {
			                           fr.engine = FR_E_CUDA;
                            } else if (fr.engine == FR_E_CUDA) {
                                fr.engine = FR_E_C;
                            }
                            update = true;
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
                        } else if (cevent.key.keysym.sym == 't' && cevent.key.repeat == 0) {
	                          show_extra = !show_extra;
                            update = true;
                        } else if (cevent.key.keysym.sym == 'm' && cevent.key.repeat == 0) {
                            fractal_types_idx = (fractal_types_idx + 1) % FR_FRACTAL_NUM;
                            fr.fractal_type = fractal_types[fractal_types_idx];
                            update = true; update_anim = true;
                            reset_fr = true;
                        } else if (cevent.key.keysym.sym == 'n' && cevent.key.repeat == 0) {
                            fractal_types_idx = (fractal_types_idx - 1 + FR_FRACTAL_NUM) % FR_FRACTAL_NUM;
                            fr.fractal_type = fractal_types[fractal_types_idx];
                            update = true; update_anim = true;
                            reset_fr = true;
                        }
                        break;
                    case SDL_MOUSEMOTION:
                        if (l_down) {
                            fr.cX -= cevent.motion.xrel / (fr.Z * fr.w);
                            fr.cY += cevent.motion.yrel / (fr.Z * fr.w);
                            update = true; update_anim = true;
                        }
                        if (r_down) {
                            double zoomin = 1 + fabs(2.0 * (cevent.motion.yrel + cevent.motion.xrel) / (fr.w + fr.h));
                            if (!s_down) {
                                fr.Z *= zoomin;
                            } else {
                                fr.Z /= zoomin;
                            }
                            update = true; update_anim = true;
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
            //SDL_PumpEvents();
            //inner_do = false;
        }
        if (!keep_going) {
            break;
        }
        if (update_anim) {
            is_anim = false;
        }
        if (update || reset_fr) {
            if (reset_fr) {
                log_trace("resetting fractal");
                fr.cX = 0; fr.cY = 0;
                fr.u = 1; fr.v = 0;
                fr.Z = .4;
            }
            //is_anim = false;
            log_trace("recomputing fractal");
            //window_refresh();
            if (num_animations == 0 || !is_anim) {
                last_update_ticks = SDL_GetTicks();
                is_anim = false;
                window_refresh();
            }

            //_fr_interactive_sdl_recompute(fr, fr_engine);
        }
        if (num_animations > 0 && (SDL_GetTicks() - last_update_ticks) / 1000 >= timeout + 1.0 / last_fps) {
            if (!is_anim) {
              log_debug("starting auto-animation");
              curr_anim = 0;
              last_anim_switch_ticks = SDL_GetTicks();
              fractal_types_idx = animations[0].fractal_id;
              fr.fractal_type = fractal_types[0];
              anim_start_cX = fr.cX;
              anim_start_cY = fr.cY;
              anim_start_u = fr.u;
              anim_start_v = fr.v;
              anim_start_Z = fr.Z;
            }
            is_anim = true;
            if (SDL_GetTicks() - last_anim_switch_ticks > animations[curr_anim].ticks) {
                fractal_types_idx = animations[curr_anim].fractal_id;
                fr.fractal_type = fractal_types[fractal_types_idx];
                anim_start_cX = animations[curr_anim].cX;
                anim_start_cY = animations[curr_anim].cY;
                anim_start_u = animations[curr_anim].u;
                anim_start_v = animations[curr_anim].v;
                anim_start_Z = animations[curr_anim].Z;
                curr_anim = (curr_anim + 1) % num_animations;
                last_anim_switch_ticks = SDL_GetTicks();
                /*
                anim_start_cX = fr.cX;
                anim_start_cY = fr.cY;
                anim_start_Z = fr.Z;
                */
            }

            int t0 = last_anim_switch_ticks;
            int t1 = t0 + animations[curr_anim].ticks;
            int t = SDL_GetTicks();

            double p = (double)(t - t0) / (t1 - t0);
            //p += ((tan((t - t0) / 1000.0) + 1) / 2) / 5;
            /*p = 2 * p - 1;
            p = p * p * p + p;
            p = (p + 2) / 4;*/
            //p = 4 * (2 * p - 1);
            //p = erf(p / 2.0 + 1);
            //p = (p - erf(1)) / (erf(1.5) - erf(1));
            if (p <= 1.0) {

              fr.cX = (1 - p) * (anim_start_cX) + (p * animations[curr_anim].cX);
              fr.cY = (1 - p) * (anim_start_cY) + (p * animations[curr_anim].cY);
              fr.u = (1 - p) * (anim_start_u) + (p * animations[curr_anim].u);
              fr.v = (1 - p) * (anim_start_v) + (p * animations[curr_anim].v);
              fr.Z = exp(log(anim_start_Z) * (1 - p) + p * log(animations[curr_anim].Z));
            }
            window_refresh();
        }
#define PT_FITS(x, xr, y, yr, minz, maxz) (fabs(fr.cX - (x)) <= (xr) && fabs(fr.cY - (y)) <= (yr) && (fr.Z >= (minz) && fr.Z <= (maxz)))
        if (PT_FITS(.35, .1, 0, .4, 5, 10000000)) {
            area_num = 0;
        } else if (PT_FITS(-1.7, .3, 0, .2, 10, 100000000000)) {
            area_num = 1;
        } else if (PT_FITS(-0.75, .3, 0, .2, 10, 100000000000)) {
            area_num = 1;
        } else {
            area_num = -1;
        }
        if (fractal_types_idx == 0 && area_num != last_area_num) {
            if (area_num >= 0 && area_num < num_areas) {
                sprintf(notification_cmd, "notify-send -t 3000 'fractalexplorer' 'Entering area: %s'", area_names[area_num]);
                log_info("running cmd: '%s', code: %d", notification_cmd, system(notification_cmd));
            } else if (last_area_num >= 0 && last_area_num < num_areas) {
                sprintf(notification_cmd, "notify-send -t 3000 'fractalexplorer' 'Leaving area: %s'", area_names[last_area_num]);
                log_info("running cmd: '%s', code: %d", notification_cmd, system(notification_cmd));
            }
        }
        last_area_num = area_num;
        //do_update = false;
    }

    log_info("quitting now");

    MPI_Abort(MPI_COMM_WORLD, 0);
}
