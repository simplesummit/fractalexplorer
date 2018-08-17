/* fractalexplorer.h -- main header file, defines types, includes libraries, etc



*/

#ifndef __FRACTALEXPLORER_H__
#define __FRACTALEXPLORER_H__

// configuration file generated from cmake
#include "fractalexplorerconfig.h"

// standard header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <mpi.h>


// include OpenCL header files
#ifdef HAVE_OpenCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

// SDL library
#include <SDL2/SDL.h>

// other header files within this project
#include "log.h"

#define FRACTAL_MODEL_MANDELBROT (1 << 0)

#define FRACTAL_FLAG_NONE (0)

typedef struct _fractal_t {

    // width and height in pixels
    int16_t width, height;

    // maximum number of iterations to carry out
    uint16_t max_iter;

    // center complex coordinates
    double center_r, center_i;

    // what model, see FRACTAL_MODEL_* macros
    int8_t model;

    // flags, see FRACTAL_FLAG_* macros
    int32_t flags;

    // 'q' coefficient can be used in some models
    double q_r, q_i;

    double zoom;

} fractal_t;



// gets time in seconds
double get_time();



// packed color, for output over a network
typedef struct _packed_color_t {

    // integer index into the palette
    uint8_t index;

    // this is a fixed point value, indicating scaling between palette[index] and palette[index+1]
    // when prop==0, the result color is palette[index]
    // when prop==X, the result color is palette[index] * X/256 + palette[index + 1] * (256 - X) / 256
    uint8_t prop;

} packed_color_t;


// highquality color
typedef struct _hq_color_t {

    uint8_t R, G, B;

} hq_color_t;

typedef struct _palette_t {

    int num_colors;
    // allocated to store 'num_colors' instances
    hq_color_t * colors;

} palette_t;

/*
    color library
*/

hq_color_t hq_color_rgb(uint8_t R, uint8_t G, uint8_t B);

hq_color_t hq_color_packed(palette_t palette, packed_color_t packed);

hq_color_t hq_color_mix(hq_color_t a, hq_color_t b, double proportion);


/*
  visuals.c library
*/

// initializes windows, buffers, textures, etc
void visuals_init(int win_width, int win_height);

bool visuals_update(hq_color_t * fractal_pixels);

void visuals_finish();

// workload definition
typedef struct _workload_t {

    // this means that the workload consists of start, start+1, ...start+len-1
    // ex: start=0, length=5
    // this workload would be rows 0,1,2,3,4
    int16_t start, length;

} workload_t;


/*
    compute libraries
*/

void compute_C(workload_t workload, packed_color_t * output);

// how many diagnostics to store?
#define NUM_DIAGNOSTICS 200

typedef struct _diagnostics_t {

    // all timing
    double total_time;

    // specifics
    double compute_time, io_time, format_time, display_time;

    // sum of all workload transfer sizes
    // only what is sent over the network is included here, so if compression is used, this will change
    // each frame based on how compressable the results are
    int bytes_transferred;

} diagnostics_t;

// history of the last `NUM_DIAGNOSTICS`
diagnostics_t * diagnostics;


#define DIAGNOSTICS_IDX ((n_frames) % NUM_DIAGNOSTICS)
// used to get previous diagnostics stuff
#define PREVIOUS_DIAGNOSTICS(i) (diagnostics[((((n_frames - (i)) % NUM_DIAGNOSTICS) + NUM_DIAGNOSTICS) % NUM_DIAGNOSTICS)])
// how many frames have been computed? Starts at 0, never loops around
int n_frames;

int world_size, world_rank;

// main fractal object
fractal_t fractal;
// global palette object
palette_t palette;

// main method
int main(int argc, char ** argv);


#endif


