
#ifndef __FR_H__
#define __FR_H__

#include <mpi.h>


/*
typedef struct image_t {
    // bitmap image
    
    // width, height 

    // 3 bytes per pixel
    
    // row major order, so the [x, y] pixel ([0, 0] is top left)
    // is pixel_data[3 * (width * y + x)]
    // this should be allocated to 3 * width * height bytes
    char * pixel_data;

}
*/



// z^2 + c
#define FRACTAL_TYPE_MANDELBROT 0x0101



#define FRACTAL_FLAG_GRADIENT 0x0001
#define FRACTAL_FLAG_SPLIT_REAL 0x0002
#define FRACTAL_FLAG_SPLIT_IMAG 0x0004
#define FRACTAL_FLAG_ADD_PERIOD 0x0008



// time performance
typedef struct tperf_t {
    struct timeval stime, etime;

    double elapsed_s;
} tperf_t;

#define tperf_start(tp) gettimeofday((tp).stime, NULL);
#define tperf_end(tp) gettimeofday((tp).etime, NULL); (tp).elapsed_s = ((tp).etime.tv_sec - (tp).stime.tv_sec) + ((tp).etime.tv_usec - (tp).stime.tv_usec) / 1000000.0;


// how many diagnostics frames to save
#define NUM_DIAGNOSTICS_SAVE 100

// a copy of this is kept on all nodes, and is updated every loop
typedef struct fractal_params_t {
    // parameters to the fractal

    // width and height of the master image
    int width, height;

    // see FRACTAL_TYPE_* macros above
    int type;

    // colorization flags
    int flags;

    // maximum iterations
    int max_iter;

    // used in some formulas as a variable in the formula (q = q_r + i * q_i)
    double q_r, q_i;

    // center of the image
    double center_r, center_i;
    
    // how zoomed in?
    double zoom;

    /*

     -----------------
     |                |
     |                |
     |________________|
            

     height = 1 / zoom


    */
} fractal_params_t;

#define NODE_TYPE_CPU 0x0001
#define NODE_TYPE_GPU 0x0002
#define NODE_TYPE_MASTER 0x0003

typedef struct node_t {

    int type;

    char * processor_name;

} node_t;

typedef struct node_diagnostics_report_t {
    // separate data structure for MPI passing

    // fahrenheit
    float temperature;

    float time_compute, time_compress;

} node_diagnostics_report_t;

typedef struct node_diagnostics_t {

    // this came from what the node reported
    node_diagnostics_report_t reported;

    float time_decompress, time_transfer, time_total;

    int total_rows;

} node_diagnostics_t;

typedef struct diagnostics_t {

    // array for each node
    node_diagnostics_t * node_information;

    // timestamp?
    struct timeval timestamp;

} diagnostics_t;


#endif

