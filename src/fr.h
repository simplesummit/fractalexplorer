
#ifndef __FR_H__
#define __FR_H__

#include <mpi.h>
#include <time.h>
#include <sys/time.h>


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

//#define X_PIXEL_TO_RE(px, w, h, c, z) ((c) + ((2.0 * (px)) / (h) - 1.0) / (z))

#define X_PIXEL_TO_RE(px, w, h, c, z) ((c) + (2.0 * (px) - (w)) / ((w) * (z)))

#define Y_PIXEL_TO_IM(py, w, h, c, z) ((c) + ((h) - 2.0 * (py)) / (w * z))



// z^2 + c
#define FRACTAL_TYPE_MANDELBROT 0x0101



#define FRACTAL_FLAG_NONE 0x0000
#define FRACTAL_FLAG_GRADIENT 0x0001
#define FRACTAL_FLAG_SPLIT_REAL 0x0002
#define FRACTAL_FLAG_SPLIT_IMAG 0x0004
#define FRACTAL_FLAG_ADD_PERIOD 0x0008
#define FRACTAL_FLAG_USE_COMPRESSION 0x0010



// time performance
typedef struct tperf_t {

    //struct timeval stime, etime;

    double stime, etime;

    double elapsed_s;

} tperf_t;
#define tperf_init(tp) { tperf_start(tp); tperf_end(tp); }
#define tperf_loop(tp) { tperf_end(tp); tperf_start(tp); }
#define tperf_start(tp) (tp).stime = MPI_Wtime();
#define tperf_end(tp) { (tp).etime = MPI_Wtime(); (tp).elapsed_s = (tp).etime - (tp).stime; }


// how many diagnostics frames to save
#define NUM_DIAGNOSTICS_SAVE 12

// a copy of this is kept on all nodes, and is updated every loop
typedef struct fractal_params_t {
    // parameters to the fractal

    // width and height of the master image
    int width, height;

    // see FRACTAL_TYPE_* macros above
    int type;

    // colorization flags, stuff like that
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


typedef struct workload_t {
    /*

    this is the actual workload 

    */

   int assigned_cols_len;

   int * assigned_cols;


} workload_t;

#define NODE_TYPE_CPU 0x0001
#define NODE_TYPE_GPU 0x0002
#define NODE_TYPE_MASTER 0x0003

typedef struct node_t {

    int type;

    char * processor_name;

} node_t;


typedef struct node_diagnostics_t {

    /* these are reported*/
    // fahrenheit
    float temperature;

    float time_compute, time_compress, time_total;

    int total_cols;

} node_diagnostics_t;

typedef struct diagnostics_t {

    /* these are from the master node */

    float time_control_update, time_assign, time_wait, time_decompress, time_recombo, time_visuals, time_total;

    // array for each node
    node_diagnostics_t * node_information;

    
    // array of which nodes were assigned to the (index)th column
    int * node_assignments;

    // array of how many iterations there are for each column
    int * col_iters;

    // timestamp?
    struct timeval timestamp;

} diagnostics_t;


#endif

