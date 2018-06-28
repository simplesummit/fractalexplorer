
// CUDA interaction


#include <cuComplex.h>

// since CUDA is c++, we wrap our code here
extern "C" {


#include <stdio.h>
#include <stdlib.h>




#include "fr.h"
#include "log.h"


fractal_params_t fractal_params;

// macros to help
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
       log_error("GPUassert (at %s:%d) (code %d): %s\n", file, line, code,
                       cudaGetErrorString(code));
 
       // TODO: determine when to exit. Some kernel launch failures seem to be
       // recoverable
       // codes: 35 is insufficient driver

       //if (code != 35) {
       //    exit(code);
       //}
    }
 }




//#include "engine_c.h"
//#include "fractalexplorer.h"

//#include "math.h"
//#include "complex.h"


struct {
    // color scheme in memory buffer
    int color_scheme_len;
    unsigned char * color_scheme;

    // which columns to do
    int assigned_cols_len;
    int * assigned_cols;

    // packed array
    unsigned char * output;

    // seperate for each column and row
    int * output_iters_col_seperate;

} GPU_memory;


struct {

    int * output_iters_col_seperate;

} CPU_memory;


void engine_cuda_init(fractal_params_t _fr_p, int color_scheme_len, unsigned char * color_scheme) {
    fractal_params = _fr_p;
    gpuErrchk(cudaMalloc((void **)&GPU_memory.assigned_cols, sizeof(int) * fractal_params.width));
    gpuErrchk(cudaMalloc((void **)&GPU_memory.output, 3 * fractal_params.width * fractal_params.height));

    GPU_memory.color_scheme_len = color_scheme_len;
    gpuErrchk(cudaMalloc((void **)&GPU_memory.color_scheme, 3 * color_scheme_len));

    gpuErrchk(cudaMemcpy(GPU_memory.color_scheme, color_scheme, 3 * color_scheme_len, cudaMemcpyHostToDevice));
    CPU_memory.output_iters_col_seperate = (int *)malloc(sizeof(int) * fractal_params.width * fractal_params.height);


    gpuErrchk(cudaMalloc((void **)&GPU_memory.output_iters_col_seperate, sizeof(int) * fractal_params.width * fractal_params.height));



}



// internal GPU kernel
__global__ void _engine_cuda_kernel(fractal_params_t frp, int color_scheme_len, unsigned char * color_scheme, int num_assigned_cols, int * assigned_cols, unsigned char * output, int * result_iters) {
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;
    //sometimes we assign more for efficiency reasons
    if (col_index >= num_assigned_cols) {
        return;
    }
    int px = assigned_cols[col_index], py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= frp.width || py >= frp.height) {
        return; //printf("ILLEGAL INDEX\n");
    }


    double c_r = X_PIXEL_TO_RE(px, frp.width, frp.height, frp.center_r, frp.zoom);
    double c_i = Y_PIXEL_TO_IM(py, frp.width, frp.height, frp.center_i, frp.zoom);

    double z_r = c_r, z_i = c_i;

    double z_r2, z_i2;

    // stuff for colorization
    int iter = 0;
    double partial_iteration;

    if (frp.type == FRACTAL_TYPE_MANDELBROT) {

        z_r2 = z_r * z_r;
        z_i2 = z_i * z_i;

        for (iter = 0; iter < frp.max_iter && z_r2 + z_i2 <= 256.0; ++iter) {
            z_i = 2 * z_r * z_i + c_i;
            z_r = z_r2 - z_i2 + c_r;
            z_r2 = z_r * z_r;
            z_i2 = z_i * z_i;
        }
        partial_iteration = 3 + iter - log(log(z_r2 + z_i2)) / log(2.0);
    }

    double idx_final = partial_iteration;

    int color_idx = (int)floor(partial_iteration);
    double gradient = idx_final - color_idx;
    
    int before_idx = (color_idx % color_scheme_len + color_scheme_len) % color_scheme_len;
    int after_idx = (before_idx + 1) % color_scheme_len;

    RGB_t color_before = ((RGB_t *)color_scheme)[before_idx];
    RGB_t color_after = ((RGB_t *)color_scheme)[after_idx];

    
    RGB_t color;

    color.R = lin_mix(color_before.R, color_after.R, gradient);
    color.G = lin_mix(color_before.G, color_after.G, gradient);
    color.B = lin_mix(color_before.B, color_after.B, gradient);

    ((RGB_t *)output)[frp.height * col_index + py] = color;
    result_iters[frp.height * col_index + py] = iter;

}


// returns 3byte packed pixels of successive rows, and stores the number of iterations for each column in output_iters
void engine_cuda_compute(workload_t workload, unsigned char * output, int * output_iters) {

    dim3 d_block(16, 12);
    dim3 d_grid;
    d_grid.x = (workload.assigned_cols_len + d_block.x - 1) / d_block.x;
    d_grid.y = (fractal_params.height + d_block.y - 1) / d_block.y;



    int i, j;
    for (i = 0; i < workload.assigned_cols_len; ++i) {
        output_iters[i] = 0;
    }


    gpuErrchk(cudaMemcpy(GPU_memory.assigned_cols, workload.assigned_cols, sizeof(int) * workload.assigned_cols_len, cudaMemcpyHostToDevice));

    _engine_cuda_kernel<<<d_grid, d_block>>>(fractal_params, GPU_memory.color_scheme_len, GPU_memory.color_scheme, workload.assigned_cols_len, GPU_memory.assigned_cols, GPU_memory.output, GPU_memory.output_iters_col_seperate);

    int res_size = 3 * fractal_params.height * workload.assigned_cols_len;

    gpuErrchk(cudaMemcpy(output, GPU_memory.output, res_size, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(CPU_memory.output_iters_col_seperate, GPU_memory.output_iters_col_seperate, sizeof(int) * workload.assigned_cols_len * fractal_params.height, cudaMemcpyDeviceToHost));



    for (i = 0; i < workload.assigned_cols_len; ++i) {
        for (j = 0; j < fractal_params.height; ++j) {
            output_iters[i] += CPU_memory.output_iters_col_seperate[fractal_params.height * i + j];
        }
    }

    //done boi

}


}



