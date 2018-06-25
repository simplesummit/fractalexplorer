// CUDA interaction


#include <cuComplex.h>

// since CUDA is c++, we wrap our code here
extern "C" {


#include <stdio.h>
#include <stdlib.h>


#include "fr.h"
#include "log.h"
//#include "fractalexplorer.h"


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

    // seperate for each column
    int ** output_iters_col_seperate;

} GPU_memory;


void engine_cuda_init(fractal_params_t _fr_p) {
    fractal_params = _fr_p;
    gpuErrchk(cudaMalloc((void **)&GPU_memory.assigned_cols, sizeof(int) * fractal_params.width));
    gpuErrchk(cudaMalloc((void **)&GPU_memory.output, 3 * fractal_params.width * fractal_params.height));
}



// internal GPU kernel
__global__ void _engine_cuda_kernel(int total_width, int total_height, int num_assigned_cols, int * assigned_cols, unsigned char * output) {
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;
    //sometimes we assign more for efficiency reasons
    if (col_index >= num_assigned_cols) {
        return;
    }
    int px = assigned_cols[col_index], py = blockIdx.y * blockDim.y + threadIdx.y;

    RGB_t color;

    color.R = 0;
    color.G = 0;
    color.B = (255 * py / 480) & 0xff;

    ((RGB_t *)output)[total_height * col_index + py] = color;

}


// returns 3byte packed pixels of successive rows, and stores the number of iterations for each column in output_iters
void engine_cuda_compute(workload_t workload, unsigned char * output, int * output_iters) {

    dim3 d_block(16, 16);
    dim3 d_grid;
    d_grid.x = (fractal_params.width + d_block.x - 1) / d_block.x;
    d_grid.y = (fractal_params.height + d_block.y - 1) / d_block.y;


    gpuErrchk(cudaMemcpy(GPU_memory.assigned_cols, workload.assigned_cols, sizeof(int) * workload.assigned_cols_len, cudaMemcpyHostToDevice));

    _engine_cuda_kernel<<<d_grid, d_block>>>(fractal_params.width, fractal_params.height, workload.assigned_cols_len, GPU_memory.assigned_cols, GPU_memory.output);

    int res_size = 3 * fractal_params.height * workload.assigned_cols_len;

    gpuErrchk(cudaMemcpy(output, GPU_memory.output, res_size, cudaMemcpyDeviceToHost));

    int i;
    for (i = 0; i < workload.assigned_cols_len; ++i) {
        output_iters[i] = 2000;
    }

    //done boi

}


}



