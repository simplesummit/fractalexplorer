

extern "C" {

#include <stdio.h>
#include <stdlib.h>

#include "fr.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

int lw = 0, lh = 0;

int colnum;

unsigned char * _gpu_output = NULL, * _gpu_col = NULL;

__global__
void mand_cuda_kernel(fr_t fr, int my_h, int my_off, unsigned char * col, int ncol, unsigned char * output) {

    int px = (blockIdx.x * blockDim.x) + threadIdx.x;
    int py = (blockIdx.y * blockDim.y) + threadIdx.y;

    // these are added as buffers
    if (px >= fr.w || py >= my_h || my_off + py >= fr.h) {
        return;
    }
    int ri = 4 * (py * fr.w + px), ci, c0, c1;

    py += my_off;

    double fri, mfact, _q;

    double c_r = fr.cX - (fr.w - 2 * px) / (fr.Z * fr.w), c_i = fr.cY + (fr.h - 2 * py) / (fr.Z * fr.w);

    double z_r = c_r, z_i = c_i;

    double z_r2 = z_r * z_r, z_i2 = z_i * z_i;

    _q = (z_r - .25f);
    _q = _q * _q + z_i2;
    if (_q * (_q + (z_r - .25f)) < z_i2 / 4.0f) {
        ci = fr.max_iter;
    } else {
        ci = 0;
    }

    for (; ci < fr.max_iter && z_r2 + z_i2 <= 16.0; ++ci) {
        z_i = 2 * z_r * z_i + c_i;
        z_r = z_r2 - z_i2 + c_r;
        z_r2 = z_r * z_r; z_i2 = z_i * z_i;
    }

    if (ci == fr.max_iter) {
        fri = 0.0;
    } else {
        fri = 2.0 + ci - log(log(z_r2 + z_i2)) / log(2.0);
    }

    mfact = fri - floor(fri);
    //
    mfact = 0;

    c0 = (int)floor(fri) % ncol;
    c1 = (c0 + 1) % ncol;

    c0 *= 4; c1 *= 4;

    #define MIX(a, b, F) ((b) * (F) + (a) * (1 - (F)))


    output[ri + 0] = (int)floor(MIX(col[c0 + 0], col[c1 + 0], mfact));
    output[ri + 1] = (int)floor(MIX(col[c0 + 1], col[c1 + 1], mfact));
    output[ri + 2] = (int)floor(MIX(col[c0 + 2], col[c1 + 2], mfact));
    output[ri + 3] = (int)floor(MIX(col[c0 + 3], col[c1 + 3], mfact));

}

void mand_cuda_init(fr_col_t col) {
    colnum = col.num;
    gpuErrchk(cudaMalloc((void **)&_gpu_col, 4 * colnum));
    gpuErrchk(cudaMemcpy(_gpu_col, col.col, 4 * colnum, cudaMemcpyHostToDevice));
}

void mand_cuda(fr_t fr, int my_h, int my_off, unsigned char * output) {
    dim3 dimBlock(4, 4);
    dim3 dimGrid(fr.w / 4, my_h / 4);


    if (lw != fr.mem_w || lh != my_h) {
        if (_gpu_output != NULL) {
            cudaFree(_gpu_output);
        }
        gpuErrchk(cudaMalloc((void **)&_gpu_output, fr.mem_w * my_h));
    }


    // we dont need cudaMalloc(), because of ZEROcopy buffers that the GPU and CPU can share sys memory
    mand_cuda_kernel<<<dimGrid, dimBlock>>>(fr, my_h, my_off, _gpu_col, colnum, _gpu_output);


    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());


    gpuErrchk(cudaMemcpy(output, _gpu_output, fr.mem_w * my_h, cudaMemcpyDeviceToHost));

    lw = fr.mem_w;
    lh = my_h;

}


}
