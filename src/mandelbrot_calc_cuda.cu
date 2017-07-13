/* mandelbrot_calc_cuda.cu -- CUDA engine, which is only included if the
                           -- platform supports it

  This file is part of the small-summit-fractal project.

  small-summit-fractal source code, as well as any other resources in this
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

extern "C" {

#include <stdio.h>
#include <stdlib.h>


typedef
double
complex[2];

/* complex function macros */

// sets r to a
// r can be a
#define cset(r, a) r[0] = a[0]; r[1] = a[1];

// returns a + b
// r can be a or b
#define cadd(r, a, b) r[0] = a[0] + b[0]; r[1] = a[1] + b[1];

// returns a - b
// r can be a or b
#define csub(r, a, b) r[0] = a[0] - b[0]; r[1] = a[1] - b[1];

// returns a * b
// r has to be different variable than a and b
#define cmul(r, a, b) r[0] = a[0] * b[0] - a[1] * b[1]; r[1] = a[0] * b[1] + a[1] * b[0];

// returns a * a
// r has to be a different variable than a
#define csqr(r, a) r[0] = a[0] * a[0] - a[1] * a[1]; r[1] = 2 * a[0] * a[1];


// the absolute value of x, squared
#define cabs2(x) (x[0] * x[0] + x[1] * x[1])
#define cabs(x) sqrt(cabs2(x))

#include "fr.h"

// a macro to check a result and then print out info if failed, and (possibly)
// exit
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert (at %s:%d) (code %d): %s\n", file, line, code,
                      cudaGetErrorString(code));

      // TODO: determine when to exit. Some kernel launch failures seem to be
      // recoverable
      // codes: 35 is insufficient driver
      if (code != 35) {
          exit(code);
      }
   }
}

// last width and height
int lw = 0, lh = 0;

// number of colors
int colnum;

// GPU buffers
unsigned char * _gpu_output = NULL, * _gpu_col = NULL;


// a CUDA device kernel to compute fractal pixels value. Takes fr as parameters,
// my_h and my_off for custom rank based parameters, color and the number of
// colors, and an output buffer. Note that the buffers should be allocated
// with CUDA device memory functions
__global__
void mand_cuda_kernel(fr_t fr, int my_h, int my_off, unsigned char * col, int ncol, unsigned char * output) {

    // compute the current pixel offset
    int px = (blockIdx.x * blockDim.x) + threadIdx.x;
    int py = (blockIdx.y * blockDim.y) + threadIdx.y;

    // it may be desirable to launch a job with more than neccessary dimensions
    // so, we simply don't run when this is true
    if (px >= fr.w || py >= my_h || my_off + py >= fr.h) {
        return;
    }

    // the return index, current iteration, and color indexes
    int ri = py * fr.mem_w + 4 * px, ci, c0, c1;

    // the offset is added (as the px and py are 0 based)
    py += my_off;

    double fri, mfact, _q;

    // real, imaginary
    complex c = {
        fr.cX - (fr.w - 2 * px) / (fr.Z * fr.w),
        fr.cY + (fr.h - 2 * py) / (fr.Z * fr.w)
    };

    complex z;
    cset(z, c);
    complex tmp;

    _q = (z[0] - .25f);
    _q = _q * _q + z[1] * z[1];
    if (_q * (_q + (z[0] - .25f)) < z[1] * z[1] / 4.0f) {
        ci = fr.max_iter;
    } else {
        for (ci = 0; ci < fr.max_iter && cabs(c) <= 16.0; ++ci) {
            csqr(tmp, z);
            cadd(z, tmp, c);
        }
    }


    if (ci == fr.max_iter) {
        fri = 0.0;
    } else {
        fri = 2.0 + ci - log(log(cabs2(z))) / log(2.0);
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
