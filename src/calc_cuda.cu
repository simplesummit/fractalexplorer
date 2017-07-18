/* calc_cuda.cu -- CUDA engine, which is only included if the platform supports
                -- it

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

#include <cuComplex.h>


extern "C" {

#include <stdio.h>
#include <stdlib.h>

typedef cuDoubleComplex complex;


#include "fr.h"

#define creal(a) a.x
#define cimag(a) a.y


/*

our complex number library, in cuda

*/


#define cabs(a) hypot(a.x*a.x, a.y*a.y)

// pow, returns e**(x)
__host__ __device__ static __inline__
cuDoubleComplex cuCexp(cuDoubleComplex x) {
    cuDoubleComplex result = { 0.0, 0.0 };
    double tmp_scale = exp(x.x);
    sincos(x.y, &result.y, &result.x);
    result.x *= tmp_scale;
    result.y *= tmp_scale;
    return result;
}

// sin, returns sin(x)
__host__ __device__ static __inline__
cuDoubleComplex cuCsin(cuDoubleComplex x) {
    cuDoubleComplex result = { 0.0, 0.0 };
    double tmp_scale = exp(x.x);
    sincos(x.y, &result.y, &result.x);
    result.x *= tmp_scale;
    result.y *= tmp_scale;
    return result;
}


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
int * _gpu_err = NULL;
bool cuda_has_init = false;


// a CUDA device kernel to compute fractal pixels value. Takes fr as parameters,
// my_h and my_off for custom rank based parameters, color and the number of
// colors, and an output buffer. err will be set to non-zero if an error occured
// Note that the buffers should be allocated with CUDA device memory functions
__global__
void cuda_kernel(fr_t fr, int my_h, int my_off, unsigned char * col, int ncol, unsigned char * output, int * err) {

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

    // fractional index
    double fri, mfact, _q, tmp;

    // real, imaginary
    complex z, c;

    c = make_cuDoubleComplex(
        fr.cX - (fr.w - 2 * px) / (fr.Z * fr.w),
        fr.cY + (fr.h - 2 * py) / (fr.Z * fr.w)
    );

    z = c;

    switch (fr.fractal_type) {
        case FR_MANDELBROT:
            // see above in this file, this method determines whether
            // we should skip the computation
            _q = (z.x - .25f);
            _q = _q * _q + z.y * z.y;
            if (_q * (_q + (z.x - .25f)) < z.y * z.y / 4.0f) {
                ci = fr.max_iter;
                fri = ci + 0.0;
            } else {
                for (ci = 0; ci < fr.max_iter && cabs(z) < 16.0; ++ci) {
                    z = cuCmul(z, z);
                    z = cuCadd(z, c);
                }
                tmp = log(log(creal(z)*creal(z) + cimag(z)*cimag(z)));
                if (fr.fractal_flags & FRF_TANKTREADS) {
                    fri = 2.0 + ci - tmp / log(2.5);
                } else {
                    fri = 2.0 + ci - tmp / log(2.0);
                }
            }
            break;
        case FR_MANDELBROT_3:
            // similar to the default mandelbrot, we will loop and use
            // an escape value of 16.0. However, we have no speedups,
            // like bulb_check_0 for this, so we will just iterate
            // the function
            for (ci = 0; ci < fr.max_iter && cabs(z) <= 16.0; ++ci) {
                z = cuCmul(z, cuCmul(z, z));
                z = cuCadd(z, c);
            }


            // we use the same basic method, but with a different
            // divisor. The divisor to get even boundaries between
            // iteration bounds is log(3.0) (log of the exponent),
            // however try 2.5 or 3.5 or 5 to get other, cool
            // fractional iteration counts
            tmp = log(log(creal(z)*creal(z) + cimag(z)*cimag(z)))
                  / log(3.0);
            fri = 2.0 + ci - tmp;
            break;
        case FR_EXP:
            //
            for (ci = 0; ci < fr.max_iter && abs(creal(z)) < 16.0; ++ci) {
                z = cuCexp(z);
                z = cuCadd(z, c);
            }
            // no current way to easily do a fractional iteration, so
            // just send a fractional iteration of the actual integer
            // value
            fri = 0.0 + ci;
            break;
        case FR_SIN:
            // the sin(z)+c may not just escape from a radius, and we
            // should check that the imaginary portion escapes
            for (ci = 0; ci < fr.max_iter && abs(cimag(z)) < 16.0; ++ci) {
                z = cuCsin(z);
                z = cuCadd(z, c);

            }
            // no current way to easily do a fractional iteration, so
            // just send a fractional iteration of the actual integer
            // value
            fri = 0.0 + ci;
            break;
        default:
            // this should never happen
            *err = 1;
            return;
            break;
    }

    // if ci is set to default values, if they have set ci to 0 or max
    // the computation might iter,
    // set the fri to corresponding values
    if (ci == fr.max_iter) {
        fri = 0.0 + fr.max_iter;
    }

    fri = fri * fr.cscale + fr.coffset;

    if (fr.fractal_flags & FRF_SIMPLE) {
        mfact = 0;
    } else {
        mfact = fri - floor(fri);
    }

    c0 = (int)floor(fri) % ncol;
    c1 = (c0 + 1) % ncol;

    c0 *= 4; c1 *= 4;

    #define MIX(a, b, F) ((b) * (F) + (a) * (1 - (F)))


    output[ri + 0] = (int)floor(MIX(col[c0 + 0], col[c1 + 0], mfact));
    output[ri + 1] = (int)floor(MIX(col[c0 + 1], col[c1 + 1], mfact));
    output[ri + 2] = (int)floor(MIX(col[c0 + 2], col[c1 + 2], mfact));
    output[ri + 3] = (int)floor(MIX(col[c0 + 3], col[c1 + 3], mfact));

}

void calc_cuda_init(fr_col_t col) {
    if (!cuda_has_init) {
        colnum = col.num;
        gpuErrchk(cudaMalloc((void **)&_gpu_err, sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&_gpu_col, 4 * colnum));
        gpuErrchk(cudaMemcpy(_gpu_col, col.col, 4 * colnum, cudaMemcpyHostToDevice));
        cuda_has_init = true;
    }
}

void calc_cuda(fr_t fr, fr_col_t col, int my_h, int my_off, unsigned char * output) {
    calc_cuda_init(col);

    dim3 dimBlock(4, 4);
    dim3 dimGrid(fr.w / 4, my_h / 4);


    if (lw != fr.mem_w || lh != my_h) {
        if (_gpu_output != NULL) {
            cudaFree(_gpu_output);
        }
        gpuErrchk(cudaMalloc((void **)&_gpu_output, fr.mem_w * my_h));
    }


    // we dont need cudaMalloc(), because of ZEROcopy buffers that the GPU and CPU can share sys memory
    cuda_kernel<<<dimGrid, dimBlock>>>(fr, my_h, my_off, _gpu_col, colnum, _gpu_output, _gpu_err);


    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    int res_err = 0;

    gpuErrchk(cudaMemcpy(output, _gpu_output, fr.mem_w * my_h, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&res_err, _gpu_err, sizeof(int), cudaMemcpyDeviceToHost));

    if (res_err != 0) {
        fprintf(stderr, "result from cuda kernel is non-zero: %d\n", res_err);
    }

    lw = fr.mem_w;
    lh = my_h;

}


}
