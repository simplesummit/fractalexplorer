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

#include "fr.h"
#include "log.h"

#define creal(a) a.x
#define cimag(a) a.y


/*

our complex number library, in cuda

*/


#define cabs(a) hypot(a.x*a.x, a.y*a.y)
#define cabs2(a) (a.x*a.x+a.y*a.y)
#define ccreate(x, y) ((cuDoubleComplex){ (x), (y) })


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

// 1 / x
__host__ __device__ static __inline__
cuDoubleComplex cuCrec(cuDoubleComplex x) {
    double to_scale = cabs2(x);
    return ccreate(x.x / to_scale, -x.y / to_scale);
}

// natural logarithm, base e ~= 2.71828
__host__ __device__ static __inline__
cuDoubleComplex cuClog(cuDoubleComplex x) {
    return ccreate(log(cabs2(x)) / 2.0, atan2(x.y, x.x));
}

// log_y(x), or log base y of x
__host__ __device__ static __inline__
cuDoubleComplex cuClogb(cuDoubleComplex x, cuDoubleComplex y) {
    return cuCdiv(cuClog(x), cuClog(y));
}

// x * x, or x ** 2, x squared
__host__ __device__ static __inline__
cuDoubleComplex cuCsqr(cuDoubleComplex x) {
    return ccreate(x.x*x.x - x.y*x.y, 2 * x.x*x.y);
}

// x ** y, but optimized for integers
__host__ __device__ static __inline__
cuDoubleComplex cuCpowi(cuDoubleComplex x, int y) {
    bool is_neg_pow = y < 0;
    y = abs(y);
    // holds track of x^{2^{exp bit}}
    cuDoubleComplex xt2eb = x, result = ccreate(1, 0);
    while (y > 0) {
        if (y & 1) {
            result = cuCmul(result, xt2eb);
        }
        xt2eb = cuCsqr(xt2eb);
        y >>= 1;
    }
    return (is_neg_pow) ? cuCrec(result) : result;
}


// x ** y, or x to the y power
__host__ __device__ static __inline__
cuDoubleComplex cuCpow(cuDoubleComplex x, cuDoubleComplex y) {
    return cuCexp(cuCmul(cuClog(x), y));
}

// sin, returns sin(x). Highly optimized method
// correct for all complex numbers
__host__ __device__ static __inline__
cuDoubleComplex cuCsin(cuDoubleComplex x) {
    return ccreate(sin(x.x) * cosh(x.y), cos(x.x) * sinh(x.y));
}


// cos, returns cos(x). Highly optimized and
// works for complex numbers
__host__ __device__ static __inline__
cuDoubleComplex cuCcos(cuDoubleComplex x) {
    return ccreate(cos(x.x) * cosh(x.y), -sin(x.x) * sinh(x.y));
}


// a macro to check a result and then print out info if failed, and (possibly)
// exit
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      log_error("GPUassert (at %s:%d) (code %d): %s\n", file, line, code,
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
    cuDoubleComplex z, c;

    c = ccreate(
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
                    z = cuCsqr(z);
                    z = cuCadd(z, c);
                }
                tmp = log(log(cabs2(z)));
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
            for (ci = 0; ci < fr.max_iter && fabs(creal(z)) < 16.0; ++ci) {
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
            for (ci = 0; ci < fr.max_iter && fabs(cimag(z)) < 16.0; ++ci) {
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
        int gpu_err_start = 0;
        gpuErrchk(cudaMemcpy(_gpu_err, &gpu_err_start, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void **)&_gpu_col, 4 * colnum));
        gpuErrchk(cudaMemcpy(_gpu_col, col.col, 4 * colnum, cudaMemcpyHostToDevice));
        cuda_has_init = true;
    }
}


// returns the grid from a block value such that b * grid_from_block(a, b) >= a,
// and grid_from_block(a, b) % b == 0
int grid_from_block(int a, int b) {
    return a / b + (a % b != 0);
}

void calc_cuda(fr_t fr, fr_col_t col, int my_h, int my_off, unsigned char * output) {
    calc_cuda_init(col);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(grid_from_block(fr.w,  dimBlock.x), 
                 grid_from_block(my_h, dimBlock.y));


    if (lw != fr.mem_w || lh != my_h) {
        if (_gpu_output != NULL) {
            cudaFree(_gpu_output);
        }
        gpuErrchk(cudaMalloc((void **)&_gpu_output, fr.mem_w * my_h));
    }

    log_debug("cuda kernel launched at center: %f%+f, zoom: %lf, iter: %d with grid: (%d,%d), block (%d,%d)", fr.cX, fr.cY, fr.Z, fr.max_iter, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // we dont need cudaMalloc(), because of ZEROcopy buffers that the GPU and CPU can share sys memory
    cuda_kernel<<<dimGrid, dimBlock>>>(fr, my_h, my_off, _gpu_col, colnum, _gpu_output, _gpu_err);


    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    int res_err = 0;

    gpuErrchk(cudaMemcpy(output, _gpu_output, fr.mem_w * my_h, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&res_err, _gpu_err, sizeof(int), cudaMemcpyDeviceToHost));

    if (res_err != 0) {
        log_error("result from cuda kernel is non-zero: %d\n", res_err);
    }

    lw = fr.mem_w;
    lh = my_h;

}


}
