/* calc_c.c -- the default C engine that will work on all platforms

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
#include <math.h>
#include <complex.h>

// type generic math macros, which finds the correct function based on input.
// this slows down compile time at the preprocessing step, and this should be
// removed and instead use `complex.h` with cabs, csin, and similarly named
// functions
//#include "tgmath.h"

// additional macros for complex numbers
// returns the square of the absolute value (cabs(a)*cabs(a))
#define cabs2(a) (creal(a)*creal(a)+cimag(a)*cimag(a))


// a macro to scale between values. when F = 0, MIX macro takes the
// value of a, when F = 1.0, the macro takes the value of b, and all
// values inbetween are linearly scaled

// these two are called with () around arguments from MIX
#define __LIN_MIX(a, b, F) (b * F + a * (1 - F))
#define __CUBIC_MIX_FLIP(F) (1-(1-F)*(1-F)*(1-F))
#define __CUBIC_MIX(a, b, F) __LIN_MIX(a, b, __CUBIC_MIX_FLIP(F))
#define __CBRT_MIX(a, b, F) __LIN_MIX(a, b, (cbrt(F)))

#define MIX(a, b, F) __LIN_MIX((a), (b), (F))

// keeps track of whether the engine has initialized
bool c_has_init = false;

// initialize any buffers, caches, etc here. The C engine uses normal system
// memory, so no special buffers are required
void calc_c_init() {
    if (!c_has_init) {
        c_has_init = true;
    }
}


// return true if p is in the bulb 0 (main cardoid) for the mandelbrot set. this
// greatly reduces render times when looking at the main bulb, and should be
// inlined to reduce the cost of possible function stack jumping within the
// compute loop
inline bool bulb_check_0(double complex p) {
    // this comes from Wikipedia:
    // https://en.wikipedia.org/wiki/Mandelbrot_set#Optimizations
    // and is an efficient way to rule out points that we know will take all
    // max_iter iterations to process, when we could easily break out ealier.
    // this is essentially a polar inequality check of a cardoid.
    double q = creal(p) - .25f;
    q = q*q + cimag(p) * cimag(p);
    return q * (q + (creal(p) - .25f)) < cimag(p) * cimag(p) / 4.0f;
}


// uses parameters from fr, and custom parameters my_* to output in output. See
// fr.h for an explanation of parameters fr_t struct members.
// my_h and my_off are specific to compute nodes, and should not be included in
// the struct fr_t that is transferred to all nodes. my_off represents the
// vertical offset the current compute node is from (0, 0). my_h represents the
// number of rows the current compute node should compute all column pixel
// values. output is a buffer (which should be pre-allocated) that should be
// written to as a byte array of RGBA values that represent the pixels starting
// at (0, my_off) and lasting till (fr.w - 1, my_off + my_h), with a pixel depth
// of 4 bytes/px. Although, the index of point (px, py) is determined by
// fr.mem_w (see fr.h for more on this)
void calc_c(fr_t fr, int my_h, int my_off, unsigned char * output) {
    // ensure the engine is initialized
    calc_c_init();

    // current x, y pixels
    int px, py;

    // current index, and the index into output buffer
    int ci, ri;

    // the two color indexes (for mixing betweem colors)
    int c0, c1;

    // fractional return index, for mixing, mixing factor (0.0 through 1.0) and
    // the delta per pixel (this is mentioned in fr.h)
    double tmp, fri, mfact, dppx;

    // complex numbers representing the current iteration, and the starting
    // value for a pixel
    double complex z, c;

    // set this so that we have uniform distances between pixels
    dppx = 2.0 / (fr.w * fr.Z);

    // start compute loop
    log_trace("mand_c running center (%lf,%lf), zoom %lf, dim (%d,%d)",
              fr.cX, fr.cY,
              fr.Z,
              fr.w, my_h);

    // fr.w is the bounding box, and the only reason we use fr.mem_w is for
    // buffer sizes
    for (px = 0; px < fr.w; ++px) {
        // we need to make sure we don't overshoot fr.h, and we only do (at max)
        // my_h rows
        for (py = 0; py < my_h && my_off + py < fr.h; ++py) {

            // return index is fr.mem_w (which includes the factor of 4) and
            // the factor of 4 times the horizontal offset. Then, 0 bytes away
            // is R, 1 is B, 2 is G, 3 is A
            ri = 4 * px + py * fr.mem_w;

            // scale so that pixels are even, and the center is truly in the
            // center of the image
            // for visualization, the line is split into real and complex
            // portions, which are both independently driven by px and py
            // respectively.
            // once set in this loop, `c` should not be changed, as it
            // is expected to represent the coordinate value at (px, py)
            c = 1 * (fr.cX - 1.0 / fr.Z + px * dppx) +
                I * (fr.cY + fr.h / (fr.w * fr.Z) - (py + my_off) * dppx);

            // start `z` at `c` (most fractals will use this)
            z = c;

            // here, we change based on the current fractal type enum which is
            // in fr. Each of these cases should:
            // 1. iterate until either `ci` exceeds fr.max_iter, or we've
            //      detected the point has escaped
            // 2. set `fri` to the fractional iteration count (an integer will
            //      work as well), or: in a case that we've detected that the
            //      point will be inside the set (such as bulb_check_0), set
            //      `ci` to fr.max_iter and skip further computations. If we
            //      know the point will escape instantly, set `ci` to 0 and skip
            //      further computations
            // see fr.h for a list that fr.fractal_type can be. Typically, those
            // have comments on the equation and behaviour of the fractal type
            switch (fr.fractal_type) {
                case FR_MANDELBROT:
                    // see above in this file, this method determines whether
                    // we should skip the computation
                    if (!bulb_check_0(c)) {

                        // loop through following condition `1` listed before
                        // the switch statement.
                        // the default error value (16.0) to determine escape
                        // state. For the mandelbrot set, this can be lower (as
                        // low as 2 to correctly identify all escaped points),
                        // however, this leads to some visual artifacts and
                        // should be kept >= 4.0, with 16.0 further reducing
                        // visual imperfections
                        for (ci = 0; ci < fr.max_iter && cabs(z) <= 16.0; ++ci) {

                            // this is the standard function, z = z ** 2 + c
                            z = z * z + c;

                        }

                        // this is part of the fractional iteration count, and
                        // the divisor (by default log(2.0)) can be changed for
                        // interesting effects. For example, try 2.5 for a `tank
                        // treads` aesthetic, 5 for a `flower` aesthetic
                        // this method for colorizing is found (at least
                        // partially) here:
                        // https://linas.org/art-gallery/escape/escape.html
                        // We have modified it a bit to produce even more
                        // pleasing images
                        tmp = log(log(cabs2(z)));
                        fri = 2.0 + ci - tmp / log(2.0);
                    } else {
                        // skip out early, and set that this point will be part
                        // of the mandelbrot set
                        ci = fr.max_iter;
                        fri = ci + 0.0;
                    }
                    break;
                case FR_MANDELBROT_3:
                    // similar to the default mandelbrot, we will loop and use
                    // an escape value of 16.0. However, we have no speedups,
                    // like bulb_check_0 for this, so we will just iterate
                    // the function
                    for (ci = 0; ci < fr.max_iter && cabs(z) <= 16.0; ++ci) {
                        z = z * z * z + c;
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
                        z = cexp(z) + c;
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
                        z = csin(z) + c;
                    }
                    // no current way to easily do a fractional iteration, so
                    // just send a fractional iteration of the actual integer
                    // value
                    fri = 0.0 + ci;
                    break;
                default:
                    // this should never happen
                    log_error("unknown fractal type");
                    exit(3);
                    break;

            }

            // if ci is set to default values, if they have set ci to 0 or max
            // the computation might iter,
            // set the fri to corresponding values
            if (ci == fr.max_iter) {
                fri = 0.0 + fr.max_iter;
            }

            if (fr.fractal_flags & FRF_ADD_PERIOD) {
                tmp = floor(fri)-fri;
                fri += ((1-tmp)*carg(z) + (tmp)*(carg(z*z+c)));
            }

            // binary decomposition
            if (fr.fractal_flags & FRF_BINARYDECOMP_REAL && creal(z) >= 0) {
                fri += 1.0;
            } 
            if (fr.fractal_flags & FRF_BINARYDECOMP_IMAG && cimag(z) >= 0) {
                fri += 2.0;
            }
            // TODO: implement scale and offset for function
            fri = fri * fr.cscale + fr.coffset;

            // toggle for gradients or not. Keep FRF_SIMPLE as a flag to
            // increase compression ratios
            if (fr.fractal_flags & FRF_SIMPLE) {
                mfact = 0;
            } else {
                mfact = fri - floor(fri);
            }

            // TODO: add toggle to do simple or complex coloring, right now we
            // are just forcing simple

            // convert this to an index
            c0 = (int)floor(fri) % col.num;
            // the color is mixed by the next in the result index
            c1 = (c0 + 1) % col.num;

            // the offset is 4 times the index
            c0 *= 4; c1 *= 4;

            // a macro to scale between values. when F = 0, MIX macro takes the
            // value of a, when F = 1.0, the macro takes the value of b, and all
            // values inbetween are linearly scaled

            // set each to the mix between colorscheme values
            output[ri + 0] = (int)floor(MIX(col.col[c0 + 0], col.col[c1 + 0], mfact));
            output[ri + 1] = (int)floor(MIX(col.col[c0 + 1], col.col[c1 + 1], mfact));
            output[ri + 2] = (int)floor(MIX(col.col[c0 + 2], col.col[c1 + 2], mfact));
            output[ri + 3] = (int)floor(MIX(col.col[c0 + 3], col.col[c1 + 3], mfact));
        }
    }
}
