//

#include "mandelbrot.h"
#include "mandelbrot_calc_c.h"

#include "tgmath.h"


void mand_c_init() {
    // do nothing
}


// return true if p is not in the bulb 0 (main cardoid)
inline bool bulb_check_0(double complex p) {
    double q = creal(p) - .25f;
    q = q*q + cimag(p) * cimag(p);
    return !(q * (q + (creal(p) - .25f)) < cimag(p) * cimag(p) / 4.0f);
}

// calculates iterations
void mand_c(fr_t fr, int my_h, int my_off, unsigned char * output) {
    int px, py, ci, ri, c0, c1;
    double fri, mfact, dppx;
    double complex z, c;
    dppx = 2.0 / (fr.w * fr.Z);
    log_trace("mand_c running center (%lf,%lf), zoom %lf, dim (%d,%d)", fr.cX, fr.cY, fr.Z, fr.w, my_h);
    for (px = 0; px < fr.w; ++px) {
        for (py = 0; py < my_h && my_off + py < fr.h; ++py) {
            ri = 4 * px + py * fr.mem_w;
            
            c = (fr.cX - 1.0 / fr.Z + px * dppx) + I * (fr.cY + fr.h / (fr.w * fr.Z) - (py + my_off) * dppx);
            z = c;
            c0 = 0;
            // COMPUTE FRACTALS HERE
            switch (fr.fractal_type) {
                case FR_MANDELBROT:
                    if (bulb_check_0(c)) { 
                        for (ci = 0; ci < fr.max_iter && abs(z) < 4.0; ++ci) {
                            z = z * z + c;
                        }
                        fri = 2.0 + ci - log(log(creal(z)*creal(z) + cimag(z)*cimag(z))) / log(2.0);
                    } else {
                        ci = fr.max_iter; 
                    }
                    break;
                case FR_MANDELBROT_3:
                    for (ci = 0; ci < fr.max_iter && abs(z) < 4.0; ++ci) {
                        z = z * z * z + c;
                    }
                    fri = 2.0 + ci - log(log(creal(z)*creal(z) + cimag(z)*cimag(z))) / log(3.0);
                    break;
                case FR_SIN:
                    for (ci = 0; ci < fr.max_iter && abs(z) < 4.0; ++ci) {
                        z = sin(z) + c;
                    }
                    fri = 0.0 + ci;
                    break;
                case FR_TESTING:
                    for (ci = 0; ci < fr.max_iter && abs(z) < 16.0; ++ci) {
                        z = z / c + c;
                    }
                    fri = 0.0 + ci;
                    break;
                default:
                    log_error("unknown fractal type");
                    exit(3);
                    break;

            }
            
            if (ci == fr.max_iter) {
                fri = 0.0;
            }

            //fri = fri * fr.cscale + fr.coffset;
            mfact = fri - floor(fri);
            mfact = 0;

            c0 = (int)floor(fri) % col.num;
            c1 = (c0 + 1) % col.num;

            c0 *= 4; c1 *= 4;
            #define MIX(a, b, F) ((b) * (F) + (a) * (1 - (F)))

            output[ri + 0] = (int)floor(MIX(col.col[c0 + 0], col.col[c1 + 0], mfact));
            output[ri + 1] = (int)floor(MIX(col.col[c0 + 1], col.col[c1 + 1], mfact));
            output[ri + 2] = (int)floor(MIX(col.col[c0 + 2], col.col[c1 + 2], mfact));
            output[ri + 3] = (int)floor(MIX(col.col[c0 + 3], col.col[c1 + 3], mfact));
        }
    }
}
