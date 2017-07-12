//

#include "mandelbrot.h"
#include "mandelbrot_calc_c.h"
#include "math.h"


void mand_c_init() {
    // do nothing
}


// calculates iterations
void mand_c(fr_t fr, int my_h, int my_off, unsigned char * output) {
    int px, py, ci, ri, c0, c1;
    double c_r, _c_i, c_i, z_r, z_i, z_r2, z_i2, _q, fri, mfact, dppx;
    _c_i = fr.cY + fr.h / (fr.w * fr.Z);
    c_r = fr.cX - 1.0 / fr.Z;
    dppx = 2.0 / (fr.w * fr.Z);
    _c_i -= dppx * my_off;
    log_trace("mand_c running center (%lf,%lf), zoom %lf, dim (%d,%d)", fr.cX, fr.cY, fr.Z, fr.w, my_h);
    for (px = 0; px < fr.w; ++px) {
        c_i = _c_i;
        for (py = 0; py < my_h && my_off + py < fr.h; ++py) {
            ri = 4 * px + py * fr.mem_w;
            z_r = c_r;
            z_i = c_i;
            z_r2 = z_r * z_r;
            z_i2 = z_i * z_i;
            _q = (z_r - .25f);
            _q = _q * _q + z_i2;
            if (_q * (_q + (z_r - .25f)) < z_i2 / 4.0f) { 
                ci = fr.max_iter; 
            } else {
                ci = 0;
            }
            for (; ci < fr.max_iter && z_r2 + z_i2 < 16.0; ++ci) {
                z_i = 2 * z_r * z_i + c_i;
                z_r = z_r2 - z_i2 + c_r;
                z_r2 = z_r * z_r;
                z_i2 = z_i * z_i;
            }
            if (ci == fr.max_iter) {
                fri = 0.0;
            } else {
                fri = 2.0 + ci - log(log(z_r2 + z_i2)) / log(2.0);
            }
            //fri = fri * fr.cscale + fr.coffset;
            mfact = fri - floor(fri); mfact = 0;
            c0 = (int)floor(fri) % col.num;
            c1 = (c0 + 1) % col.num;

            c0 *= 4; c1 *= 4;
            #define MIX(a, b, F) ((b) * (F) + (a) * (1 - (F)))

            output[ri + 0] = (int)floor(MIX(col.col[c0 + 0], col.col[c1 + 0], mfact));
            output[ri + 1] = (int)floor(MIX(col.col[c0 + 1], col.col[c1 + 1], mfact));
            output[ri + 2] = (int)floor(MIX(col.col[c0 + 2], col.col[c1 + 2], mfact));
            output[ri + 3] = (int)floor(MIX(col.col[c0 + 3], col.col[c1 + 3], mfact));
            c_i -= dppx;
        }
    c_r += dppx;
    }
}
