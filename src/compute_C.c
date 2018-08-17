/* compute_C.c -- C engine to do the actual fractal computing


*/


#include "fractalexplorer.h"


// output is guaranteed to be sizeof(packed_color_t) * workload.length * fractal.width
void compute_C(workload_t workload, packed_color_t * output) {
    int py;
    for (py = workload.start; py < workload.start + workload.length && py < fractal.height; ++py) {
        int px;
        for (px = 0; px < fractal.width; ++px) {
            // this indexing is because the output array starts with our first row
            int output_idx = (py - workload.start) * fractal.width + px;
            packed_color_t output_val;

            double c_r = fractal.center_r + (double)(2 * px - fractal.width) / (fractal.zoom * fractal.width);
            // sign is flipped because  top is 0
            double c_i = fractal.center_i - (double)(2 * py - fractal.height) / (fractal.zoom * fractal.width);

            double z_r = c_r, z_i = c_i;

            // store the squares of these to save multiplications
            double z_r2 = z_r * z_r, z_i2 = z_i * z_i;

            int i;
            for (i = 0; i < fractal.max_iter && z_r2 + z_i2 <= 16.0; ++i) {
                z_i = 2.0 * z_r * z_i + c_i;
                z_r = z_r2 - z_i2 + c_r;
                z_r2 = z_r * z_r;
                z_i2 = z_i * z_i;
            }

            double out_d = 1 + i - log(log(z_r2 + z_i2)) / log(2.0);

            output_val.index = (int)floor(out_d) % palette.num_colors;
            output_val.prop = (uint8_t)(256.0 * (out_d - floor(out_d)));

            output[output_idx] = output_val;
        }
    }
}


