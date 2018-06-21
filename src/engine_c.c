/*

C computation engine


*/

#include "engine_c.h"
#include "fractalexplorer.h"
#include "math.h"


void engine_c_init() {
    // init method, but C doesn't need it
}


// returns 3byte packed pixels of successive rows, and stores the number of iterations for each column in output_iters
void engine_c_compute(workload_t workload, unsigned char * output, int * output_iters) {
    int col_idx;

    int row, col;

    int output_idx = 0;

    int i;

    /*
    for (i = 0; i < workload.assigned_cols_len; ++i) {
        printf("%d,", workload.assigned_cols[i]);
    }
    printf("\n");

    */

    double z_r, z_i, c_r, c_i;

    double z_r2, z_i2;


    int iter;
    
    int col_iters;

    RGB_t cur_color;

    RGB_t * _output_rgb = (RGB_t *)output;

    RGB_t col_before, col_after;

    double partial_iteration;

    RGB_t _local_color_scheme[MAX_COLOR_SCHEME_LENGTH];

    double final_phase;

    for (i = 0; i < color_scheme.len; ++i) {
        _local_color_scheme[i].R = color_scheme.rgb_vals[3 * i + 0];
        _local_color_scheme[i].G = color_scheme.rgb_vals[3 * i + 1];
        _local_color_scheme[i].B = color_scheme.rgb_vals[3 * i + 2];
    }

    for (col_idx = 0; col_idx < workload.assigned_cols_len; ++col_idx) {
        col = workload.assigned_cols[col_idx];
        col_iters = 0;

        for (row = 0; row < fractal_params.height; ++row) {
            output_idx = col_idx * fractal_params.height + row;

            c_r = X_PIXEL_TO_RE(col, fractal_params.width, fractal_params.height, fractal_params.center_r, fractal_params.zoom);
            c_i = Y_PIXEL_TO_IM(row, fractal_params.width, fractal_params.height, fractal_params.center_i, fractal_params.zoom);
            
            z_r = c_r;
            z_i = c_i;

            z_r2 = z_r * z_r;
            z_i2 = z_i * z_i;

            for (iter = 0; iter < fractal_params.max_iter && z_r2 + z_i2 <= 16.0; ++iter) {
                z_i = 2 * z_r * z_i + c_i;
                z_r = z_r2 - z_i2 + c_r;
                z_r2 = z_r * z_r;
                z_i2 = z_i * z_i;  
            }

            final_phase = atan2(c_i, c_r);

            partial_iteration = 2 + iter - log(log(z_r2 + z_i2)) / log(2.0);

            double color_final = iter;// partial_iteration;

            double partial_int = floor(color_final);
            double gradient = color_final - partial_int;

            col_iters += iter;

            int before_idx = (((int)partial_int) % color_scheme.len + color_scheme.len) % color_scheme.len;
            int after_idx = (before_idx + 1) % color_scheme.len;

            col_before = _local_color_scheme[before_idx];
            col_after = _local_color_scheme[after_idx];

            // mix them
            cur_color.R = lin_mix(col_before.R, col_after.R, gradient);
            cur_color.G = lin_mix(col_before.G, col_after.G, gradient);
            cur_color.B = lin_mix(col_before.B, col_after.B, gradient);

            //cur_color.R = (10 * iter) % 256;
            //cur_color.G = 0;
            //cur_color.B = 0;

            // assign color
            //output[output_idx + 0] = 255 * col / fractal_params.width;
            _output_rgb[output_idx] = cur_color;
        }
        
        output_iters[col_idx] = col_iters;
    }
}

