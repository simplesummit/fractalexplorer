/*

C computation engine


*/

#include "engine_c.h"
#include "fractalexplorer.h"
#include "math.h"
#include "complex.h"


void engine_c_init() {
    // init method, but C doesn't need it for anything
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

    /*

    complex numbers

    */
    
    double complex z, c;

    // shouldn't change
    double complex q = fractal_params.q_r + I * fractal_params.q_i;
    

    // for optimized routines
    double z_r, z_i, c_r, c_i;
    double z_r2, z_i2;


    // shouldn't change
    double q_r = fractal_params.q_r, q_i = fractal_params.q_i;


    for (col_idx = 0; col_idx < workload.assigned_cols_len; ++col_idx) {
        col = workload.assigned_cols[col_idx];
        col_iters = 0;

        for (row = 0; row < fractal_params.height; ++row) {
            output_idx = col_idx * fractal_params.height + row;

            /*
            // complex numbers

            c = X_PIXEL_TO_RE(col, fractal_params.width, fractal_params.height, fractal_params.center_r, fractal_params.zoom) + I * Y_PIXEL_TO_IM(row, fractal_params.width, fractal_params.height, fractal_params.center_i, fractal_params.zoom);


            if (fractal_params.type == FRACTAL_TYPE_MANDELBROT) {
                z = c;

                for (iter = 0; iter < fractal_params.max_iter && cabs(z) <= 16.0; ++iter) {
                    z = z * z + c;
                }

                partial_iteration = 2 + iter - log(log(cabs(z))) / log(2.0);
            }

            */

            // using doubles only
            c_r = X_PIXEL_TO_RE(col, fractal_params.width, fractal_params.height, fractal_params.center_r, fractal_params.zoom);
            c_i = Y_PIXEL_TO_IM(row, fractal_params.width, fractal_params.height, fractal_params.center_i, fractal_params.zoom);

            c = c_r + I * c_i;
            
            if (fractal_params.type == FRACTAL_TYPE_MANDELBROT) {
                // z**2 + c
                z_r = c_r;
                z_i = c_i;

                z_r2 = z_r * z_r;
                z_i2 = z_i * z_i;

                for (iter = 0; iter < fractal_params.max_iter && z_r2 + z_i2 <= 256.0; ++iter) {
                    z_i = 2 * z_r * z_i + c_i;
                    z_r = z_r2 - z_i2 + c_r;
                    z_r2 = z_r * z_r;
                    z_i2 = z_i * z_i;  
                }

                partial_iteration = 3 + iter - log(log(z_r2 + z_i2)) / log(2.0);
            } else if (fractal_params.type == FRACTAL_TYPE_MULTIBROT) {
                // z**q + c

                z = c;

                for (iter = 0; iter < fractal_params.max_iter && cabs(z) <= 16.0; ++iter) {
                    z = cpow(z, q) + c;
                }

                partial_iteration = 2 + iter - log(log(cabs(z))) / log(cabs(q));
            } else if (fractal_params.type == FRACTAL_TYPE_JULIA) {
                // z**2 + q

                z = c;

                for (iter = 0; iter < fractal_params.max_iter && cabs(z) <= 16.0; ++iter) {
                    z = z * z + q;
                }

                partial_iteration = 2 + iter - log(log(cabs(z))) / log(2.0);
            }


            // these work for all fractals
            
            double color_final = partial_iteration;

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

            _output_rgb[output_idx] = cur_color;
        }
        
        output_iters[col_idx] = col_iters;
    }
}

