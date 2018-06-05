/*

C computation engine


*/

#include "engine_c.h"
#include "fractalexplorer.h"


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

    for (col_idx = 0; col_idx < workload.assigned_cols_len; ++col_idx) {
        col = workload.assigned_cols[col_idx];
        col_iters = 0;

        for (row = 0; row < fractal_params.height; ++row) {
            output_idx = 3 * (col_idx * fractal_params.height + row);

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

            col_iters += iter;

            // assign color
            //output[output_idx + 0] = 255 * col / fractal_params.width;
            output[output_idx + 0] = (10 * iter) % 256;
            output[output_idx + 1] = 0;
            output[output_idx + 2] = 0;
        }
        
        output_iters[col_idx] = col_iters;
    }
}

