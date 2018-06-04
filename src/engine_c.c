/*

C computation engine


*/

#include "engine_c.h"
#include "fractalexplorer.h"


void engine_c_init() {
    // init method, but C doesn't need it
}


// returns 3byte packed pixels of successive rows
void engine_c_compute(workload_t workload, unsigned char * output) {
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


    for (col_idx = 0; col_idx < workload.assigned_cols_len; ++col_idx) {
        col = workload.assigned_cols[i];

        for (row = 0; row < fractal_params.height; ++row) {
            output_idx = 3 * (col_idx * fractal_params.height + row);

            // assign color
            output[output_idx + 0] = 255 * row / fractal_params.height;
            output[output_idx + 1] = 255;
            output[output_idx + 2] = 0;
        }
    }
}

