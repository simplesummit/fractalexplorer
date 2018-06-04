/*

communication stuff between head and child nodes

*/


#include "fractalexplorer.h"
#include "commloop.h"

#include "control_loop.h"
#include "visuals.h"

#include "engine_c.h"

#include <mpi.h>

#include <stdbool.h>


diagnostics_t * diagnostics_history = NULL;


void send_workload(workload_t workload, int node) {
    MPI_Request r0, r1;
    MPI_Isend(&workload.assigned_cols_len, 1, MPI_INT, node, 0, MPI_COMM_WORLD, &r0);
    MPI_Isend(workload.assigned_cols, workload.assigned_cols_len, MPI_INT, node, 0, MPI_COMM_WORLD, &r1);
    MPI_Request_free(&r0);
    MPI_Request_free(&r1);
}


void recv_workload(workload_t * workload) {
    MPI_Recv(&workload->assigned_cols_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(workload->assigned_cols, workload->assigned_cols_len, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}



void master_loop() {
    diagnostics_history = malloc(sizeof(diagnostics_t) * NUM_DIAGNOSTICS_SAVE);

    node_diagnostics_t _dft;

    _dft.temperature = 0.0f;
    _dft.time_compute = 0.0f;
    _dft.time_compress = 0.0f;
    _dft.time_transfer = 0.0f;
    _dft.time_decompress = 0.0f;
    _dft.time_total = 0.0f;

    _dft.total_rows = -1;

    int i, j;
    for (i = 0; i < NUM_DIAGNOSTICS_SAVE; ++i) {
        diagnostics_history[i].node_information = (node_diagnostics_t *)malloc(sizeof(node_diagnostics_t) * world_size);
        for (j = 0; j < world_size; ++j) {
            diagnostics_history[i].node_information[j] = _dft;
        }
    }


    /* start main loop here */

    bool keep_going = true;

    MPI_Request * send_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);

    MPI_Request * recv_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);
    MPI_Status * recv_statuses = (MPI_Status *)malloc(sizeof(MPI_Status) * world_size);


    workload_t * node_workloads = (workload_t *)malloc(sizeof(workload_t) * world_size);
    // arrays of columns
    unsigned char ** node_results = (unsigned char **)malloc(sizeof(unsigned char *) * world_size);
    int * node_results_len = (int *)malloc(sizeof(int) * world_size);


    unsigned char * total_image = (unsigned char *)malloc(3 * fractal_params.width * fractal_params.height);

    for (i = 1; i < world_size; ++i) {
        node_workloads[i].assigned_cols_len = 0;
        node_workloads[i].assigned_cols = (int *)malloc(sizeof(int) * fractal_params.width);
        node_results[i] = (unsigned char *)malloc(3 * fractal_params.width * fractal_params.height);
        memset(node_results[i], 0, 3 * fractal_params.width * fractal_params.height);
    }


    int to_send;


    while (keep_going) {

        control_update_t control_update = control_update_loop();

        // start them all off at 0
        for (i = 1; i < world_size; ++i) {
            node_workloads[i].assigned_cols_len = 0;
        }

        for (i = 0; i < fractal_params.width; ++i) {
            int assigned_worker = 1 + (i % compute_size);
            node_workloads[assigned_worker].assigned_cols[node_workloads[assigned_worker].assigned_cols_len++] = i;
        }

        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);

        int node_workload_size;
        for (i = 1; i < world_size; ++i) {
            send_workload(node_workloads[i], i);

            node_workload_size = 3 * fractal_params.height * node_workloads[i].assigned_cols_len;

            MPI_Irecv(node_results[i], node_workload_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
        }



        for (i = 1; i < world_size; ++i) {

            MPI_Wait(&recv_requests[i], &recv_statuses[i]);
            
            int col;

            for (j = 0; j < node_workloads[i].assigned_cols_len; ++j) {
                col = node_workloads[i].assigned_cols[j];
                //memcpy(total_image + 3 * fractal_params.height * col, node_results + 3 * fractal_params.height * j, 3 * fractal_params.height);
                int k, to_idx, from_idx;
                for (k = 0; k < fractal_params.height; ++k) {
                    to_idx = 3 * (fractal_params.height * col + k);
                    from_idx = 3 * (fractal_params.height * j + k);

                    total_image[to_idx + 0] = node_results[i][from_idx + 0];
                    total_image[to_idx + 1] = node_results[i][from_idx + 1];
                    total_image[to_idx + 2] = node_results[i][from_idx + 2];
                }

                //memset(total_image + 3 * fractal_params.height * col, 0, 3 * fractal_params.height);
            }
        }

        visuals_update(total_image);
        

        // sync up optionally

    }
    for (i = 0; i < world_size; ++i) {
        free(node_workloads[i].assigned_cols);
        free(node_results[i]);
    }

    free(total_image);

    free(node_workloads);
    free(node_results);

    free(node_results_len);

    free(send_requests);
    free(recv_requests);
    free(diagnostics_history);
}


void slave_loop() {

    bool keep_going = true;

    workload_t my_workload;
    my_workload.assigned_cols = (int *)malloc(sizeof(int) * fractal_params.width);
    unsigned char * my_result = malloc(3 * fractal_params.width * fractal_params.height);
    memset(my_result, 0, 3 * fractal_params.width * fractal_params.height);
    
    int my_result_size = 0;

    // initialize engine
    engine_c_init();
    

    while (keep_going) {
        // receive any updates
        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);

        recv_workload(&my_workload);

        engine_c_compute(my_workload, my_result);

        my_result_size = 3 * fractal_params.height * my_workload.assigned_cols_len;

        MPI_Send(my_result, my_result_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);

    }
}



