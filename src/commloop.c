/*

communication stuff between head and child nodes

*/


#include "fractalexplorer.h"
#include "commloop.h"

#include "control_loop.h"
#include "visuals.h"
#include "fr.h"

#include "lz4.h"

#include "engine_c.h"
#ifdef HAVE_CUDA
#include "engine_cuda.h"
#endif

#include <mpi.h>

#include <stdbool.h>


// stores the idx of the currently writing diagnostics history
// but, to predict workloads, get the previous (in a rolling buffer)
int diagnostics_history_idx = 0;
diagnostics_t * diagnostics_history = NULL;

int n_frames = 0;


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
    _dft.time_total = 0.0f;

    _dft.total_cols = -1;

    int i, j;
    for (i = 0; i < NUM_DIAGNOSTICS_SAVE; ++i) {
        // default them out

        diagnostics_history[i].time_control_update = 0.0f;
        diagnostics_history[i].time_assign = 0.0f;
        diagnostics_history[i].time_wait = 0.0f;
        diagnostics_history[i].time_decompress = 0.0f;
        diagnostics_history[i].time_recombo = 0.0f;
        diagnostics_history[i].time_visuals = 0.0f;
        diagnostics_history[i].time_total = 0.0f;
        diagnostics_history[i].node_assignments = (int *)malloc(sizeof(int) * fractal_params.width);
        diagnostics_history[i].col_iters = (int *)malloc(sizeof(int) * fractal_params.width);
        
        diagnostics_history[i].node_information = (node_diagnostics_t *)malloc(sizeof(node_diagnostics_t) * world_size);
        for (j = 0; j < world_size; ++j) {
            diagnostics_history[i].node_information[j] = _dft;
        }
    }

    // float packed diagnostics information
    // sent: temperature (dummy rn), time_cmopute, time_compress
    float ** recv_diagnostics = (float **)malloc(sizeof(float *) * world_size);
    unsigned char ** uncompressed_workloads = (unsigned char **)malloc(sizeof(unsigned char *) * world_size);

    int compress_bound = LZ4_compressBound(3 * fractal_params.width * fractal_params.height);

    for (i = 0; i < world_size; ++i) {
        uncompressed_workloads[i] = (char *)malloc(3 * fractal_params.width * fractal_params.height);

        recv_diagnostics[i] = (float *)malloc(sizeof(float) * 5);
        for (j = 0; j < 5; ++j) {
            recv_diagnostics[i][j] = 0.0f;
        }
    }




    /* start main loop here */

    bool keep_going = true;

    MPI_Request * send_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);
    MPI_Request * recv_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);
    MPI_Request * diagnostics_recv_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);
    MPI_Request * col_iters_recv_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);

    MPI_Status * recv_statuses = (MPI_Status *)malloc(sizeof(MPI_Status) * world_size);


    workload_t * node_workloads = (workload_t *)malloc(sizeof(workload_t) * world_size);
    workload_t * previous_node_workloads = (workload_t *)malloc(sizeof(workload_t) * world_size);

    
    // arrays of columns
    unsigned char ** node_results_recv = (unsigned char **)malloc(sizeof(unsigned char *) * world_size);
    int * node_results_recv_len = (int *)malloc(sizeof(int) * world_size);

    // how many iters is in each column
    int ** recv_col_iters = (int **)malloc(sizeof(int *) * world_size);
    for (i = 0; i < world_size; ++i) {
        recv_col_iters[i] = (int *)malloc(sizeof(int) * fractal_params.width);
    }


    // these are copied into so we can do smart things about when to use it, and the buffer doesnt conflict
    unsigned char ** prev_node_results = (unsigned char **)malloc(sizeof(unsigned char *) * world_size);
    int * prev_node_results_len = (int *)malloc(sizeof(int) * world_size);


    unsigned char * total_image = (unsigned char *)malloc(3 * fractal_params.width * fractal_params.height);

    for (i = 1; i < world_size; ++i) {
        node_workloads[i].assigned_cols_len = 0;
        previous_node_workloads[i].assigned_cols_len = 0;
        node_workloads[i].assigned_cols = (int *)malloc(sizeof(int) * fractal_params.width);
        previous_node_workloads[i].assigned_cols = (int *)malloc(sizeof(int) * fractal_params.width);
        node_results_recv[i] = (unsigned char *)malloc(compress_bound);
        prev_node_results[i] = (unsigned char *)malloc(compress_bound);
        //memset(node_results[i], 0, 3 * fractal_params.width * fractal_params.height);
    }

    // assigning utils:

    // in proportion of iter/sec of each node
    float * performance_proportion = (float *)malloc(sizeof(float) * world_size);

    // how many to assign
    int * worker_assign_count = (int *)malloc(sizeof(int) * world_size);

    // work proportion by column
    float * col_work_proportion = (float *)malloc(sizeof(float) * fractal_params.width);


    // this is used for status codes to quit, or send some other signal
    // -1 means keep going
    int to_send = -1;

    tperf_t total_perf, recombo_perf, visuals_perf, waiting_perf, assigning_perf, control_update_perf;

    tperf_init(total_perf);
    tperf_init(recombo_perf);
    tperf_init(visuals_perf);
    tperf_init(waiting_perf);
    tperf_init(assigning_perf);
    tperf_init(control_update_perf);

    control_update_init();

    while (keep_going) {
        log_trace("start frame %d", n_frames);
        
        // for calculating loop performance
        tperf_start(total_perf);

        MPI_Bcast(&to_send, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // this means quit without error
        if (to_send == 0) {
            M_EXIT(0);
        }
        

        tperf_start(assigning_perf);

        // start them all off at 0
        for (i = 1; i < world_size; ++i) {
            node_workloads[i].assigned_cols_len = 0;
        }

        // assign everything here
        // ASSIGN ALGORITHM, probably needs work
        if (n_frames > 0 && true) {
            // previous data - sequential and completely proportional

            diagnostics_t previous_data = diagnostics_history[(diagnostics_history_idx - 1 + NUM_DIAGNOSTICS_SAVE) % NUM_DIAGNOSTICS_SAVE];
            // in iter/s
            float total_performance = 0.0f;
            float cur_performance = 0.0f;
            float total_col_work = 0.0f;
            int cur_node;
            for (i = 1; i < world_size; ++i) {
                performance_proportion[i] = 0.0f; 
            }
            for (i = 0; i < fractal_params.width; ++i) {
                col_work_proportion[i] = 0.0f;
            }

            for (i = 0; i < fractal_params.width; ++i) {
                cur_node = previous_data.node_assignments[i];
                cur_performance = previous_data.col_iters[i] / previous_data.node_information[cur_node].time_compute;
                performance_proportion[cur_node] += cur_performance;
                total_performance += cur_performance;

                // and calc column work

                col_work_proportion[i] += previous_data.col_iters[i];
                total_col_work += col_work_proportion[i];
            }
            
            // normalize some values beforehand (detect near neibors)
            /*
            for (i = 1 + 1; i < world_size; ++i) {
                if (((float)world_size * (performance_proportion[i] - performance_proportion[i - 1])) / total_performance < .4f) {
                    float nval = (performance_proportion[i] + performance_proportion[i - 1]) / 2.0f;
                    performance_proportion[i - 1] = nval;
                    performance_proportion[i] = nval;
                }
            }
            */


            for (i = 1; i < world_size; ++i) {
                performance_proportion[i] /= total_performance;
            }

            for (i = 0; i < fractal_params.width; ++i) {
                col_work_proportion[i] /= total_col_work;
            }

            // performance is calculated
        
            int cur_worker_assigning = 1;
            float cur_proportion_goal = performance_proportion[cur_worker_assigning];

            float cumulative_work_assigned = 0.0f;

            int given_to_cur_worker = 0;


            for (i = 0; i < fractal_params.width; ++i) {
                // mimum assignment of 10
                if (given_to_cur_worker >= 10 && cumulative_work_assigned >= cur_proportion_goal && cur_worker_assigning < world_size - 1) {
                    cur_worker_assigning++;
                    cur_proportion_goal += performance_proportion[cur_worker_assigning];
                }
                // now assign
                int assigned_worker = cur_worker_assigning;
                node_workloads[assigned_worker].assigned_cols[node_workloads[assigned_worker].assigned_cols_len++] = i;
                diagnostics_history[diagnostics_history_idx].node_assignments[i] = assigned_worker;
                cumulative_work_assigned += col_work_proportion[i];
                given_to_cur_worker++;
            }

            
        } else {
            // the first frame is default assigned
            for (i = 0; i < fractal_params.width; ++i) {
                int assigned_worker = 1 + (i % compute_size);
                // assigned randomly
                //int assigned_worker = 1 + (rand() % compute_size);
                //int assigned_worker = 1 + i / (fractal_params.width/compute_size);
                //int assigned_worker = 1 + (((int)(20 * (sinf(.4 * i) + 1)) % (compute_size)));
                node_workloads[assigned_worker].assigned_cols[node_workloads[assigned_worker].assigned_cols_len++] = i;
                diagnostics_history[diagnostics_history_idx].node_assignments[i] = assigned_worker;
            }
        }

        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);

        int node_workload_size;
        for (i = 1; i < world_size; ++i) {
            send_workload(node_workloads[i], i);

            node_workload_size = 3 * fractal_params.height * node_workloads[i].assigned_cols_len;

            MPI_Irecv(node_results_recv[i], LZ4_compressBound(node_workload_size), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
            MPI_Irecv(recv_col_iters[i], node_workloads[i].assigned_cols_len, MPI_INT, i, 0, MPI_COMM_WORLD, &col_iters_recv_requests[i]);
            MPI_Irecv(recv_diagnostics[i], 5, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &diagnostics_recv_requests[i]);
        }

        tperf_end(assigning_perf);
        diagnostics_history[diagnostics_history_idx].time_assign = assigning_perf.elapsed_s;

        /* FREE COMPUTE TIME WHILE WAITING FOR Irecv's to go through */



        // we could do all the visual stuff one frame behind

        tperf_start(recombo_perf);
        if (n_frames > 0) {

        for (i = 1; i < world_size; ++i) {
            int col;

            // you'd uncompress here
            if (fractal_params.flags & FRACTAL_FLAG_USE_COMPRESSION) {
                LZ4_decompress_safe((char *)prev_node_results[i], (char *)uncompressed_workloads[i], prev_node_results_len[i], 3 * fractal_params.width * fractal_params.height);
            } else {
                memcpy(uncompressed_workloads[i], prev_node_results[i], prev_node_results_len[i]);
            }

            log_trace("TRACE POINT b4 %d", i);


            for (j = 0; j < previous_node_workloads[i].assigned_cols_len; ++j) {
                col = previous_node_workloads[i].assigned_cols[j];

                // change from packed column major to full image row major
                int row_i, to_idx, from_idx;
                for (row_i = 0; row_i < fractal_params.height; ++row_i) {
                    from_idx = fractal_params.height * j + row_i;
                    to_idx = fractal_params.width * row_i + col;

                    //two methods of doing this
                    ((RGB_t *)total_image)[to_idx] = ((RGB_t *)(uncompressed_workloads[i]))[from_idx];
                   // total_image[3 * to_idx + 0] = uncompressed_workloads[i][3 * from_idx + 0];
                   // total_image[3 * to_idx + 1] = uncompressed_workloads[i][3 * from_idx + 1];
                   // total_image[3 * to_idx + 2] = uncompressed_workloads[i][3 * from_idx + 2];
                }
            }
        }
        
        }

        tperf_end(recombo_perf);
        diagnostics_history[diagnostics_history_idx].time_recombo = recombo_perf.elapsed_s;
        
        tperf_start(visuals_perf);

        log_trace("before visuals update");

        visuals_update(total_image);

        log_trace("after visuals update");

        tperf_end(visuals_perf);


        diagnostics_history[diagnostics_history_idx].time_visuals = visuals_perf.elapsed_s;


        tperf_start(control_update_perf);

        control_update_t control_update = control_update_loop();


        if (control_update.quit == true) {
            to_send = 0;
        }

        tperf_end(control_update_perf);
        diagnostics_history[diagnostics_history_idx].time_control_update = control_update_perf.elapsed_s;
        

        tperf_start(waiting_perf);

        MPI_Waitall(world_size-1, recv_requests + 1, recv_statuses + 1);
        MPI_Waitall(world_size-1, diagnostics_recv_requests + 1, MPI_STATUSES_IGNORE);
        MPI_Waitall(world_size-1, col_iters_recv_requests + 1, MPI_STATUSES_IGNORE);

        tperf_end(waiting_perf);
        diagnostics_history[diagnostics_history_idx].time_wait = waiting_perf.elapsed_s;
        

   

        for (i = 1; i < world_size; ++i) {

           // MPI_Wait(&recv_requests[i], &recv_statuses[i]);
           // MPI_Wait(&diagnostics_recv_requests[i], MPI_STATUS_IGNORE);
            
            diagnostics_history[diagnostics_history_idx].node_information[i].total_cols = node_workloads[i].assigned_cols_len;

            diagnostics_history[diagnostics_history_idx].node_information[i].temperature = recv_diagnostics[i][0];
            diagnostics_history[diagnostics_history_idx].node_information[i].time_compute = recv_diagnostics[i][1];
            diagnostics_history[diagnostics_history_idx].node_information[i].time_compress = recv_diagnostics[i][2];
            diagnostics_history[diagnostics_history_idx].node_information[i].time_total = recv_diagnostics[i][3];
            diagnostics_history[diagnostics_history_idx].node_information[i].time_io = recv_diagnostics[i][4];

            for (j = 0; j < node_workloads[i].assigned_cols_len; ++j) {
                diagnostics_history[diagnostics_history_idx].col_iters[node_workloads[i].assigned_cols[j]] = recv_col_iters[i][j];

            }

            int num_bytes = 0;
            MPI_Get_count(&recv_statuses[i], MPI_UNSIGNED_CHAR, &num_bytes);
            prev_node_results_len[i] = num_bytes;
            memcpy(prev_node_results[i], node_results_recv[i], num_bytes);
        }


     

        tperf_end(total_perf);
        diagnostics_history[diagnostics_history_idx].time_total = total_perf.elapsed_s;


        /* PRINT DIAGNOSTICS */
        /*
        int k;
        float min_time = INFINITY;
        float max_time = -INFINITY;
        for (k = 1; k < world_size; ++k) {
            if (recv_diagnostics[k][3] < min_time) {
                min_time = recv_diagnostics[k][3];
            }
            if (recv_diagnostics[k][3] > max_time) {
                max_time = recv_diagnostics[k][3];
            }
        }
        

        float differential = (max_time - min_time) / max_time;

        */
        //printf("%f, max differential: %%%f\n", max_time, 100.0 * differential);

        /*

        float longest_compute_time = 0.0f;
        int k;
        for (k = 1; k < world_size; ++k) {
            if (recv_diagnostics[k][1] > longest_compute_time) {
                longest_compute_time = recv_diagnostics[k][1];
            }
        }
        */


        //printf("FPS: %03.1f, diff: %%%03.1f\n", 1.0 / total_perf.elapsed_s, 100.0 * differential);
        //printf("visuals FPS: %.1f\n", 1.0 / visuals_perf.elapsed_s);

        //printf("longest compute %%%f\n", 100.0 *longest_compute_time/total_perf.elapsed_s);
        //printf("total %f\n", total_perf.elapsed_s);
        //printf("waiting %f\n", (waiting_perf.elapsed_s) / total_perf.elapsed_s);
        //printf("visual %%%f\n", 100.0 * visuals_perf.elapsed_s / total_perf.elapsed_s);
        //printf("assigning %%%f\n", 100.0 * assigning_perf.elapsed_s / total_perf.elapsed_s);
        //printf("recombo %%%f\n", 100.0 * recombo_perf.elapsed_s / total_perf.elapsed_s);
        //printf("control_update %%%f\n", 100.0 * control_update_perf.elapsed_s / total_perf.elapsed_s);


        // sync up optionally

        // update indexes
        diagnostics_history_idx = (diagnostics_history_idx + 1) % NUM_DIAGNOSTICS_SAVE;
        n_frames += 1;
        for (i = 1; i < world_size; ++i) {
            memcpy(previous_node_workloads[i].assigned_cols, node_workloads[i].assigned_cols, sizeof(int) * node_workloads[i].assigned_cols_len);
            previous_node_workloads[i].assigned_cols_len = node_workloads[i].assigned_cols_len;
        }
        
    }
    for (i = 0; i < world_size; ++i) {
        free(node_workloads[i].assigned_cols);
        free(prev_node_results[i]);
        free(recv_diagnostics[i]);
    }


    free(prev_node_results);
    free(node_workloads);
    free(recv_diagnostics);
    
    free(total_image);

    free(diagnostics_recv_requests);

    free(prev_node_results_len);

    free(send_requests);
    free(recv_requests);
    free(diagnostics_history);
}


void slave_loop() {

    bool keep_going = true;

    // status code
    int to_recv;

    // temperature (F, unused), time_compute, time_compress, time_total
    float * diagnostics = (float *)malloc(sizeof(float) * 5);

    workload_t my_workload;
    my_workload.assigned_cols = (int *)malloc(sizeof(int) * fractal_params.width);
    unsigned char * my_result = malloc(3 * fractal_params.width * fractal_params.height);
    // holds an array of iterations per column
    int * my_result_iters = (int *)malloc(sizeof(int) * fractal_params.width);
    unsigned char * my_compressed_buffer = malloc(LZ4_compressBound(3 * fractal_params.width * fractal_params.height));
    memset(my_result, 0, 3 * fractal_params.width * fractal_params.height);
    
    int my_result_size = 0;


    // initialize engine
    engine_c_init();

    #ifdef HAVE_CUDA
    engine_cuda_init(fractal_params, color_scheme.len, color_scheme.rgb_vals);
    #endif

    tperf_t compute_perf, compress_perf, total_perf, io_perf;

    tperf_init(compute_perf);
    tperf_init(compress_perf);
    tperf_init(total_perf);
    tperf_init(io_perf);
    

    while (keep_going) {


        // receive updates
        MPI_Bcast(&to_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // this means quit without error
        if (to_recv == 0) {
            M_EXIT(0);
        }

        tperf_start(total_perf);

        // receive any updates
        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);

        recv_workload(&my_workload);

        tperf_start(compute_perf);        

        #ifdef HAVE_CUDA
            log_debug("GPU run");
        if (this_node.type == NODE_TYPE_CPU) {
            engine_c_compute(my_workload, my_result, my_result_iters);
        } else if (this_node.type == NODE_TYPE_GPU) {
            engine_cuda_compute(my_workload, my_result, my_result_iters);
        }
        #else
        engine_c_compute(my_workload, my_result, my_result_iters);
        #endif

        tperf_end(compute_perf);

        my_result_size = 3 * fractal_params.height * my_workload.assigned_cols_len;

        if (fractal_params.flags & FRACTAL_FLAG_USE_COMPRESSION) {
            tperf_start(compress_perf);
            int compressed_size = LZ4_compress_default((char *)my_result, (char *)my_compressed_buffer, my_result_size, LZ4_compressBound(my_result_size));
            tperf_end(compress_perf);

            tperf_start(io_perf);
            MPI_Send(my_compressed_buffer, compressed_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            tperf_end(io_perf);
        } else {
            // so it is zero
            tperf_start(compress_perf);
            tperf_end(compress_perf);

            tperf_start(io_perf);
            MPI_Send(my_result, my_result_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            tperf_end(io_perf);
        }

        // send iterations per column
        MPI_Send(my_result_iters, my_workload.assigned_cols_len, MPI_INT, 0, 0, MPI_COMM_WORLD);
        tperf_end(total_perf);


        // temp, unused
        diagnostics[0] = -1.0f;
        // raw compute time
        diagnostics[1] = (float)compute_perf.elapsed_s;
        // compression, unused currently
        diagnostics[2] = (float)compress_perf.elapsed_s;

        
        // total loop time
        diagnostics[3] = (float)total_perf.elapsed_s;
        diagnostics[4] = (float)io_perf.elapsed_s;

        MPI_Send(diagnostics, 5, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);


    }

    free(my_result_iters);
    free(diagnostics);
    free(my_workload.assigned_cols);
    free(my_result);
    free(my_compressed_buffer);
}



