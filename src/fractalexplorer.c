/* fractalexplorer.c -- the entry point for the program, basically figures out what to do
                     -- then uses other routines to get it done


*/


#include "fractalexplorer.h"

// MPI data
int world_size, world_rank;

fractal_t fractal;

diagnostics_t * diagnostics = NULL;
int n_frames = 0;

palette_t palette;

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // set defaults
    log_set_level(LOG_DEBUG);

    log_info("rank %d/%d", world_rank, world_size);

    diagnostics = malloc(sizeof(diagnostics_t) * NUM_DIAGNOSTICS);

    // set them to defaults
    int i;
    for (i = 0; i < NUM_DIAGNOSTICS; ++i) {
        diagnostics[i].total_time = 0.0;
        diagnostics[i].compute_time = 0.0;
        diagnostics[i].io_time = 0.0;
        diagnostics[i].format_time = 0.0;
        diagnostics[i].display_time = 0.0;
        diagnostics[i].bytes_transferred = 0;
    }


    fractal.center_r = 0.0;
    fractal.center_i = 0.0;
    fractal.zoom = 1.0;
    fractal.q_r = 0.0;
    fractal.q_i = 0.0;
    fractal.max_iter = 25;
    fractal.model = FRACTAL_MODEL_MANDELBROT;
    fractal.flags = FRACTAL_FLAG_NONE;

    if (world_rank == 0) {
        // master node
        visuals_init(1920, 1080);
        // TODO: read palette from file
        palette.num_colors = 2;
        palette.colors = malloc(sizeof(hq_color_t) * palette.num_colors);
        palette.colors[0] = hq_color_rgb(255, 0, 255);
        palette.colors[1] = hq_color_rgb(0, 255, 0);

        MPI_Bcast(&palette.num_colors, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&palette.colors, palette.num_colors * sizeof(hq_color_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        int status = 0;

        workload_t * workloads = malloc(sizeof(workload_t) * world_size);
        packed_color_t ** outputs = malloc(sizeof(packed_color_t *) * world_size);
        hq_color_t * full_output = malloc(sizeof(hq_color_t) * fractal.width * fractal.height);

        for (i = 0; i < world_size; ++i) {
            outputs[i] = NULL;
        }

        for (i = 0; i < fractal.width * fractal.height; ++i) {
            full_output[i] = hq_color_rgb(0, 0, 0);
        }

        double ltime = get_time();

        do {
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&fractal, sizeof(fractal_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            full_output = realloc(full_output, sizeof(hq_color_t) * fractal.width * fractal.height);

            // SYNC POINT

            MPI_Request _req;

            // now, todo: assign
            for (i = 1; i < world_size; ++i) {
                // loop through the workers
                workloads[i].start = fractal.height * (i - 1) / (world_size - 1);
                workloads[i].length = fractal.height / (world_size - 1);
                MPI_Isend(&workloads[i], 2, MPI_INT16_T, i, 0, MPI_COMM_WORLD, &_req);
            }

            for (i = 1; i < world_size; ++i) {
                outputs[i] = realloc(outputs[i], sizeof(packed_color_t) * workloads[i].length * fractal.width);
                MPI_Recv(outputs[i], sizeof(packed_color_t) * workloads[i].length * fractal.width, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            

            for (i = 1; i < world_size; ++i) {
                int py;
                for (py = workloads[i].start; py < workloads[i].start + workloads[i].length; ++py) {
                    int px;
                    for (px = 0; px < fractal.width; ++px) {
                        full_output[px + fractal.width * py] = hq_color_packed(palette, outputs[i][px + (py - workloads[i].start) * fractal.width]);
                    }
                }
            }

            log_debug("fps: %.1lf", 1.0 / (get_time() - ltime));
            ltime = get_time();

        } while (visuals_update(full_output));

        status = 1;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

        visuals_finish();

    } else {
        // slave node

        int status;
        MPI_Bcast(&palette.num_colors, 1, MPI_INT, 0, MPI_COMM_WORLD);
        palette.colors = malloc(palette.num_colors * sizeof(hq_color_t));
        MPI_Bcast(&palette.colors, palette.num_colors * sizeof(hq_color_t), MPI_BYTE, 0, MPI_COMM_WORLD);


        workload_t my_work;
        packed_color_t * my_output = NULL;

        do {
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (status == 0) {
                MPI_Bcast(&fractal, sizeof(fractal_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                // SYNC POINT
                // receive my workload
                MPI_Recv(&my_work, 2, MPI_INT16_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                my_output = realloc(my_output, sizeof(packed_color_t) * my_work.length * fractal.width);

                compute_C(my_work, my_output);

                MPI_Send(my_output, sizeof(packed_color_t) * my_work.length * fractal.height, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }

        } while (status == 0);
    }

    MPI_Finalize();
    
    return 0;
}

