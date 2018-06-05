

#include "fractalexplorer.h"

#include "fr.h"
#include "log.h"
#include "commloop.c"
#include "visuals.h"

int num_nodes = -1;
node_t * nodes = NULL;

node_t this_node;


int world_size, world_rank;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int processor_name_len;


#define NUM_FRACTAL_PARAMS 10

MPI_Datatype mpi_params_type;

int mpi_params_blocklengths[NUM_FRACTAL_PARAMS] = { 
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
};

MPI_Datatype mpi_params_typearray[NUM_FRACTAL_PARAMS] = {
    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE
};

MPI_Aint mpi_params_offsets[NUM_FRACTAL_PARAMS];

fractal_params_t fractal_params;



int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Get_processor_name(processor_name, &processor_name_len);

    log_set_level(LOG_INFO);

    fractal_params.width = 640;
    fractal_params.height = 480;
    fractal_params.max_iter = 250;
    fractal_params.type = FRACTAL_TYPE_MANDELBROT;
    fractal_params.flags = FRACTAL_FLAG_USE_COMPRESSION;
    fractal_params.center_r = 0.2821;
    fractal_params.center_i = 0.01;
    fractal_params.q_r = 0.0;
    fractal_params.q_i = 0.0;
    fractal_params.zoom = 100;

    // parsing arguments

    // if it is -1000, let it pass
    int exit_code = -1000;

    if (world_rank == 0) {
        char c;

        while ((c = getopt(argc, argv, "v:h")) != (char)(-1)) {
            switch (c) {
            case 'h':
                printf("Usage: fractal explorer [-h] [-v VERBOSE]\n");
                printf("  -h                 help menu\n");
                printf("  -v [N]             set verbosity (1=error only, 5=trace)\n");
                printf("\n");
                exit_code = 0;
                break;
            case 'v':
                if (strlen(optarg) >= 1) {
                    int verbose_int = -1;
                    if (sscanf(optarg, "%d", &verbose_int) == 1) {
                        log_set_level(verbose_int);
                    } else {
                        printf("Warning: setting verbosity failed\n");
                    }
                }
                break;
            case '?':
                printf("Unknown argument: -%c\n", optopt);
                exit_code = 1;
                break;
            default:
                printf("ERROR: Unknown getopt val\n");
                exit_code = 2;
                break;
            }
        }



        /*

        starting communications here

        */

        /*

        send arguments that were parsed

        */

        int i, j;

        int verbose_send = log_get_level();

        for (i = 1; i < world_size; ++i) {
            MPI_Send(&exit_code, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&verbose_send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }       

        MPI_Barrier(MPI_COMM_WORLD);
        // all syncronized

        if (exit_code != -1000) {
            M_EXIT(exit_code);
        }    



        num_nodes = world_size;
        nodes = (node_t *)malloc(sizeof(node_t) * num_nodes);


        nodes[0].type = NODE_TYPE_MASTER;
        nodes[0].processor_name = strdup(processor_name);

        this_node = nodes[0];


        MPI_Status status;
        char * _tmp_recv_processor_name = malloc(MPI_MAX_PROCESSOR_NAME + 1);

        // used to see how many GPUs have been assigned to a processor
        int cur_proc_gpus_assigned = 0;

        // default
        int GPUs_per_proc = 1;


        // set up node relations
        for (i = 1; i < world_size; ++i) {

            // ask for node info
            MPI_Recv(_tmp_recv_processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);

            nodes[i].processor_name = strdup(_tmp_recv_processor_name);            


            cur_proc_gpus_assigned = 0;

            for (j = 1; j < i; ++j) {
                if (strcmp(nodes[j].processor_name, nodes[i].processor_name) == 0 && nodes[j].type == NODE_TYPE_GPU) {
                    cur_proc_gpus_assigned++;
                }
            }

            if (cur_proc_gpus_assigned < GPUs_per_proc) {
                nodes[i].type = NODE_TYPE_GPU;
            } else {
                nodes[i].type = NODE_TYPE_CPU;
            }

            // tell the node what type they are
            MPI_Send(&nodes[i].type, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        }

        // free the tmp, we are duplicating the things
        free(_tmp_recv_processor_name);

    } else {

        /* get arguments */

        int verbose_recv;
        // exit if not -1000
        int exit_code_recv;

        MPI_Recv(&exit_code_recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&verbose_recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        log_set_level(verbose_recv);

        MPI_Barrier(MPI_COMM_WORLD);

        if (exit_code_recv != -1000) {
            M_EXIT(exit_code_recv);        
        }

        /* respond to the above block */

        MPI_Send(processor_name, processor_name_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        this_node.processor_name = strdup(processor_name);

        // get what type this node has been assigned to
        MPI_Recv(&this_node.type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (this_node.type == NODE_TYPE_CPU) {
            log_debug("This node is CPU rendering");
        } else if (this_node.type == NODE_TYPE_GPU) {
            log_debug("This node is GPU rendering");
        } else {
            log_warn("Not CPU or GPU rendering, what is happening?");
        }

    }

    log_debug("MPI node %d/%d [machine: %s]", world_rank, world_size, processor_name);

    mpi_params_offsets[0] = offsetof(fractal_params_t, width);
    mpi_params_offsets[1] = offsetof(fractal_params_t, height);
    mpi_params_offsets[2] = offsetof(fractal_params_t, type);
    mpi_params_offsets[3] = offsetof(fractal_params_t, flags);
    mpi_params_offsets[4] = offsetof(fractal_params_t, max_iter);
    mpi_params_offsets[5] = offsetof(fractal_params_t, q_r);
    mpi_params_offsets[6] = offsetof(fractal_params_t, q_i);
    mpi_params_offsets[7] = offsetof(fractal_params_t, center_r);
    mpi_params_offsets[8] = offsetof(fractal_params_t, center_i);
    mpi_params_offsets[9] = offsetof(fractal_params_t, zoom);

    // MPI type
    MPI_Type_create_struct(NUM_FRACTAL_PARAMS, mpi_params_blocklengths, mpi_params_offsets, mpi_params_typearray, &mpi_params_type);
    MPI_Type_commit(&mpi_params_type);

    // broadcast it
    MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);

    // barriers so everything is defined
    MPI_Barrier(MPI_COMM_WORLD);

    /*  MAIN LOOP   */
    if (world_rank == 0) {
        visuals_init();
        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);
        master_loop();
        visuals_finish();
    } else {
        MPI_Bcast(&fractal_params, 1, mpi_params_type, 0, MPI_COMM_WORLD);
        slave_loop();
    }
    

    // ending
    MPI_Barrier(MPI_COMM_WORLD);
    
    M_EXIT(0);

    return 0;

}




