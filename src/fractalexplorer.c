

#include "fractalexplorer.h"

#include "fr.h"
#include "log.h"
#include "commloop.c"
#include "visuals.h"

int num_nodes = -1;
node_t * nodes = NULL;

node_t this_node;

// min 3, max 12
int COMPRESS_LEVEL = 3;

fractal_type_t * fractal_types;
int fractal_type_idx = 0;


#define ASSIGN_ALLCPU 1
#define ASSIGN_ALLGPU 2

#define ASSIGN_FIRSTGPU 3

#define ASSIGN_RATIO_1GPU_1CPU 4

#define ASSIGN_RATIO_1GPU_5CPU 5

#define ASSIGN_RATIO_2GPU_4CPU 6

int node_assign_pattern = ASSIGN_ALLCPU;

int world_size, world_rank;

char * font_path = NULL;

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
color_scheme_t color_scheme;


int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);

    // little srand for stuff to be consistent 
    srand(42);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Get_processor_name(processor_name, &processor_name_len);

    log_set_level(LOG_INFO);

    fractal_params.width = 640;
    fractal_params.height = 480;
    fractal_params.max_iter = 125;
    fractal_params.type = FRACTAL_TYPE_MANDELBROT;
    fractal_params.flags = 0 | FRACTAL_FLAG_USE_COMPRESSION | FRACTAL_FLAG_GRADIENT;
    fractal_params.center_r = 0.0;//0.2821;
    fractal_params.center_i = 0.0;//0.01;
    fractal_params.q_r = 0.0;
    fractal_params.q_i = 0.0;
    fractal_params.zoom = 1;//100

    font_path = "./UbuntuMono.ttf";

    fractal_types = malloc(sizeof(fractal_type_t) * NUM_FRACTAL_TYPES);

    fractal_types[0].flag = FRACTAL_TYPE_MANDELBROT;
    fractal_types[0].name = strdup("Mandelbrot");
    fractal_types[0].equation = strdup("z^2+c");

    fractal_types[1].flag = FRACTAL_TYPE_MULTIBROT;
    fractal_types[1].name = strdup("Multibrot");
    fractal_types[1].equation = strdup("z^q+c");

    fractal_types[2].flag = FRACTAL_TYPE_JULIA;
    fractal_types[2].name = strdup("Julia");
    fractal_types[2].equation = strdup("z^2+q");



    // parsing arguments

    // if it is -1000, let it pass
    int exit_code = -1000;

    if (world_rank == 0) {

        char c;

        char * color_scheme_path = NULL;

        log_info("world_size: %d", world_size);

        while ((c = getopt(argc, argv, "v:q:s:c:z:i:A:T:C:Fh")) != (char)(-1)) {
            switch (c) {
            case 'h':
                printf("Usage: fractal explorer [-h] [-v VERBOSE]\n");
                printf("  -h                 help menu\n");
                printf("  -v [N]             set verbosity (1=error only ... 5=trace)\n");
                printf("  -C [N]             compression level (3=min, 12=max)\n");
                printf("  -A [LABEL]         How to assign work (ALLCPU, ALLGPU)\n");
                printf("  -c [a+bi]          set the starting center point\n");
                printf("  -q [a+bi]          set q parameter start\n");
                printf("  -z [f]             set the starting zoom\n");
                printf("  -i [N]             set maximum iterations\n");
                printf("  -s [WxH]           set screen size (0x0 for full screen)\n");
                printf("  -T [font]          Font path\n");
                printf("  -F                 full screen display\n");
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
            case 'C':
                sscanf(optarg, "%d", &COMPRESS_LEVEL);
                break;
            case 'T':
                font_path = optarg;
                break;
            case 'A':
                if (strcmp(optarg, "ALLCPU") == 0) {
                    node_assign_pattern = ASSIGN_ALLCPU;
                } else if (strcmp(optarg, "ALLGPU") == 0) {
                    node_assign_pattern = ASSIGN_ALLGPU;
                } else if (strcmp(optarg, "FIRSTGPU") == 0) {
                    node_assign_pattern = ASSIGN_FIRSTGPU;
                } else if (strcmp(optarg, "RATIO_1GPU_1CPU") == 0) {
                    node_assign_pattern = ASSIGN_RATIO_1GPU_1CPU;
                } else if (strcmp(optarg, "RATIO_1GPU_5CPU") == 0) {
                    node_assign_pattern = ASSIGN_RATIO_1GPU_5CPU;
                } else if (strcmp(optarg, "RATIO_2GPU_4CPU") == 0) {
                    node_assign_pattern = ASSIGN_RATIO_2GPU_4CPU;
                } else {

                    log_error("unknown assign type: %s", optarg);
                    return 1;
                }
                break;
            case 'c':
                sscanf(optarg, "%lf%lfi", &fractal_params.center_r, &fractal_params.center_i);
                break;
            case 'z':
                sscanf(optarg, "%lf", &fractal_params.zoom);
                break;
            case 'q':
                sscanf(optarg, "%lf%lfi", &fractal_params.q_r, &fractal_params.q_i);
                break;
            case 's':
                sscanf(optarg, "%dx%d", &fractal_params.width, &fractal_params.height);
                break;
            case 'F':
                fractal_params.width = 0;
                fractal_params.height = 0;
                break;
            case 'i':
                fractal_params.max_iter = atoi(optarg);
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


        #ifdef HAVE_CUDA
        log_info("Compiled with CUDA support");

        #else

        if (node_assign_pattern == ASSIGN_ALLGPU) {
            log_warn("ALLGPU was used as assign pattern, but not compiled with CUDA support! falling back to C");
        }
        #endif


        if (color_scheme_path == NULL) {
            
            color_scheme.len = 12;
            color_scheme.rgb_vals = malloc(3 * color_scheme.len);

            int i;
            for (i = 0; i < color_scheme.len; ++i) {
                color_scheme.rgb_vals[3 * i + 0] = rand() & 0xff;
                color_scheme.rgb_vals[3 * i + 1] = rand() & 0xff;
                color_scheme.rgb_vals[3 * i + 2] = rand() & 0xff;
            }

        } else {
            // find the path and read the file

            free(color_scheme_path);
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
            MPI_Send(&COMPRESS_LEVEL, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
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



            if (false) { // TODO: 1 gpu per node
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
            } else if (node_assign_pattern == ASSIGN_ALLCPU) {
                nodes[i].type = NODE_TYPE_CPU;
            } else if (node_assign_pattern == ASSIGN_ALLGPU) {
                nodes[i].type = NODE_TYPE_GPU;
            } else if (node_assign_pattern == ASSIGN_FIRSTGPU) {
                if (i == 1) nodes[i].type = NODE_TYPE_GPU;
                else nodes[i].type = NODE_TYPE_CPU;
            } else if (node_assign_pattern == ASSIGN_RATIO_1GPU_1CPU) {
                if (i % 2 == 0) nodes[i].type = NODE_TYPE_GPU;
                else nodes[i].type = NODE_TYPE_CPU;
            } else if (node_assign_pattern == ASSIGN_RATIO_1GPU_5CPU) {
                if (i % 6 == 0) nodes[i].type = NODE_TYPE_GPU;
                else nodes[i].type = NODE_TYPE_CPU;
            } else if (node_assign_pattern == ASSIGN_RATIO_2GPU_4CPU) {
                if (i % 6 < 2) nodes[i].type = NODE_TYPE_GPU;
                else nodes[i].type = NODE_TYPE_CPU;
            }

            // tell the node what type they are
            MPI_Send(&nodes[i].type, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        }

        log_debug("starting center: %lf%+lfi", fractal_params.center_r, fractal_params.center_i);
        log_debug("starting zoom: %lf", fractal_params.zoom);
        log_debug("starting max_iter: %d", fractal_params.max_iter);
        log_debug("starting q: %lf%+lfi", fractal_params.q_r, fractal_params.q_i);
        log_debug("window size: %dx%d (0x0 means fullscreen)", fractal_params.width, fractal_params.height);


        // free the tmp, we are duplicating the things
        free(_tmp_recv_processor_name);

    } else {

        /* get arguments */

        int verbose_recv;
        // exit if not -1000
        int exit_code_recv;

        MPI_Recv(&exit_code_recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&COMPRESS_LEVEL, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    MPI_Bcast(&color_scheme.len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) color_scheme.rgb_vals = malloc(3 * color_scheme.len);
    MPI_Bcast(color_scheme.rgb_vals, 3 * color_scheme.len, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

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




