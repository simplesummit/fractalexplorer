//



#include <mpi.h>
#include "mandelbrot.h"
#include "mandelbrot_render.h"
#include "mandelbrot_calc_c.h"
#include "mandelbrot_calc_cuda.h"
#include "color.h"

int world_size, world_rank;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int processor_name_len;

#define mpi_fr_numitems (6)
MPI_Datatype mpi_fr_t;
int mpi_fr_blocklengths[mpi_fr_numitems] = { 1, 1, 1, 1, 1, 1 };
MPI_Datatype mpi_fr_types[mpi_fr_numitems] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT };
MPI_Aint mpi_fr_offsets[mpi_fr_numitems];

fr_col_t col;


#define M_EXIT(n) MPI_Finalize(); exit(0);

#define GETOPT_STOP ((char)-1)

#define E_C  (0x101)
#define E_CUDA (0x102)

int engine = E_CUDA;

char * csch = "green";

void mandelbrot_show_help() {
    printf("Usage: mandelbrot [-h]\n");
    printf("  -h             show this help menu\n");
    printf("  -v[N]          set verbosity (1...5)\n");
    printf("  -N[N]          set width\n");
    printf("  -M[N]          set height\n");
    printf("  -i[N]          set iterations\n");
    printf("  -x[F]          set center x\n");
    printf("  -y[F]          set center y\n");
    printf("  -z[F]          set zoom\n");
    printf("  -c[S]          set scheme\n");
    printf("  -k[N]          set number of colors\n");
    printf("  -e[S]          set engine\n");
    printf("\n");
    printf("Questions? Issues? Please contact:\n");
    printf("<brownce@ornl.gov>\n");
}

// returns exit code, or -1 if we shouldn't exit
int parse_args(int argc, char ** argv) {
    char c;
    while ((c = getopt(argc, argv, "v:N:M:i:e:k:x:y:z:c:h")) != GETOPT_STOP) {
	switch (c) {
            case 'h':
		mandelbrot_show_help();
		return 0;
		break;
            case 'v':
                log_set_level(atoi(optarg));
                break;
            case 'N':
                fr.w = atoi(optarg);
                break;
            case 'M':
                fr.h = atoi(optarg);
                break;
            case 'i':
                fr.max_iter = atoi(optarg);
                break;
            case 'x':
                fr.cX = atof(optarg);
                break;
            case 'y':
                fr.cY = atof(optarg);
                break;
            case 'z':
                fr.Z = atof(optarg);
                break;
            case 'c':
                csch = optarg;
                break;
            case 'k':
                col.num = atoi(optarg);
                break;
            case 'e':
#define SEQ(a, b) (strcmp(a, b) == 0)
                if (SEQ(optarg, "c")) {
                    engine = E_C;
                } else if (SEQ(optarg, "cuda")) {
                    engine = E_CUDA;
                } else {
                    printf("Error: Unkown engine '%s'\n", optarg);
                    return 1;
                }
                break;
            case '?':
		printf("Unknown argument: -%c\n", optopt);
		return 1;
		break;
            default:
		printf("ERROR: unknown getopt return val\n");
		return 1;
                break;
	}
    }
    return -1;
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    MPI_Get_processor_name(processor_name, &processor_name_len);

    int res = -100, loglvl;

    fr.cX = .2821;
    fr.cY = .01;
    fr.Z = 1;
    fr.max_iter = 20;
    fr.w = 640;
    fr.h = 480;

    col.num = 20;

    if (IS_HEAD) {
        res = parse_args(argc, argv);
        loglvl = log_get_level();
    }

    MPI_Bcast(&engine, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loglvl, 1, MPI_INT, 0, MPI_COMM_WORLD);

    log_set_level(loglvl);

    if (res >= 0) {
        M_EXIT(res);
    }

    if (compute_size <= 0) {
        printf("ERROR: please run with more than 1 thread (need at least 1 compute node!)\n");
        M_EXIT(1);
    }

    if (IS_HEAD) {
        printf("head node, name: %s, %d/%d\n", processor_name, world_rank + 1, world_size);
    } else {
        printf("compute node, name: %s, %d/%d (compute %d/%d)\n", processor_name, world_rank + 1, world_size, compute_rank + 1, compute_size);
    }


    mpi_fr_offsets[0] = offsetof(fr_t, cX);
    mpi_fr_offsets[1] = offsetof(fr_t, cY);
    mpi_fr_offsets[2] = offsetof(fr_t, Z);
    mpi_fr_offsets[3] = offsetof(fr_t, max_iter);
    mpi_fr_offsets[4] = offsetof(fr_t, w);
    mpi_fr_offsets[5] = offsetof(fr_t, h);

    MPI_Type_create_struct(mpi_fr_numitems, mpi_fr_blocklengths, mpi_fr_offsets, mpi_fr_types, &mpi_fr_t);
    MPI_Type_commit(&mpi_fr_t);

    if (IS_HEAD) {
        col.col = (unsigned char *)malloc(4 * col.num);
        setcol(col, csch);
    }

    MPI_Bcast(&col.num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (IS_COMPUTE) {
        col.col = (unsigned char *)malloc(4 * col.num);
    }
    
    log_debug("setting chars, num: %d", col.num);
    MPI_Bcast(col.col, col.num * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    log_debug("after chars");


    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);

    if (IS_HEAD) {

        //fr.h_off = 0;

        mandelbrot_render(&argc, argv);
        /*
        int rowseach_compute = fr.h / compute_size;

        if (fr.h % compute_size != 0) {
            printf("fail compute size\n");
            exit(3);
        }

        fr_recombo_t fr_recombo;
        fr_recombo.num_workers = compute_size;
        fr_recombo.workers = (fr_t *)malloc(sizeof(fr_t) * fr_recombo.num_workers);
        fr_recombo.idata = (fr_wr_t *)malloc(sizeof(fr_wr_t) * fr_recombo.num_workers);

        int i;
        for (i = 0; i < fr_recombo.num_workers; ++i) {
            fr_recombo.workers[i] = fr;
            fr_recombo.workers[i]._data = (double *)malloc(sizeof(double) * fr.w * rowseach_compute);
            fr_recombo.idata[i]._data = (double *)malloc(sizeof(double) * fr.w * rowseach_compute);
        }

        fr_recombo._data = (double *)malloc(sizeof(double) * fr.w * fr.h);

        */

    } else {
        start_compute();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;

}

void start_render() {

    /*int sizeeach_compute = rowseach_compute * fr.w;
    fr_t * compute_nodes = (fr_t * )malloc(sizeof(fr_t) * compute_size);
    double ** compute_nodes_out = (double **)malloc(sizeof(double *) * compute_size);
    int i;
    for (i = 0; i < compute_size; ++i) {
        compute_nodes[i].cX = fr.cX;
        compute_nodes[i].cY = fr.cY;
        compute_nodes[i].Z = fr.Z;
        compute_nodes[i].max_iter = fr.max_iter;
        compute_nodes[i].h_off = i * rowseach_compute;
        compute_nodes_out[i] = (double *)malloc(sizeof(double) * sizeeach_compute);
    }

*/
}


void start_compute() {
    log_debug("starting compute");

    fr_t fr_last;

    fr_last = fr;

    bool has_ran = false;

    unsigned char * pixels = NULL;

    int my_h, my_off;

    tperf_t tp_sc, tp_ms;
    if (engine == E_C) {
        log_debug("engine C");
        mand_c_init();
    } else if (engine == E_CUDA) {
        log_debug("engine CUDA");
        mand_cuda_init(col);
    } else {
        log_error("Unknown engine");
    }

    while (true) {
        MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
        if (fr.h % compute_size != 0) {
            log_fatal("bad height and compute size");
            M_EXIT(2);
        }
        
        my_h = fr.h / compute_size;
        my_off = compute_rank * fr.h / compute_size;
        //MPI_Bcast(&fr.Z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (pixels == NULL || !has_ran || fr_last.w != fr.w || fr_last.h != fr.h) {
            if (pixels != NULL) {
                free(pixels);
            }
            pixels = (unsigned char *)malloc(4 * fr.w * my_h);
        }
        C_TIME(tp_sc,
            if (engine == E_C) {
                mand_c(fr.w, fr.h, my_h, my_off, fr.cX, fr.cY, fr.Z, fr.max_iter, pixels);
            } else if (engine == E_CUDA) {
                mand_cuda(fr, my_h, my_off, pixels);
            } else {
                log_error("Unknown engine");
            }
            // scan line
            scanline(pixels, fr.w, 0);
        )

        log_debug("computation fps: %lf", 1.0 / tp_sc.elapsed_s);
        
        C_TIME(tp_ms,
        log_trace("sending pixels back");
        MPI_Send(pixels, 4 * fr.w * my_h, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        )
        

        fr_last = fr;
        has_ran = true;
    }
}
