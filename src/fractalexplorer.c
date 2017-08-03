/* fractalexplorer.c -- executable main file

  This file is part of the fractalexplorer project.

  fractalexplorer source code, as well as any other resources in this
project are free software; you are free to redistribute it and/or modify them
under the terms of the GNU General Public License; either version 3 of the
license, or any later version.

  These programs are hopefully useful and reliable, but it is understood
that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
<cade@cade.site> for more info on this.

  Here is a copy of the GPL v3, which this software is licensed under. You
can also find a copy at http://www.gnu.org/licenses/.

*/


#include <mpi.h>
#include <stdlib.h>

#include "lz4.h"
#include "fr.h"
#include <stddef.h>
#include "fractalexplorer.h"
#include "render.h"
#include "calc_c.h"
#include "calc_cuda.h"
#include "color.h"

#define SEQ(a, b) (strcmp(a, b) == 0)

int world_size, world_rank;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int processor_name_len;

int fractal_types_idx = 0;
int fractal_types[FR_FRACTAL_NUM] = {
    FR_MANDELBROT, FR_MANDELBROT_3, FR_EXP, FR_SIN, FR_JULIA, FR_MULTIBROT

};

char * fractal_types_names[FR_FRACTAL_NUM] = {
    "Mandelbrot: z^2 + c",
    "Multibrot[3]: z^3 + c",
    "exp(z) + c",
    "sin(z) + c",
    "Julia: z^2 + q",
    "Multibrot: z^q + c"
};

int * gargc;

char *** gargv;


bool use_fullscreen = false;
bool show_cursor = true;


MPI_Datatype mpi_fr_t;
int mpi_fr_blocklengths[mpi_fr_numitems] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
MPI_Datatype mpi_fr_types[mpi_fr_numitems] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
MPI_Aint mpi_fr_offsets[mpi_fr_numitems];


fr_col_t col;



#define M_EXIT(n) MPI_Finalize(); exit(0);

#define GETOPT_STOP ((char)-1)

char * csch = "random";

char * fractal_path_file = "default.frpath";

void mandelbrot_show_help() {
    printf("Usage: " PACKAGE " [-h]\n");
    printf("  -h             show this help menu\n");
    printf("  -v[N]          set verbosity (1...5)\n");
    printf("  -N[N]          set width\n");
    printf("  -M[N]          set height\n");
    printf("  -E[F]          set color index offset\n");
    printf("  -G[F]          set color index multiplyer\n");
    printf("  -P[S]          set fractal path file name\n");
    printf("  -F             use fullscreen\n");
    printf("  -L             disable cursor\n");
    printf("  -i[N]          set iterations\n");
    printf("  -x[F]          set center x\n");
    printf("  -y[F]          set center y\n");
    printf("  -z[F]          set zoom\n");
    printf("  -c[S]          set scheme\n");
    printf("  -k[N]          set number of colors\n");
    printf("  -e[S]          set engine\n");
    printf("\n");
    printf(PACKAGE " version: " PACKAGE_VERSION "\n");
    printf("\n");
    printf("Questions? Issues? Please contact:\n");
    printf("<" PACKAGE_BUGREPORT ">\n");
}

// returns exit code, or -1 if we shouldn't exit
int parse_args(int argc, char ** argv) {
    char c;
    while ((c = getopt(argc, argv, "v:N:P:M:E:G:i:e:k:x:y:z:c:FLh")) != GETOPT_STOP) {
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
        case 'F':
            use_fullscreen = true;
            break;
        case 'L':
            show_cursor = false;
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
        case 'P':
            fractal_path_file = optarg;
            break;
        case 'c':
            csch = optarg;
            break;
        case 'E':
            fr.coffset = atof(optarg);
            break;
        case 'G':
            fr.cscale = atof(optarg);
            break;
        case 'k':
            col.num = atoi(optarg);
            break;
        case 'e':
            if (SEQ(optarg, "c")) {
                fr.engine = FR_E_C;
            } else if (SEQ(optarg, "cuda")) {
                fr.engine = FR_E_CUDA;
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
    gargc = &argc;
    gargv = &argv;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Get_processor_name(processor_name, &processor_name_len);

    int res = -100, loglvl;

    fr.cX = 0;
    fr.cY = 0.0;
    fr.Z = 0.5;
    fr.u = 0.0;
    fr.v = 0.0;
    fr.coffset = 0;
    fr.cscale = 1.0;
    fr.max_iter = 100;
    fr.w = 640;
    fr.h = 480;
    fr.engine = FR_E_C;

    // see fr.h for more types
    fr.fractal_type = FR_MANDELBROT;

    // see fr.h for more  flags
    fr.fractal_flags = FRF_NONE | FRF_SIMPLE;// | FRF_BINARYDECOMP_IMAG;// | FRF_BINARYDECOMP_IMAG;// | FRF_BINARYDECOMP_REAL;// | FRF_ADD_PERIOD;// | FRF_SIMPLE;

    col.num = 120;

    if (IS_HEAD) {
        res = parse_args(argc, argv);
        loglvl = log_get_level();
        fr.num_workers = compute_size;
    }

    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loglvl, 1, MPI_INT, 0, MPI_COMM_WORLD);

    log_set_level(loglvl);

    if (res >= 0) {
        M_EXIT(res);
    }

    if (compute_size <= 0) {
        printf("ERROR: please run with more than 1 thread (need at least 1 compute node!) (world size: %d, compute size: %d)\n", world_size, compute_size);
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
    mpi_fr_offsets[3] = offsetof(fr_t, u);
    mpi_fr_offsets[4] = offsetof(fr_t, v);
    mpi_fr_offsets[5] = offsetof(fr_t, coffset);
    mpi_fr_offsets[6] = offsetof(fr_t, cscale);
    mpi_fr_offsets[7] = offsetof(fr_t, max_iter);
    mpi_fr_offsets[8] = offsetof(fr_t, w);
    mpi_fr_offsets[9] = offsetof(fr_t, h);
    mpi_fr_offsets[10] = offsetof(fr_t, engine);
    mpi_fr_offsets[11] = offsetof(fr_t, fractal_type);
    mpi_fr_offsets[12] = offsetof(fr_t, fractal_flags);
    mpi_fr_offsets[13] = offsetof(fr_t, num_workers);

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
        atexit(end_render);
        start_render();
    } else {
        atexit(end_compute);
        start_compute();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;

}


void start_render() {
    fractalexplorer_render(gargc, *gargv);
}

void end_render() {
  log_info("render ending");
}



void start_compute() {
    log_debug("starting compute");

    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);

    bool has_ran = false;

#ifdef HAVE_CUDA
    bool have_cuda = true;
#else
    bool have_cuda = false;
#endif


    int max_compress_size = LZ4_compressBound(4 * fr.w * fr.h);
    int compress_size;

    unsigned char * compute_buffer = malloc(4 * fr.w * fr.h),
                  * compress_buffer = malloc(max_compress_size);

    tperf_t tp_compute, tp_compress, tp_send;
    /*
    log_debug("engine C");
    mand_c_init();

    #ifdef HAVE_CUDA
    log_debug("engine CUDA");
    mand_cuda_init(col);
    #else
    log_debug("wasn't compiled with CUDA support");
    #endif
*/

// macro to fail if we don't have cuda

#ifdef HAVE_CUDA
#define CUDA_EXEC log_trace("mand_cuda starting"); calc_cuda(fr, col, compute_rank, fr.num_workers, compute_buffer);
#else
#define CUDA_EXEC log_fatal("wasn't compiled with CUDA support"); M_EXIT(1);
#endif

    bool do_compress = true;

    memset(compute_buffer, 0, 4 * fr.w * fr.h);
    memset(compress_buffer, 0, max_compress_size);

    while (true) {
        MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
        /*
        if (fr.h % compute_size != 0) {
            log_fatal("bad height and compute size");
            M_EXIT(2);
        }
        */
        if (compute_rank < fr.num_workers) {

            // should probably memset this to clear it
            // memset(compute_buffer, 0, 4 * fr.w * fr.h);
            // memset(compress_buffer, 0, max_compress_size);


            C_TIME(tp_compute,
                if (fr.engine == FR_E_C) {
                    log_trace("mand_c starting");
                    calc_c(fr, compute_rank, fr.num_workers, compute_buffer);
                } else if (fr.engine == FR_E_CUDA) {
                    log_trace("mand_cuda starting");
                    CUDA_EXEC
                } else {
                    log_error("Unknown engine");
                }

                // scan line
                //log_trace("scanline");
                //scanline(pixels, fr.w, 0);
            )

            C_TIME(tp_compress,
                if (do_compress) {
                    compress_size = LZ4_compress_default((char *)compute_buffer, (char *)compress_buffer, 4 * fr.w * (fr.h / fr.num_workers + (compute_rank < fr.h % fr.num_workers)), max_compress_size);
                    if (compress_size <= 0) {
                        log_error("error in compression function: %d", compress_size);
                    }
                } else {
                    compress_size = -4 * fr.w * (fr.h / fr.num_workers + (compute_rank < fr.h % fr.num_workers));
                    memcpy(compress_buffer, compute_buffer, abs(compress_size));
                }
            )
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (compute_rank < fr.num_workers) {
            C_TIME(tp_send,
                MPI_Ssend(&compress_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Ssend(compress_buffer, abs(compress_size), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            )
        }

        log_debug("compute fps: %.2lf, compress fps: %.2lf", 1.0 / tp_compute.elapsed_s, 1.0 / tp_compress.elapsed_s);

        MPI_Barrier(MPI_COMM_WORLD);

    }
}
// when computing ends
void end_compute() {
    log_info("compute ending");
}
