//




#include "mandelbrot.h"
#include "mandelbrot_render.h"
#include "mandelbrot_calc_c.h"

int world_size, world_rank;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int processor_name_len;

#define mpi_fr_numitems (6)
MPI_Datatype mpi_fr_t;
int mpi_fr_blocklengths[mpi_fr_numitems] = { 1, 1, 1, 1, 1, 1 };
MPI_Datatype mpi_fr_types[mpi_fr_numitems] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT };
MPI_Aint mpi_fr_offsets[mpi_fr_numitems];


#define M_EXIT(n) MPI_Finalize(); exit(0);

#define GETOPT_STOP ((char)-1)

#define IS_HEAD (world_rank == 0)
#define IS_COMPUTE (world_rank > 0)

#define compute_size (world_size - 1)
#define compute_rank (world_rank - 1)


void mandelbrot_show_help() {
    printf("Usage: mandelbrot [-h]\n");
    printf("  -h             show this help menu\n");
    printf("  -v[N]             show this help menu\n");
    printf("\n");
    printf("Questions? Issues? Please contact:\n");
    printf("<brownce@ornl.gov>\n");
}

// returns exit code, or -1 if we shouldn't exit
int parse_args(int argc, char ** argv) {
    char c;
    while ((c = getopt(argc, argv, "v:h")) != GETOPT_STOP) {
	switch (c) {
            case 'h':
		mandelbrot_show_help();
		return 0;
		break;
            case 'v':
                log_set_level(atoi(optarg));
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

    int res = -100;

    if (IS_HEAD) {
        res = parse_args(argc, argv);
    }

    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    mpi_fr_offsets[4] = offsetof(fr_t, max_iter);
    mpi_fr_offsets[5] = offsetof(fr_t, w);
    mpi_fr_offsets[6] = offsetof(fr_t, h);

    MPI_Type_create_struct(mpi_fr_numitems, mpi_fr_blocklengths, mpi_fr_offsets, mpi_fr_types, &mpi_fr_t);
    MPI_Type_commit(&mpi_fr_t);


    if (IS_HEAD) {

        fr.cX = .2821;
        fr.cY = .01;
        fr.Z = 1;
        fr.max_iter = 20;
        fr.w = 640;
        fr.h = 480;
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
    fr.w = 640;
    fr.h = 480;
    fr.cX = .2821;
    fr.cY = .01;
    fr.Z = 1.0;
    fr.max_iter = 10;

    fr_t fr_last;

    fr_last = fr;

    bool has_ran = false;

    unsigned char * pixels = NULL;

    while (true) {
        printf("calling recv...\n");
        MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
        //MPI_Bcast(&fr.Z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("recv zoom: %lf\n", fr.Z);
        if (pixels == NULL || !has_ran || fr_last.w != fr.w || fr_last.h != fr.h) {
            if (pixels != NULL) {
                free(pixels);
            }
            pixels = (unsigned char *)malloc(4 * fr.w * fr.h);
        }

        mand_c(fr.w, fr.h, fr.cX, fr.cY, fr.Z, fr.max_iter, pixels);
        MPI_Send(pixels, 4 * fr.w * fr.h, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);

        fr_last = fr;
        has_ran = true;
    }
}
