/* fractalexplorer.h -- header for most files

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

#ifndef __FRACTALEXPLORER_H__
#define __FRACTALEXPLORER_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

#include <mpi.h>

#include <time.h>
#include <sys/time.h>
#include <getopt.h>

bool use_fullscreen;

#include "log.h"

#include "fr.h"

int fractal_types_idx;
int fractal_types[FR_FRACTAL_NUM];
char * fractal_types_names[FR_FRACTAL_NUM];

// time performance
typedef struct tperf_t {
    struct timeval stime, etime;

    double elapsed_s;

} tperf_t;


#define C_TIME(stor, ST) gettimeofday(&stor.stime, NULL); ST; gettimeofday(&stor.etime, NULL); stor.elapsed_s = (stor.etime.tv_sec - stor.stime.tv_sec) + (stor.etime.tv_usec - stor.stime.tv_usec) / 1000000.0;


// result from worker
typedef struct fr_wr_t {
    double * _data;

    // hash of cX, cY, Z, etc, so we know when to update
    int hash;
} fr_wr_t;

typedef struct fr_recombo_t {
    int num_workers;
    fr_t * workers;
    fr_wr_t * idata;

    double * _data;

} fr_recombo_t;


typedef struct mandelbrot_argp_t {

     bool show_help;

} mandelbrot_argp_t;


#define mpi_fr_numitems (14)

/*MPI_Datatype mpi_fr_t;
int mpi_fr_blocklengths[mpi_fr_numitems];
MPI_Datatype mpi_fr_types[mpi_fr_numitems];
MPI_Aint mpi_fr_offsets[mpi_fr_numitems];
*/

fr_t fr;
fr_col_t col;
MPI_Datatype mpi_fr_t;

char * fractal_path_file;

int world_size, world_rank;
#define IS_HEAD (world_rank == 0)
#define IS_COMPUTE (world_rank > 0)

#define compute_size (world_size - 1)
#define compute_rank (world_rank - 1)


int main(int argc, char ** argv);

void start_render();
void end_render();

void start_compute();
void end_compute();

#endif
