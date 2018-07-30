
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

#include <stddef.h>
#include "fr.h"


#define M_EXIT(code) MPI_Finalize(); exit(code);


int num_nodes;
node_t * nodes;

node_t this_node;

int world_size, world_rank;

#define compute_size ((world_size) - 1)
#define compute_rank ((world_rank) - 1)


int COMPRESS_LEVEL;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int processor_name_len;

char * font_path;

MPI_Datatype mpi_params_type;

fractal_params_t fractal_params;

int fractal_type_idx;
fractal_type_t * fractal_types;

color_scheme_t color_scheme;


int main(int argc, char ** argv);


#endif

