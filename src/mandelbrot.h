//


#ifndef __MANDELBROT_H__
#define __MANDELBROT_H__

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <getopt.h>

#include <mpi.h>



typedef struct fr_t {
    double cX, cY, Z, er2;

    int max_iter;
} fr_t;

typedef struct mandelbrot_argp_t {

     bool show_help;

} mandelbrot_argp_t;



fr_t fr;

int main(int argc, char ** argv);


#endif




