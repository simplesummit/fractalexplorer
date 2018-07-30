/*

C computation engine


*/

#ifndef __ENGINE_C_H__
#define __ENGINE_C_H__

#include "fr.h"

void engine_c_init();

void engine_c_compute(workload_t workload, unsigned char * output, int * output_iters);

void c_colmajor_to_rowmajor(RGB_t * input, RGB_t * output);

#endif
