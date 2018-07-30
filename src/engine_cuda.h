
#ifndef __ENGINE_CUDA_H__
#define __ENGINE_CUDA_H__


#include "fr.h"

// give this so we know what to use
void engine_cuda_init(fractal_params_t _fr_p, int color_scheme_len, unsigned char * color_scheme);

void engine_cuda_compute(workload_t workload, unsigned char * output, int * output_iters);

// fast conversion boi
void engine_cuda_min_init();

void cuda_colmajor_to_rowmajor(RGB_t * input, RGB_t * output);

#endif


