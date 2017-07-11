//

#ifndef __MAND_CUDA_H__
#define __MAND_CUDA_H__

void mand_cuda_init(fr_col_t col);

void mand_cuda(fr_t fr, int my_h, int my_off, unsigned char * output);


#endif

