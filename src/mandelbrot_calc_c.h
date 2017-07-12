//

#ifndef __MANDELBROT_CALC_C_H__
#define __MANDELBROT_CALC_C_H__

void mand_c_init();

// calculate image and store it in output
void mand_c(fr_t fr, int my_h, int my_off, unsigned char * output);

#endif
