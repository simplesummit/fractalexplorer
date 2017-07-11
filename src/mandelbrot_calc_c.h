//

#ifndef __MANDELBROT_CALC_C_H__
#define __MANDELBROT_CALC_C_H__

void mand_c_init();

// calculate image and store it in output
void mand_c(int w, int h, int my_h, int my_off, double cX, double cY, double Z, int max_iter, unsigned char * output);

#endif
