/* calc_cuda.h -- exposes CUDA engine to other files. Specifically, calc_cuda.cu
               -- must be compiled by NVCC and then linked. This header file is
               -- neccesary to define unresolved symbols before the final link
               -- step

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

#ifndef __CALC_CUDA_H__
#define __CALC_CUDA_H__

void calc_cuda_init(fr_col_t col);

void calc_cuda(fr_t fr, fr_col_t fr_col, int my_h, int my_off, unsigned char * output);


#endif
