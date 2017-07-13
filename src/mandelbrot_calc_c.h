/* mandelbrot_calc_c.h -- defines C engine functions

  This file is part of the small-summit-fractal project.

  small-summit-fractal source code, as well as any other resources in this
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

#ifndef __MANDELBROT_CALC_C_H__
#define __MANDELBROT_CALC_C_H__

void mand_c_init();

// calculate image and store it in output
void mand_c(fr_t fr, int my_h, int my_off, unsigned char * output);

#endif
