/* color.c -- common functions dealing with generating color

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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "fr.h"
#include "math.h"

// color schemes are: BGRA


// creates a scanline, a straight white line across the screen at h pixels

void scanline(unsigned char * out, int w, int h) {
    int i;
    for (i = 0; i < w; ++i) {
        out[4 * (i + h * w) + 0] = 255;
        out[4 * (i + h * w) + 1] = 255;
        out[4 * (i + h * w) + 2] = 255;
        out[4 * (i + h * w) + 3] = 255;
    }
}

// red color scheme (linear scale)
void setcol__red(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 0;
    col.col[ri + 1] = 0;
    col.col[ri + 2] = (int)floor(255 * v);
    col.col[ri + 3] = 255;
}

// green color scheme (linear scale)
void setcol__green(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 0;
    col.col[ri + 1] = (int)floor(255 * v);
    col.col[ri + 2] = 0;
    col.col[ri + 3] = 255;
}

// blue color scheme (linear scale)
void setcol__blue(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = (int)floor(255 * v);
    col.col[ri + 1] = 0;
    col.col[ri + 2] = 0;
    col.col[ri + 3] = 255;
}

// mocha color scheme
/*

ratio: x:x**2:x**3 creates a coffee like gradient

*/
void setcol__mocha(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = (int)floor(255 * v * v * v);
    col.col[ri + 1] = (int)floor(255 * v * v);
    col.col[ri + 2] = (int)floor(255 * v);
    col.col[ri + 3] = 255;
}


void setcol__simple (fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 255 * (i % 3 == 2);
    col.col[ri + 1] = 255 * (i % 3 == 1);
    col.col[ri + 2] = 255 * (i % 3 == 0);
    col.col[ri + 3] = 255;
}

// FLAG colors
void setcol__usa(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 255 * ((i % 3 == 2) || (i % 3 == 1));
    col.col[ri + 1] = 255 * (i % 3 == 1);
    col.col[ri + 2] = 255 * ((i % 3 == 0) || (i % 3 == 1));
    col.col[ri + 3] = 255;
}



// random color scheme. Each R,G,B value is a random value, but alpha is 255
void setcol__random(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = rand() & 0xff;
    col.col[ri + 1] = rand() & 0xff;
    col.col[ri + 2] = rand() & 0xff;
    col.col[ri + 3] = 255;
}

// sets the color to a string (print an error if not found). the memory in col
// needs to be allocated before calling this (4 * col.num bytes)
void setcol(fr_col_t col, char * scheme) {
    srand(time(NULL));

    // a function pointer to generate and store a color
    void (*cfnc)(fr_col_t, int, int, float);

    // string equals
    #define SEQ(a, b) (strcmp(a, b) == 0)
    if (SEQ(scheme, "red")) {
        cfnc = &setcol__red;
    } else if (SEQ(scheme, "green")) {
        cfnc = &setcol__green;
    } else if (SEQ(scheme, "blue")) {
        cfnc = &setcol__blue;
    } else if (SEQ(scheme, "mocha")) {
        cfnc = &setcol__mocha;
    } else if (SEQ(scheme, "simple")) {
        cfnc = &setcol__simple;
    } else if (SEQ(scheme, "usa")) {
        cfnc = &setcol__usa;
    } else if (SEQ(scheme, "random")) {
        cfnc = &setcol__random;
    } else {
        cfnc = NULL;
        //printf("error, unrecognized color scheme '%s'\n", scheme);
        //exit(1);

        FILE * fp = NULL;
        fp = fopen(scheme, "r");
        if (fp == NULL) {
            printf("Unknown color scheme '%s'\n", scheme);
            exit(1);
        }
        int i;
        char r, g, b, a;
        for (i = 0; i < col.num; ++i) {
            // send index and a float scaled value for convenience of calculation
            fscanf(fp, ",,,", );
            col.col[4 * i + 0];
        }
        return;
    }

    if (cfnc != NULL) {
        int i;
        for (i = 0; i < col.num; ++i) {
            // send index and a float scaled value for convenience of calculation
            (*cfnc)(col, 4 * i, i, (float)(i+1) / col.num);
        }
    }

}
