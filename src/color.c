//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "fr.h"
#include "math.h"

// BGRA


void scanline(unsigned char * out, int w, int h) {
    int i;
    for (i = 0; i < w; ++i) {
        out[4 * (i + h * w) + 0] = 255;
        out[4 * (i + h * w) + 1] = 255;
        out[4 * (i + h * w) + 2] = 255;
        out[4 * (i + h * w) + 3] = 255;
    }
}

void setcol__red(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 0;
    col.col[ri + 1] = 0;
    col.col[ri + 2] = (int)floor(255 * v);
    col.col[ri + 3] = 255;
}

void setcol__green(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = 0;
    col.col[ri + 1] = (int)floor(255 * v);
    col.col[ri + 2] = 0;
    col.col[ri + 3] = 255;
}
void setcol__blue(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = (int)floor(255 * v);
    col.col[ri + 1] = 0;
    col.col[ri + 2] = 0;
    col.col[ri + 3] = 255;
}
void setcol__mocha(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = (int)floor(255 * v * v * v);
    col.col[ri + 1] = (int)floor(255 * v * v);
    col.col[ri + 2] = (int)floor(255 * v);
    col.col[ri + 3] = 255;
}
void setcol__random(fr_col_t col, int ri, int i, float v) {
    col.col[ri + 0] = rand() & 0xff;
    col.col[ri + 1] = rand() & 0xff;
    col.col[ri + 2] = rand() & 0xff;
    col.col[ri + 3] = 255;
}


void setcol(fr_col_t col, char * scheme) {
    srand(time(NULL));

    //col.col = (unsigned char *)malloc(col.num * 4);

    void (*cfnc)(fr_col_t, int, int, float);

    #define SEQ(a, b) (strcmp(a, b) == 0)
    if (SEQ(scheme, "red")) {
        cfnc = &setcol__red;
    } else if (SEQ(scheme, "green")) {
        cfnc = &setcol__green;
    } else if (SEQ(scheme, "blue")) {
        cfnc = &setcol__blue;
    } else if (SEQ(scheme, "mocha")) {
        cfnc = &setcol__mocha;
    } else if (SEQ(scheme, "random")) {
        cfnc = &setcol__random;
    } else {
        cfnc = NULL;
        printf("error, unrecognized color scheme '%s'\n", scheme);
        exit(1);
        return;
    }


    int i;
    for (i = 0; i < col.num; ++i) {
        (*cfnc)(col, 4 * i, i, (float)(i+1) / col.num);
    }
    

}



