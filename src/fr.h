

#ifndef __FR_H__
#define __FR_H__

typedef struct fr_t {
    double cX, cY, Z;

    int max_iter, w, h, mem_w;

    int num_workers;

} fr_t;


typedef struct fr_col_t {
    int num;
    // BGRA ordering
    unsigned char * col;
} fr_col_t;


#endif
