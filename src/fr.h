

#ifndef __FR_H__
#define __FR_H__


// fractal_type enums

#define FR_MANDELBROT        (0x0101)
#define FR_MANDELBROT_3      (0x0102)
#define FR_SIN               (0x0103)

#define FR_TESTING           (0x0201)

#define __FR_FRACTAL_NUM       (4)

// leave of testing
#ifdef DEV
#define FR_FRACTAL_NUM __FR_FRACTAL_NUM
#else
#define FR_FRACTAL_NUM (__FR_FRACTAL_NUM - 1)
#endif

int fractal_types_idx;
int fractal_types[FR_FRACTAL_NUM];

typedef struct fr_t {
    double cX, cY, Z;
    
//    double coffset, cscale;

    int max_iter, w, h, mem_w;

    int fractal_type;

    int num_workers;

} fr_t;


typedef struct fr_col_t {
    int num;


    // BGRA ordering
    unsigned char * col;
} fr_col_t;


#endif


