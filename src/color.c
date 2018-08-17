/* color.c -- color library for transforming, interpolating, etc colors


*/

#include "fractalexplorer.h"

hq_color_t hq_color_rgb(uint8_t R, uint8_t G, uint8_t B) {
    hq_color_t r;
    r.R = R;
    r.G = G;
    r.B = B;
    return r;
}

// linear mixing
#define _LIN_MIX(X, Y, P) X * (1.0 - P) + Y * P
#define MIX(X, Y, P) (_LIN_MIX(((double)(X)), ((double)(Y)), ((double)(P))))

// when proportion=0.0, the result is 'a', when proportion=1.0, the result is 'b'.
// anything in between is scaled between the two
hq_color_t hq_color_mix(hq_color_t a, hq_color_t b, double proportion) {
    hq_color_t r;
    r.R = (uint8_t)MIX(a.R, b.R, proportion);
    r.G = (uint8_t)MIX(a.G, b.G, proportion);
    r.B = (uint8_t)MIX(a.B, b.B, proportion);
    return r;
}

hq_color_t hq_color_packed(palette_t palette, packed_color_t packed) {
    // returns the actual color using a palette
    if (packed.index > palette.num_colors) {
        return hq_color_rgb(0, 0, 0);
    } if (packed.prop == 0) {
        return palette.colors[packed.index];
    } else {
        int next_idx = (packed.index + 1) % palette.num_colors;
        return hq_color_mix(palette.colors[packed.index], palette.colors[next_idx], (double)packed.prop / 256.0);
    }
}

