#include "visuals.h"
#include "fractalexplorer.h"

#include "commloop.h"
#include "log.h"
#include "SDL.h"


typedef struct RGB_t {
    unsigned char rgb[3];
} RGB_t;

typedef struct RGBA_t {
    unsigned char R, G, B, A;
} RGBA_t;


// main stuff
SDL_Window * window;
SDL_Renderer * renderer;

// where we store the fractal image
SDL_Texture * texture;

// graphs
int assign_col_graph_w, assign_col_graph_h;
SDL_Texture * assign_col_graph_texture;
unsigned char * assign_col_graph_texture_raw;


void visuals_init() {
    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO) != 0) {
        log_fatal("Fail on SDL_Init(): %s", SDL_GetError());
        M_EXIT(1);
    }

    atexit(SDL_Quit);

    int window_flags = 0;// SDL_WINDOW_FULLSCREEN_DESKTOP;

    window = SDL_CreateWindow("fractalexplorer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, fractal_params.width, fractal_params.height, window_flags);

    if (window == NULL) {
        log_fatal("Fail on SDL_CreateWindow(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_GetWindowSize(window, &fractal_params.width, &fractal_params.height);
//    SDL_ShowCursor(SDL_DISABLE);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    if (renderer == NULL) {
        log_fatal("Fail on SDL_CreateRenderer(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, fractal_params.width, fractal_params.height);
    if (texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }

    assign_col_graph_w = fractal_params.width;
    assign_col_graph_h = fractal_params.height / 4;

    assign_col_graph_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, assign_col_graph_w, assign_col_graph_h);
    if (texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_SetTextureBlendMode(assign_col_graph_texture, SDL_BLENDMODE_BLEND);

    assign_col_graph_texture_raw = malloc(4 * assign_col_graph_w * assign_col_graph_h);

}

// used for hsl conversion
float _color_helper_0(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}


RGBA_t get_nth_node_color(int n) {
    RGBA_t res;
    res.R = (43 * n) & 0xff;
    res.G = (57 * n + 111) & 0xff;
    res.B = (89 * n + 22) & 0xff;
    res.A = 255;
    return res;
}

void visuals_update(unsigned char * fractal_pixels) {
    // fractal_pixels contains column major order in RGB
    // needs to be converted into "texture" into row major RGB

    SDL_RenderClear(renderer);

    SDL_UpdateTexture(texture, NULL, fractal_pixels, 3 * fractal_params.width);
    SDL_RenderCopy(renderer, texture, NULL, NULL);


    // do graphs here

    if (n_frames > 0) {

        int last_diagnostics_idx = (diagnostics_history_idx - 1 + NUM_DIAGNOSTICS_SAVE) % NUM_DIAGNOSTICS_SAVE;
        diagnostics_t last_diagnostics = diagnostics_history[last_diagnostics_idx];


        int i, j, ct;
        int idx;

       // for (i = 0 ; i < NUM_DIAGNOSTICS_SAVE; ++i) {
        //    printf("%p\n", diagnostics_history[i].node_assignments);
        //}

        int total_iterations = 0;
        int max_iterations = 1;
        for (i = 0; i < assign_col_graph_w; ++i) {
            total_iterations += last_diagnostics.col_iters[i];
            if (last_diagnostics.col_iters[i] > max_iterations) {
                max_iterations = last_diagnostics.col_iters[i];
            }
        }


        for (i = 0; i < assign_col_graph_w; ++i) {
            RGBA_t col_color = get_nth_node_color(last_diagnostics.node_assignments[i]);

            // this makes the height based on proportion
            //int col_height =(int) (((float)assign_col_graph_h * assign_col_graph_h * last_diagnostics.col_iters[i]) / total_iterations);
            // this is based on max scaling
            int col_height =(int) (((float)assign_col_graph_h * last_diagnostics.col_iters[i]) / max_iterations);
            
            ct = 0;

            for (j = assign_col_graph_h; ct < col_height && ct < assign_col_graph_h; ++ct) {
                idx = i + j * assign_col_graph_w;
                ((RGBA_t *)assign_col_graph_texture_raw)[idx] = col_color;
                assign_col_graph_texture_raw[4 * idx + 3] = 140;
                j--;
            }
            for (; ct < assign_col_graph_h; ct++) {
                idx = i + j * assign_col_graph_w;

                assign_col_graph_texture_raw[4 * idx + 3] = 0;
                j--;
            }
        }


        SDL_UpdateTexture(assign_col_graph_texture, NULL, assign_col_graph_texture_raw, 4 * assign_col_graph_w);
    }


    // put it on the screen
    SDL_RenderCopy(renderer, texture, NULL, NULL);


    SDL_Rect assign_col_dst_rect;
    assign_col_dst_rect.x = 0;
    assign_col_dst_rect.y = fractal_params.height - assign_col_graph_h;
    assign_col_dst_rect.w = assign_col_graph_w;
    assign_col_dst_rect.h = assign_col_graph_h;
    
    SDL_RenderCopy(renderer, assign_col_graph_texture, NULL, &assign_col_dst_rect);


    
    SDL_RenderPresent(renderer);

}

void visuals_finish() {


    SDL_DestroyWindow(window);

    SDL_Quit();

}
