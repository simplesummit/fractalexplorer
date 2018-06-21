#include "visuals.h"
#include "fractalexplorer.h"

#include "commloop.h"
#include "log.h"
#include "SDL.h"
#include "FontCache/SDL_FontCache.h"


// main stuff
SDL_Window * window;
SDL_Renderer * renderer;

// where we store the fractal image
SDL_Texture * texture;

// graphs
int assign_col_graph_w, assign_col_graph_h;
SDL_Texture * assign_col_graph_texture;
unsigned char * assign_col_graph_texture_raw;

int info_graph_w, info_graph_h;
int info_graph_texture_xoff, info_graph_texture_yoff;
SDL_Texture * info_graph_texture;
unsigned char * info_graph_texture_raw;

#define NUM_INFO_GRAPH_MESSAGES 10
#define MAX_INFO_GRAPH_MESSAGE_LEN 100
char ** info_graph_messages = NULL;

int font_size;
FC_Font * font = NULL;


void visuals_init() {
    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO) != 0) {
        log_fatal("Fail on SDL_Init(): %s", SDL_GetError());
        M_EXIT(1);
    }

    atexit(SDL_Quit);

    int window_flags = SDL_WINDOW_FULLSCREEN_DESKTOP;

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

    // font caching

    font_size = fractal_params.width / 28;
    font = FC_CreateFont();
    FC_LoadFont(font, renderer, "./UbuntuMono.ttf", font_size, FC_MakeColor(0, 0, 0, 255), TTF_STYLE_NORMAL);


    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, fractal_params.width, fractal_params.height);
    if (texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }

    assign_col_graph_w = fractal_params.width;
    assign_col_graph_h = fractal_params.height / 4;

    assign_col_graph_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, assign_col_graph_w, assign_col_graph_h);
    if (assign_col_graph_texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }
    SDL_SetTextureBlendMode(assign_col_graph_texture, SDL_BLENDMODE_BLEND);
    assign_col_graph_texture_raw = malloc(4 * assign_col_graph_w * assign_col_graph_h);


    info_graph_w = 45 * fractal_params.width / 100;
    info_graph_h = fractal_params.width / 4;
    info_graph_texture_xoff = font_size / 2;
    info_graph_texture_yoff = font_size / 4;

    info_graph_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, info_graph_w, info_graph_h);
    if (info_graph_texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }
    SDL_SetTextureBlendMode(info_graph_texture, SDL_BLENDMODE_BLEND);
    info_graph_texture_raw = malloc(4 * info_graph_w * info_graph_h);

    int i, j;
    for (i = 0; i < info_graph_w; ++i) {
        for (j = 0; j < info_graph_h; ++j) {
            info_graph_texture_raw[4 * (info_graph_w * j + i) + 0] = 120;
            info_graph_texture_raw[4 * (info_graph_w * j + i) + 1] = 120;
            info_graph_texture_raw[4 * (info_graph_w * j + i) + 2] = 120;
            info_graph_texture_raw[4 * (info_graph_w * j + i) + 3] = 190;
        }
    }
    SDL_UpdateTexture(info_graph_texture, NULL, info_graph_texture_raw, 4 * info_graph_w);

    info_graph_messages = malloc(sizeof(char *) * NUM_INFO_GRAPH_MESSAGES);

    for (i = 0; i < NUM_INFO_GRAPH_MESSAGES; ++i) {
        info_graph_messages[i] = malloc(MAX_INFO_GRAPH_MESSAGE_LEN);
    }

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

            col_color.A = 170;

            // this makes the height based on proportion
            //int col_height =(int) (((float)assign_col_graph_h * assign_col_graph_h * last_diagnostics.col_iters[i]) / total_iterations);
            // this is based on max scaling
            int col_height =(int) (((float)assign_col_graph_h * last_diagnostics.col_iters[i]) / max_iterations);
            
            ct = 0;

            for (j = assign_col_graph_h; ct < col_height && ct < assign_col_graph_h; ++ct) {
                idx = i + j * assign_col_graph_w;
                ((RGBA_t *)assign_col_graph_texture_raw)[idx] = col_color;
                //assign_col_graph_texture_raw[4 * idx + 3] = 140;
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

    // text rendering
    // put it on the screen
    SDL_RenderCopy(renderer, texture, NULL, NULL);


    SDL_Rect assign_col_dst_rect;
    assign_col_dst_rect.x = 0;
    assign_col_dst_rect.y = fractal_params.height - assign_col_graph_h;
    assign_col_dst_rect.w = assign_col_graph_w;
    assign_col_dst_rect.h = assign_col_graph_h;
    
    SDL_RenderCopy(renderer, assign_col_graph_texture, NULL, &assign_col_dst_rect);


    SDL_Rect info_dst_rect;
    info_dst_rect.x = 0;
    info_dst_rect.y = 0;
    info_dst_rect.w = info_graph_w;
    info_dst_rect.h = info_graph_h;

    SDL_RenderCopy(renderer, info_graph_texture, NULL, &info_dst_rect);



    FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff, "Fractal Explorer");
    
    // text stuff

    if (n_frames > 0) {

        int last_diagnostics_idx = (diagnostics_history_idx - 1 + NUM_DIAGNOSTICS_SAVE) % NUM_DIAGNOSTICS_SAVE;
        diagnostics_t last_diagnostics = diagnostics_history[last_diagnostics_idx];

        /*int prec = (int)floor(log(fractal_params.zoom) / log(10) + 2.75);
        if (prec > 8) prec = 8;
        if (prec < 1) prec = 1;
        */

        int prec = 6;

        // put center, zoom, stuff
        sprintf(info_graph_messages[1], "At: %.*f%+.*fi", prec, fractal_params.center_r, prec,fractal_params.center_i);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 1 * font_size, info_graph_messages[1]);

        sprintf(info_graph_messages[2], "Zoom: %.2e", fractal_params.zoom);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 2 * font_size, info_graph_messages[2]);

        sprintf(info_graph_messages[3], "Nodes: %d", world_size - 1);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 3 * font_size, info_graph_messages[3]);
        
        // put FPS on screen
        sprintf(info_graph_messages[0], "FPS: %.1f", 1.0 / last_diagnostics.time_total);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 4 * font_size, info_graph_messages[0]);

        // calculate node difference maximum
        int k;
        float min_time = INFINITY;
        float max_time = -INFINITY;
        for (k = 1; k < world_size; ++k) {
            if (last_diagnostics.node_information[k].time_total < min_time) {
                min_time = last_diagnostics.node_information[k].time_total;
            }
            if (last_diagnostics.node_information[k].time_total > max_time) {
                max_time = last_diagnostics.node_information[k].time_total;
            }
        }
        
        float differential = (max_time - min_time) / max_time;


        sprintf(info_graph_messages[4], "Parallelism: %.1f%s", 100.0 * (1.0 - differential), "%%");
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 5 * font_size, info_graph_messages[4]);


    }


    SDL_RenderPresent(renderer);

}

void visuals_finish() {


    SDL_DestroyWindow(window);

    FC_FreeFont(font);

    SDL_Quit();

}
