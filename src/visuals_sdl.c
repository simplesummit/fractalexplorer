#include "visuals.h"
#include "fractalexplorer.h"

#include "commloop.h"
#include "log.h"
#include "SDL2/SDL.h"
#include "FontCache/SDL_FontCache.h"

//#ifndef SDL_PIXELFORMAT_RGBA32
///#define SDL_PIXELFORMAT_RGBA32 SDL_PIXELFORMAT_RGBA8888
//#endif


#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    #define PIXEL_FORMAT SDL_PIXELFORMAT_RGBA8888
#else
    #define PIXEL_FORMAT SDL_PIXELFORMAT_ABGR8888
#endif



// main stuff
SDL_Window * window;
SDL_Renderer * renderer;

// where we store the fractal image
SDL_Texture * texture;
unsigned char * texture_raw;

// graphs
int assign_col_graph_w, assign_col_graph_h;
SDL_Texture * assign_col_graph_texture;
unsigned char * assign_col_graph_texture_raw;

// legend
int assign_col_legend_w, assign_col_legend_h;
SDL_Texture * assign_col_legend_texture;
unsigned char * assign_col_legend_texture_raw;

int performance_graph_w, performance_graph_h;
SDL_Texture * performance_graph_texture;
unsigned char * performance_graph_texture_raw;

int info_graph_w, info_graph_h;
int info_graph_texture_xoff, info_graph_texture_yoff;
SDL_Texture * info_graph_texture;
unsigned char * info_graph_texture_raw;


// blend modes

SDL_BlendMode overlay_mode;


double _last_fps_val = 0.0;

#define NUM_INFO_GRAPH_MESSAGES 10
#define MAX_INFO_GRAPH_MESSAGE_LEN 100
char ** info_graph_messages = NULL;

int font_size;
FC_Font * font = NULL;


void visuals_sdl_init() {
    if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_VIDEO) != 0) {
        log_fatal("Fail on SDL_Init(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_JoystickEventState(SDL_ENABLE);

    SDL_version comp, link;
    
    SDL_VERSION(&comp);

    SDL_GetVersion(&link);

    log_info("compiled with SDL version %d.%d.%d", comp.major, comp.minor, comp.patch);
    log_info("linked currently with SDL version %d.%d.%d", link.major, link.minor, link.patch);


    atexit(SDL_Quit);

    int window_flags = 0;// SDL_WINDOW_FULLSCREEN_DESKTOP;

    if (fractal_params.width == 0 || fractal_params.height == 0) {
        window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }

    window = SDL_CreateWindow("fractalexplorer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, fractal_params.width, fractal_params.height, window_flags);

    if (window == NULL) {
        log_fatal("Fail on SDL_CreateWindow(): %s", SDL_GetError());
        M_EXIT(1);
    }

    SDL_GetWindowSize(window, &fractal_params.width, &fractal_params.height);
//    SDL_ShowCursor(SDL_DISABLE);

    // SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_TARGETTEXTURE
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    if (renderer == NULL) {
        log_fatal("Fail on SDL_CreateRenderer(): %s", SDL_GetError());
        M_EXIT(1);
    }

   //log_info("render mode: %d", renderer->info.texture_formats[0]);

    //overlay_mode = SDL_ComposeCustomBlendMode(SDL_BLENDFACTOR_SRC_ALPHA, SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, SDL_BLENDOPERATION_ADD, SDL_BLENDFACTOR_ZERO, SDL_BLENDFACTOR_ONE, SDL_BLENDOPERATION_ADD);

    //overlay_mode = SDL_BLENDMODE_BLEND;

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 120);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    //SDL_SetRenderDrawBlendMode(renderer, overlay_mode);

    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    // font caching

    font_size = 6 + fractal_params.width / 60;
    font = FC_CreateFont();
    FC_LoadFont(font, renderer, font_path, font_size, FC_MakeColor(0, 0, 0, 255), TTF_STYLE_NORMAL);


    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, fractal_params.width, fractal_params.height);
    if (texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }

    texture_raw = (unsigned char *)malloc(sizeof(RGB_t) * fractal_params.width * fractal_params.height);

    assign_col_graph_w = fractal_params.width;
    assign_col_graph_h = fractal_params.height / 5;

    assign_col_graph_texture = SDL_CreateTexture(renderer, PIXEL_FORMAT, SDL_TEXTUREACCESS_STREAMING, assign_col_graph_w, assign_col_graph_h);
    if (assign_col_graph_texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }
    SDL_SetTextureBlendMode(assign_col_graph_texture, SDL_BLENDMODE_BLEND);
    assign_col_graph_texture_raw = malloc(4 * assign_col_graph_w * assign_col_graph_h);


#define LEGEND_TEXT_SCALE 1.0
    assign_col_legend_w = (int)floor(font_size * 4.2 * LEGEND_TEXT_SCALE);
    assign_col_legend_h = (int)floor((world_size - 1) * font_size * LEGEND_TEXT_SCALE);

    assign_col_legend_texture = SDL_CreateTexture(renderer, PIXEL_FORMAT, SDL_TEXTUREACCESS_STREAMING, assign_col_legend_w, assign_col_legend_h);
    if (assign_col_legend_texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }
    SDL_SetTextureBlendMode(assign_col_legend_texture, SDL_BLENDMODE_BLEND);
    assign_col_legend_texture_raw = malloc(4 * assign_col_legend_w * assign_col_legend_h);


    info_graph_w = 25 * font_size / 2;
    info_graph_h = 31 * font_size / 4;
    info_graph_texture_xoff = font_size / 2;
    info_graph_texture_yoff = font_size / 4;

    info_graph_texture = SDL_CreateTexture(renderer, PIXEL_FORMAT, SDL_TEXTUREACCESS_STREAMING, info_graph_w, info_graph_h);
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

    // performance chart
    performance_graph_w = 12 * font_size;
    performance_graph_h = 27 * font_size / 4;

    performance_graph_texture = SDL_CreateTexture(renderer, PIXEL_FORMAT, SDL_TEXTUREACCESS_STREAMING, performance_graph_w, performance_graph_h);
    if (performance_graph_texture == NULL) {
        log_fatal("Fail on SDL_CreateTexture(): %s", SDL_GetError());
        M_EXIT(1);
    }
    SDL_SetTextureBlendMode(performance_graph_texture, SDL_BLENDMODE_BLEND);
    performance_graph_texture_raw = malloc(4 * performance_graph_w * performance_graph_h);


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

RGBA_t invert_color(RGBA_t a) {
    a.R = 255 - a.R;
    a.G = 255 - a.G;
    a.B = 255 - a.B;
    return a;
} 

RGBA_t get_complimentary_color(RGBA_t a) {
    a.R = (unsigned char)lin_mix(a.R, 255, 0.5);
    a.G = (unsigned char)lin_mix(a.G, 255, 0.5);
    a.B = (unsigned char)lin_mix(a.B, 255, 0.5);
    return a;
}


SDL_Color SCOLOR(RGBA_t col) {
    SDL_Color result = {col.R, col.G, col.B, col.A};
    return result;
}
void visuals_sdl_update(unsigned char * fractal_pixels) {
    // fractal_pixels contains column major order in RGB
    // needs to be converted into "texture" into row major RGB

    //SDL_RenderClear(renderer);

    // if this is true, just render fractal
    bool is_theatrical_mode = false;

  //  printf("%d\n", pitch/3);
   // memcpy(req_pix, fractal_pixels, pitch * fractal_params.height);


    SDL_UpdateTexture(texture, NULL, fractal_pixels, sizeof(RGB_t) * fractal_params.width);

    SDL_RenderCopy(renderer, texture, NULL, NULL);


    if (!is_theatrical_mode) {


    RGBA_t compute_color = {255, 0, 0, 255};
    RGBA_t io_color = {0, 255, 0, 255};
    RGBA_t compress_color = {0, 0, 255, 255};
    RGBA_t rest_color = {255, 0, 255, 255};


    RGBA_t unfill_color = {0, 0, 0, 0};
    RGBA_t barrier_color = {0, 0, 0, 255};

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


        int stripe_size = (int)floor(font_size * LEGEND_TEXT_SCALE / 2);


#pragma omp parallel for
        for (i = 0; i < assign_col_graph_w; ++i) {
            RGBA_t col_color = get_nth_node_color(last_diagnostics.node_assignments[i]);

            bool worker_changed = i != 0 && (last_diagnostics.node_assignments[i] != last_diagnostics.node_assignments[i - 1]);
            col_color.A = 175;
            RGBA_t comp_color = get_complimentary_color(col_color);

            // this makes the height based on proportion
            //int col_height =(int) (((float)assign_col_graph_h * assign_col_graph_h * last_diagnostics.col_iters[i]) / total_iterations);
            // this is based on max scaling
            int col_height =(int) (((float)assign_col_graph_h * last_diagnostics.col_iters[i]) / max_iterations);
            
            ct = 0;
            int k;

            for (k = 0; k < assign_col_graph_h - col_height; ++k) {
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = unfill_color;
            }

            if (k == 0) {
                // fill top 2
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * 0 + i] = barrier_color;
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * 1 + i] = barrier_color;
            } else if (k == assign_col_graph_h - 1) {
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = barrier_color;
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * (k-1) + i] = barrier_color;
            } else {
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = barrier_color;
                ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * (k + 1) + i] = barrier_color;
            }

            // barrier
            k += 2;


            while (k < assign_col_graph_h) {
                if (worker_changed) {
                    ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = barrier_color;                
                } else if (nodes[last_diagnostics.node_assignments[i]].type == NODE_TYPE_CPU) {
                    ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = col_color;
                } else {
                    if (((k + i) % (2 * stripe_size) + (2 * stripe_size)) % (2 * stripe_size) < 2 * stripe_size / 3) {
                        ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = comp_color;
                    } else {
                        ((RGBA_t *)assign_col_graph_texture_raw)[assign_col_graph_w * k + i] = col_color;
                    }
                }


                k++;
            }

        }

        SDL_UpdateTexture(assign_col_graph_texture, NULL, assign_col_graph_texture_raw, 4 * assign_col_graph_w);



        // put node names

        int w;
        for (w = 1; w < world_size; ++w) {
            RGBA_t cur_col = get_nth_node_color(w);
            RGBA_t comp_col = get_complimentary_color(cur_col);
            cur_col.A = 200;
            for (i = 0; i < assign_col_legend_w; ++i) {
                for (j = (int)floor((w-1) * font_size * LEGEND_TEXT_SCALE); j < (int)floor(w * font_size * LEGEND_TEXT_SCALE); ++j) {
                    if (i > assign_col_legend_w - font_size * LEGEND_TEXT_SCALE * 1.75) {
                        if (nodes[w].type == NODE_TYPE_CPU) {
                            ((RGBA_t *)assign_col_legend_texture_raw)[i + assign_col_legend_w * j] = cur_col;
                        } else {
                            if (((i + j) % (2 * stripe_size) + (2 * stripe_size)) % (2 * stripe_size) < 2 * stripe_size / 3) {
                                ((RGBA_t *)assign_col_legend_texture_raw)[i + assign_col_legend_w * j] = comp_col;
                            } else {
                                ((RGBA_t *)assign_col_legend_texture_raw)[i + assign_col_legend_w * j] = cur_col;
                            }
                        }
                    } else {
                        ((RGBA_t *)assign_col_legend_texture_raw)[i + assign_col_legend_w * j].A = 0;
                    }
                }
            }

        }

        SDL_UpdateTexture(assign_col_legend_texture, NULL, assign_col_legend_texture_raw, 4 * assign_col_legend_w);



        // 4 fps, longest time to show
        float biggest_time = 1.0 / 4.0;


//#pragma omp parallel for
        for (i = 0; i < performance_graph_w; ++i) {
            
            int diag_idx = ((diagnostics_history_idx + i - performance_graph_w) % NUM_DIAGNOSTICS_SAVE + NUM_DIAGNOSTICS_SAVE) % NUM_DIAGNOSTICS_SAVE;
            diagnostics_t cur_graph_diag = diagnostics_history[diag_idx];

            int k;
            float max_compute_time = -INFINITY;
            float max_compress_time = -INFINITY;
            float max_io_time = -INFINITY;
            for (k = 1; k < world_size; ++k) {
                if (cur_graph_diag.node_information[k].time_compute > max_compute_time) {
                    max_compute_time = cur_graph_diag.node_information[k].time_compute;
                }
                if (cur_graph_diag.node_information[k].time_compress > max_compress_time) {
                    max_compress_time = cur_graph_diag.node_information[k].time_compress;
                }                
                if (cur_graph_diag.node_information[k].time_io > max_io_time) {
                    max_io_time = cur_graph_diag.node_information[k].time_io;
                }
            }
            

            float compute_prop = max_compute_time / biggest_time;
            float io_prop = cur_graph_diag.time_recombo + max_io_time;// + cur_graph_diag.time_;
            /*if (cur_graph_diag.time_wait < 0.01) {
                // only count the recombo if it was waited on
                io_prop += cur_graph_diag.time_recombo;
            }*/

            io_prop /= biggest_time;
            float compress_prop = (max_compress_time + cur_graph_diag.time_decompress) / biggest_time;

            float total_prop = cur_graph_diag.time_total / biggest_time;

            float rest_prop = total_prop - (compute_prop + io_prop + compress_prop);


            float proportion_filled = 0.0f;
            int cur_section = 0;
            int cur_section_filled = 0;

            bool is_graph = false;

            for (k = 0; k < performance_graph_h; k++) {
                proportion_filled = (float)(k - font_size) / (performance_graph_h - font_size);

                is_graph = true;
                if (k < font_size) {
                    is_graph = false;
                    ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = barrier_color;
                } else if (proportion_filled < compute_prop || (cur_section == 0 && cur_section_filled < 2)){ // sec 0
                    ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = compute_color;
                } else if (proportion_filled - compute_prop < compress_prop || (cur_section == 1 && cur_section_filled < 2)) { //sec 1
                    if (cur_section != 1) {
                        cur_section = 1;
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = barrier_color;
                    } else {
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = compress_color;
                    }
                } else if (proportion_filled - compute_prop - compress_prop < io_prop || (cur_section == 2 && cur_section_filled < 2)) { //sec 2
                    if (cur_section != 2) {
                        cur_section = 2;
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = barrier_color;
                    } else {
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = io_color;
                    }

                } else if (proportion_filled < total_prop || (cur_section == 3 && cur_section_filled < 2)) { //sec 3
                    if (cur_section != 3) {
                        cur_section = 3;
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = barrier_color;
                        j++;
                    } else {
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = rest_color;
                    }
                } else { //sec 4
                    if (cur_section != 4) {
                        cur_section = 4;
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = barrier_color;
                        j++;
                    } else {
                        ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k] = unfill_color;
                        is_graph = false;
                    }
                }
                
                if (k >= font_size && is_graph && i < performance_graph_w / 4) {
                    float xprop = 4.0 * i / performance_graph_w;
                    float alpha = xprop - (1.0 - xprop) * pow(proportion_filled, 4) / 1.5;
                    if (alpha < 0.0) alpha = 0.0;
                    if (alpha > 1.0) alpha = 1.0;
                    ((RGBA_t *)performance_graph_texture_raw)[i + performance_graph_w * k].A = (unsigned char)floor(255 * alpha);
                }

                cur_section_filled++;
            }
        }

        SDL_UpdateTexture(performance_graph_texture, NULL, performance_graph_texture_raw, 4 * performance_graph_w);

    }

    SDL_Rect assign_col_dst_rect;
    assign_col_dst_rect.x = 0;
    assign_col_dst_rect.y = fractal_params.height - assign_col_graph_h;
    assign_col_dst_rect.w = assign_col_graph_w;
    assign_col_dst_rect.h = assign_col_graph_h;
    
    SDL_RenderCopy(renderer, assign_col_graph_texture, NULL, &assign_col_dst_rect);


    SDL_Rect assign_legend_dst_rect;
    assign_legend_dst_rect.x = fractal_params.width - assign_col_legend_w;
    assign_legend_dst_rect.y = fractal_params.height - assign_col_legend_h;
    assign_legend_dst_rect.w = assign_col_legend_w;
    assign_legend_dst_rect.h = assign_col_legend_h;
    
    SDL_RenderCopy(renderer, assign_col_legend_texture, NULL, &assign_legend_dst_rect);

    int w;
    int num_cpus = 0;
    int num_gpus = 0;
    for (w = 1; w < world_size; ++w) {

        SDL_Color text_color = { 0, 0, 0, 255 };

        if (nodes[w].type == NODE_TYPE_CPU) {
            num_cpus++;
            sprintf(info_graph_messages[1], "   G#%2d", w);
        } else {
            num_gpus++;
            sprintf(info_graph_messages[1], "   C#%2d", w);
        }


        FC_DrawScaleColor(font, renderer, assign_legend_dst_rect.x + font_size * LEGEND_TEXT_SCALE * (0.5 + 0.125), assign_legend_dst_rect.y + (w-1) * font_size * LEGEND_TEXT_SCALE, FC_MakeScale(LEGEND_TEXT_SCALE, LEGEND_TEXT_SCALE), text_color, info_graph_messages[1]);
    }



    SDL_Rect info_dst_rect;
    info_dst_rect.x = 0;
    info_dst_rect.y = 0;
    info_dst_rect.w = info_graph_w;
    info_dst_rect.h = info_graph_h;

    SDL_RenderCopy(renderer, info_graph_texture, NULL, &info_dst_rect);


    SDL_Rect performance_dst_rect;
    performance_dst_rect.x = fractal_params.width - performance_graph_w;
    performance_dst_rect.y = 0;
    performance_dst_rect.w = performance_graph_w;
    performance_dst_rect.h = performance_graph_h;

    SDL_RenderCopy(renderer, performance_graph_texture, NULL, &performance_dst_rect);



    float fscale = 1.0;
    FC_Scale fc_fscale = FC_MakeScale(fscale, fscale);


    sprintf(info_graph_messages[0], "compute");


    // messages onto performance graph
    FC_DrawScaleColor(font, renderer, performance_dst_rect.x + info_graph_texture_xoff / 2, performance_dst_rect.y - font_size / 10, fc_fscale, SCOLOR(compute_color), info_graph_messages[0]);



    sprintf(info_graph_messages[0], "compress");

    FC_DrawScaleColor(font, renderer, performance_dst_rect.x + info_graph_texture_xoff / 2 + 3.74 * fscale * font_size, performance_dst_rect.y - fscale * font_size / 9, fc_fscale, SCOLOR(compress_color), info_graph_messages[0]);

    sprintf(info_graph_messages[0], "io");

    FC_DrawScaleColor(font, renderer, performance_dst_rect.x + info_graph_texture_xoff / 2 + 7.9 * fscale * font_size, performance_dst_rect.y - fscale * font_size / 9, fc_fscale, SCOLOR(io_color), info_graph_messages[0]);

    sprintf(info_graph_messages[0], "misc");

    FC_DrawScaleColor(font, renderer, performance_dst_rect.x + info_graph_texture_xoff / 2 + 9.12 * fscale * font_size, performance_dst_rect.y - fscale * font_size / 9, fc_fscale, SCOLOR(rest_color), info_graph_messages[0]);



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
        sprintf(info_graph_messages[1], "At: %+.*f%+.*fi", prec, fractal_params.center_r, prec,fractal_params.center_i);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 1 * font_size, info_graph_messages[1]);

        // put center, zoom, stuff
        sprintf(info_graph_messages[1], "q : %+.*f%+.*fi", prec, fractal_params.q_r, prec,fractal_params.q_i);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 2 * font_size, info_graph_messages[1]);

        sprintf(info_graph_messages[2], "Zoom: %.2e", fractal_params.zoom);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 3 * font_size, info_graph_messages[2]);

        sprintf(info_graph_messages[3], "Eqn: %s, Iter: %d", fractal_types[fractal_type_idx].equation, fractal_params.max_iter);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 4 * font_size, info_graph_messages[3]);
        
        // put FPS on screen

        double cur_fps_val = 1.0 / last_diagnostics.time_total;

        // smooth it out:
        double RC = 1.0 / ((cur_fps_val / 16.0) * 2 * M_PI);
        double dt = last_diagnostics.time_total;
        double alpha = dt / (RC + dt);
        

        double lped = _last_fps_val * (1.0 - alpha) + alpha * cur_fps_val;

        _last_fps_val = lped;

        //if ((lped - cur_fps_val) > 3) log_debug("DIFF: %lf", lped - cur_fps_val);

        sprintf(info_graph_messages[0], "Nodes: %d, FPS: %.1f", world_size - 1, lped);
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 5 * font_size, info_graph_messages[0]);

        // calculate node difference maximum
        int k;
        float min_time = INFINITY;
        float max_time = -INFINITY;
        float avg_time = 0.0;
        for (k = 1; k < world_size; ++k) {
            if (last_diagnostics.node_information[k].time_total < min_time) {
                min_time = last_diagnostics.node_information[k].time_total;
            }
            if (last_diagnostics.node_information[k].time_total > max_time) {
                max_time = last_diagnostics.node_information[k].time_total;
            }
            avg_time += last_diagnostics.node_information[k].time_total;
        }
        avg_time /= (world_size - 1);
        float differential = min_time / avg_time;//(max_time - min_time) / max_time;


        sprintf(info_graph_messages[4], "Parallelism: %.1f%s", 100.0 * (1.0 - differential), "%%");
        FC_Draw(font, renderer, info_graph_texture_xoff, info_graph_texture_yoff + 6 * font_size, info_graph_messages[4]);


    }

    } // non theatrical mode



    SDL_RenderPresent(renderer);

}

void visuals_sdl_finish() {


    SDL_DestroyWindow(window);

    FC_FreeFont(font);

    SDL_Quit();

}
