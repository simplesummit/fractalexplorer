//

#include "mandelbrot.h"
#include "mandelbrot_calc_c.h"
#include "mandelbrot_render.h"
#include "mandelbrot_util.h"


#include <math.h>

#include <GL/gl.h>

#include <GL/glut.h>


tperf_t tperf_render;

unsigned int prog, texture;

unsigned char * texels;

unsigned int hash;

float last_x, last_y;

int last_bt;

unsigned int hash_fr(fr_t fr) {
    return (int)floor( fr.Z + fr.w * (fr.h + fr.Z) - fr.cX - fr.cY + sin(fr.cX + fr.Z * fr.cY / fr.w) + fr.h);
}

void gather_picture() {
    tperf_t tp_bc, tp_rv;
    C_TIME(tp_bc,
    MPI_Bcast(&fr, 1, mpi_fr_t, 0, MPI_COMM_WORLD);
    )
    C_TIME(tp_rv,
    MPI_Recv(texels, 4 * fr.w * fr.h, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    )
    log_trace("MPI_Bcast(fr) fps: %lf\n", 1.0 / tp_bc.elapsed_s);
    log_trace("MPI_Recv(texels) fps: %lf\n", 1.0 / tp_rv.elapsed_s);
}


void window_refresh(int w, int h) {

    if (hash == hash_fr(fr)) {
        return;
    }

    if (w != fr.w || h != fr.h || texels == NULL) {
        log_debug("remallocing render texels\n");
        free(texels);
        texels = (unsigned char *)malloc(fr.w * fr.h * 4);
    }

    fr.w = w;
    fr.h = h;

    // GET PIXEL DATA HERE
    C_TIME(tperf_render,
    gather_picture();
    )
    log_debug("gather_picture() fps: %lf\n", 1.0 / tperf_render.elapsed_s);
    //mand_c(fr.w, fr.h, fr.cX, fr.cY, fr.Z, fr.max_iter, texels);

 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fr.w, fr.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);

}


void mandelbrot_render(int * argc, char ** argv) {

    glutInit(argc, argv);

    glutInitWindowSize(fr.w, fr.h);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glClearColor(0.0f, 0.5f, 0.0f, 1.0f);
    glutCreateWindow("Mandelbrot Render");


    glutDisplayFunc(draw);
    glutIdleFunc(idle_handler);
    glutKeyboardFunc(key_handler);
    glutMouseFunc(mouse_handler);
    glutMotionFunc(motion_handler);
    glutReshapeFunc(window_refresh);

    glGenTextures(1, &texture);
    
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glEnable(GL_TEXTURE_2D);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    

    texels = NULL;

    //window_refresh(fr.w, fr.h);
    glutMainLoop();
}

void draw() {
    C_TIME(tperf_render, 
    glDrawPixels(fr.w, fr.h, GL_RGBA, GL_UNSIGNED_BYTE, texels);

    glutSwapBuffers();
    );
    double fps = 1.0 / tperf_render.elapsed_s;
    if (fps < 15) {
        log_warn("draw() fps is low: %lf\n", fps);
    }
    log_trace("draw() fps: %lf\n", fps);
}

void idle_handler() {
    glutPostRedisplay();
}

void key_handler(unsigned char key, int x, int y) {
    printf("key '%c' pressed at %d,%d\n", key, x, y);
}

void motion_handler(int x, int y) {
    bool do_refresh = true;
    if (last_bt == GLUT_LEFT_BUTTON) {
        fr.Z *= 1 + 20 * (y - last_y) * (x - last_x) / (fr.w * fr.h);  
    } else if(last_bt == GLUT_RIGHT_BUTTON) {
        fr.cX = fr.cX - (x - last_x) / (fr.Z * fr.w);
        fr.cY = fr.cY - (y - last_y) / (fr.Z * fr.h);
    } else {
        do_refresh = false;
    }
    if (do_refresh) {
        window_refresh(fr.w, fr.h);
    }
    last_x = x; last_y = y;

}

void mouse_handler(int button, int state, int x, int y) {
    last_bt = button;
    last_x = x; last_y = y;
}

void reshape_handler(GLint w, GLint h) {
    window_refresh(w, h);

}

