//

#include "mandelbrot.h"
#include "mandelbrot_render.h"
#include "mandelbrot_util.h"

#include <GL/glut.h>
#include <GL/gl.h>



unsigned int prog;

void mandelbrot_render(int * argc, char ** argv) {
        
    glutInit(argc, argv);

    glutInitWindowSize(800, 600);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("Mandelbrot Render");
    
    glutDisplayFunc(draw);
    glutIdleFunc(idle_handler);
    glutKeyboardFunc(key_handler);
    //glutMouseFunc(bn_handler);
    //glutMotionFunc(mouse_handler);
    
    glBindTexture(GL_TEXTURE_1D, 1);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    if (!(prog = setup_shader("src/mandelbrot_shader.glsl"))) {
        printf("ERROR with shader\n");
    }

    glutMainLoop();
}

void draw() {
    set_uniform2f(prog, "center", fr.cX, fr.cY);
    set_uniform1f(prog, "zoom", fr.Z);
    set_uniform1f(prog, "er2", fr.er2);
    set_uniform1i(prog, "max_iter", fr.max_iter);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    glutSwapBuffers();
}

void idle_handler() {
    glutPostRedisplay();
}

void key_handler(unsigned char key, int x, int y) {
    printf("key '%c' pressed at %d,%d\n", key, x, y);
}


