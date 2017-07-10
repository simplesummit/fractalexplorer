//

#include "mandelbrot.h"
#include "mandelbrot_render.h"
#include "mandelbrot_util.h"

#include <GL/gl.h>

#include <GL/glut.h>



unsigned int prog;

void mandelbrot_render(int * argc, char ** argv) {

    glutInit(argc, argv);

    glutInitWindowSize(800, 600);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutCreateWindow("Mandelbrot Render");

    glEnable(GL_TEXTURE_2D);

    glutDisplayFunc(draw);
    glutIdleFunc(idle_handler);
    glutKeyboardFunc(key_handler);
    //glutMouseFunc(bn_handler);
    //glutMotionFunc(mouse_handler);

    if (!(prog = setup_shader("src/mandelbrot_shader.glsl"))) {
        printf("ERROR with shader\n");
    }

    set_uniform2f(prog, "center",  fr.cX, fr.cY);
    set_uniform1f(prog, "zoom", fr.Z);
    //set_uniform1f(prog, "er2", fr.er2);
    set_uniform1i(prog, "max_iter", fr.max_iter);

    gluOrtho2D(-1,1,-1,1);
    glLoadIdentity();

    glutMainLoop();
}

void draw() {
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
