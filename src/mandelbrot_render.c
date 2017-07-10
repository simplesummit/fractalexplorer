//

#include "mandelbrot.h"
#include "mandelbrot_render.h"
#include "mandelbrot_util.h"

#include <GL/gl.h>

#include <GL/glut.h>



unsigned int prog;

float last_x, last_y;

int last_bt;

void update_params() {
    set_uniform1f(prog, "zoom", fr.Z);
    set_uniform2f(prog, "center",  fr.cX, fr.cY);
    set_uniform1i(prog, "max_iter", fr.max_iter);
    
}

void window_refresh(int w, int h) {
    fr.w = w;
    fr.h = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    float aspect = (float)w / h;
    set_uniform1f(prog, "aspect", aspect);
    glOrtho(-1, 1, -1, 1, -1, 1);
    //gluOrtho2D(0.0, 800, 0.0, 600);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}


void mandelbrot_render(int * argc, char ** argv) {

    glutInit(argc, argv);

    glutInitWindowSize(fr.w, fr.h);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutCreateWindow("Mandelbrot Render");

    glEnable(GL_TEXTURE_2D);

    glutDisplayFunc(draw);
    glutIdleFunc(idle_handler);
    glutKeyboardFunc(key_handler);
    glutMouseFunc(mouse_handler);
    glutMotionFunc(motion_handler);
    glutReshapeFunc(window_refresh);

    if (!(prog = setup_shader("src/mandelbrot_shader.glsl"))) {
        printf("ERROR with shader\n");
    }

    //set_uniform1f(prog, "er2", fr.er2);


    window_refresh(fr.w, fr.h);
    glutMainLoop();
}

void draw() {
    update_params();


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

void motion_handler(int x, int y) {
    if (last_bt == GLUT_LEFT_BUTTON) {
        fr.Z *= 1 + 4 * (y - last_y) * (x - last_x) / (fr.w * fr.h);  
    } else if(last_bt == GLUT_RIGHT_BUTTON) {
        fr.cX = fr.cX - (x - last_x) / (fr.Z * fr.w);
        fr.cY = fr.cY - (y - last_y) / (fr.Z * fr.h);
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

