//

#ifndef __MANDELBROT_UTIL_H__
#define __MANDELBROT_UTIL_H__

#include <GL/gl.h>

// time performance
typedef struct tperf_t {
    struct timeval stime, etime;

    double elapsed_s;

} tperf_t;


#define C_TIME(stor, ST) gettimeofday(&stor.stime, NULL); ST; gettimeofday(&stor.etime, NULL); stor.elapsed_s = (stor.etime.tv_sec - stor.stime.tv_sec) + (stor.etime.tv_usec - stor.stime.tv_usec) / 1000000.0;

GLhandleARB glCreateShaderObjectARB(GLenum);
void glShaderSourceARB(GLhandleARB, int, const char**, int*);

void glCompileShaderARB(GLhandleARB);
GLhandleARB glCreateProgramObjectARB(void);
void glAttachObjectARB(GLhandleARB, GLhandleARB);
void glLinkProgramARB(GLhandleARB);
void glUseProgramObjectARB(GLhandleARB);
void glGetInfoLogARB(GLhandleARB, GLsizei, GLsizei*, GLcharARB*);
void glGetObjectParameterivARB(GLhandleARB, GLenum, int*);
GLint glGetUniformLocationARB(GLhandleARB, const char*);
void glUniform1f(GLint location, GLfloat v0);
void glUniform2f(GLint location, GLfloat v0, GLfloat v1);

unsigned int setup_shader(const char *fname);

void set_uniform1f(unsigned int prog, const char *name, float val);
void set_uniform2f(unsigned int prog, const char *name, float v1, float v2);
void set_uniform1i(unsigned int prog, const char *name, int val);


#endif
