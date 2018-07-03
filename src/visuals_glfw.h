

#ifndef __VISUALS_GLFW_H__
#define __VISUALS_GLFW_H__

#include <GL/glew.h> 
#include <GLFW/glfw3.h>

GLFWwindow * window;

void visuals_glfw_init();

void visuals_glfw_update(unsigned char * fracrtal_pixels);

void visuals_glfw_finish();

#endif

