/*

visuals library for the GLFW library

*/


#include "visuals_glfw.h"




//#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include <OpenGL/GL.h>
#include <OpenGL/glext.h>


#include "fractalexplorer.h"

#include "log.h"

// these are the required functions


// internal glfw callback
void error_callback(int error, const char* description) {
    log_error("Error[%d]: %s\n", error, description);
    exit(error);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


// data to be used
GLFWwindow* window = NULL;


GLuint framebuf_name = 0;

GLuint render_texture = 0;


void visuals_glfw_init() {

    if (!glfwInit()) {
        log_error("GLFW could't initialize!");
        exit(1);
    }
    glfwSetErrorCallback(error_callback);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    /*
    glewExperimental = true;
    printf("before init\n");
    int res = glewInit();
    printf("after init\n");
    if (res != GLEW_OK) {
        log_error("GLEW couldn't initialize!");
        exit(1);
    }

    */

    window = glfwCreateWindow(fractal_params.width, fractal_params.height, "My Title", NULL /* glfwGetPrimaryMonitor() */, NULL);
    if (!window) {
        log_error("Error creating window!");
        exit(1);
    }

    glfwMakeContextCurrent(window);
    //glfwSwapInterval(1);



    glGenFramebuffers(1, &framebuf_name);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuf_name);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_UNSIGNED_BYTE, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

float vertices[] = {
        // positions          // colors           // texture coords
        0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
        0.5f, -0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   0.0f, 0.0f,   // bottom left
        -0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   0.0f, 1.0f    // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    float vertices2[] = {
        // positions          // colors           // texture coords
        0.75f,  0.75f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 1.0f,   // top right
        0.75f, 0.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        0.0f, 0.0f, 0.0f,  0.0f,1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
        0.0f,  0.75f, 0.0f,  0.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left 
    };
    unsigned int indices2[] = {
        0, 1, 2, // first triangle
        0, 2, 3  // second triangle
    };

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int VBO[2],VAO[2],EBO[2];

    glGenVertexArrays(2, VAO);
    glGenBuffers(2, VBO);
    glGenBuffers(2, EBO);

glBindVertexArray(VAO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    

    glBindVertexArray(VAO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices2), vertices2, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices2), indices2, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    while (!glfwWindowShouldClose(window)){

    }



    glGenTextures(1, &render_texture);
    glBindTexture(GL_TEXTURE_2D, render_texture);



    // create texture
    glTexImage2D(GL_TEXTURE_2D, 0, 3, fractal_params.width, fractal_params.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_texture, 0);

    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        log_error("incomplete frame buffer");
        exit(1);
    }

}

void visuals_glfw_update(unsigned char * fractal_pixels) {
    if (glfwWindowShouldClose(window)) {
        visuals_glfw_finish();
        return;
    }

    // THIS NEEDS TO BE UPDATED FOR EFFICIENCY REASONS

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

   // glBindFramebuffer(GL_FRAMEBUFFER, framebuf_name);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fractal_params.width, fractal_params.height, GL_RGB, GL_UNSIGNED_BYTE, fractal_pixels);

   // glBindTexture(GL_TEXTURE_2D, render_texture);
    glBegin(GL_QUADS);
    
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);

    glTexCoord2f(1, 0);
    glVertex2f(width, 0);

    glTexCoord2f(1, 1);
    glVertex2f(width, height);

    glTexCoord2f(0, 1);
    glVertex2f(0, height);
    glEnd();
    glPopMatrix();



    glViewport(0, 0, width, height);
    
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 3, fractal_params.width, fractal_params.height, 0, GL_RGB, GL_UNSIGNED_BYTE, fractal_pixels);


    glfwSwapBuffers(window);
    glfwPollEvents();
}

void visuals_glfw_finish() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

