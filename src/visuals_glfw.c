/*

visuals library for the GLFW library

*/


#include "visuals_glfw.h"


//#define GL_GLEXT_PROTOTYPES

//#include <GL/glew.h>


//#include <OpenGL/GL.h>
//#include <OpenGL/glext.h>


#include "fractalexplorer.h"

#include "log.h"


// these are the required functions
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

// internal glfw callback
void error_callback(int error, const char* description) {
    log_error("Error[%d]: %s\n", error, description);
    exit(error);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
GLFWwindow * window;

GLuint texture = 0;
GLuint fbo = 0;

void visuals_glfw_init() {
                           

    // 4x antialiasing                                                  
 //   glfwWindowHint(GLFW_SAMPLES, 4);                                    


    // create texture


   /* glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    */


    // create FBO
/*
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                        GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
*/
    // We want OpenGL 3.3                                               
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);                      
  //  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);                               

    // We don't want the old OpenGL                                     
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);      
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);      

/*
    GLFWwindow* window;                                                 
    window = glfwCreateWindow(width, height, "Tutorial 01", NULL, NULL);

    if (window == NULL) {                                               
        printf("GLFW Failed to open a window. "                         
               "Intel GPUs don't support 3.3\n");                       
        glfwTerminate();                                                
    }                                                                   

    glfwMakeContextCurrent(window);                                     

    glewExperimental = 1;                                               
    if (glewInit() != GLEW_OK) {                                        
        printf("GLEW Failed to initialize.\n");                         
    }                                                                   

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);                

    do {                                                                
        glfwSwapBuffers(window);                                        
        glfwPollEvents();                                               

    } while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&      
    glfwWindowShouldClose(window) == 0 );   
*/


    if (glfwInit() != GLFW_TRUE) {                                                  
        printf("Glfw failed to init\n");     
        exit(1);                           
    }     


    window = glfwCreateWindow(fractal_params.width, fractal_params.height, "FractalExplorer", NULL, NULL);

    glfwMakeContextCurrent(window);

   // gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    if (!window) {
        visuals_glfw_finish();
        exit(1);
    }

    glfwSetErrorCallback(error_callback);


    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);



    log_info("OpenGL version: %s", glGetString(GL_VERSION));
    log_debug("OpenGL Extensions: %s", glGetString(GL_EXTENSIONS));

    // texture stuff

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);




    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fractal_params.width, fractal_params.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fractal_params.width, fractal_params.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    double ratio = (double)fractal_params.width / fractal_params.height;

/*
    glViewport(0, 0, fractal_params.width, fractal_params.height);
    glOrtho(0, fractal_params.width, 0, fractal_params.height, -1, 1);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

*/
    //GLuint Texture = loadBMP_custom("/Users/7cb/test/julia_2.bmp");

}

void visuals_glfw_update(unsigned char * fractal_pixels) {
    if (glfwWindowShouldClose(window)) {
        visuals_glfw_finish();
        return;
    }

    // update image
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fractal_params.width, fractal_params.height, GL_RGBA, GL_UNSIGNED_BYTE, fractal_pixels);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_TRIANGLE_STRIP);
    //glBegin(GL_QUADS);
        glTexCoord2f(0,0); glVertex2f(-1,-1);
        glTexCoord2f(1,0); glVertex2f(1,-1);
        glTexCoord2f(0,1); glVertex2f(-1, 1);
        glTexCoord2f(1,1); glVertex2f(1,1);
    glEnd();
    glfwSwapBuffers(window);
    //glfwPollEvents();
}

void visuals_glfw_finish() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

