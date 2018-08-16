/* fractalexplorer.h -- main header file, defines types, includes libraries, etc



*/

#ifndef __FRACTALEXPLORER_H__
#define __FRACTALEXPLORER_H__

// configuration file generated from cmake
#include "fractalexplorerconfig.h"

// standard header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// include OpenCL header files
#ifdef HAVE_OpenCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif


// other header files within this project
#include "log.h"


// main method
int main(int argc, char ** argv);


#endif
