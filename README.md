# fractalexplorer

This is a rewrite to fully meet our updated requirements.

See the [SimpleSummit project](https://simplesummit.github.io/) for information.


# Requirements

  * MPI (I prefer OpenMPI)
  * SDL2 (with TTF and Image extensions)
  * LZ4
  * (optional) CUDA
  * (optional) OpenCL
  
To install all the core dependencies on ubuntu, run:

`sudo apt install liblz4-dev libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev openmpi-bin libopenmpi-dev`

# Building

To build, clone this repository.

Once inside, run `mkdir build; cd build`.

Then, `cmake ..`, and then `make`

Now you should be able to run `./src/fractalexplorer`
