# small summit demo

This is a repository for simulation and interactive program to be ran on small summit

## Requirements

Currently, this is only known to work on Linux, either my personal desktop, or a Jetson TX2 (which small summit will use).

You need:

  * C compiler for MPI (`mpicc`)
  * MPI libraries
  * SDL2
  * LZ4 (a specific version is required, >= 1.7.0, see `install_jetsonTX2.sh` for a URL)

Optionally, the following are supported:

 * CUDA [default=yes]


## Building

To build, run:

`./configure && make`

(this should work on all Jetson TX2 machines)

to build without CUDA support, run:

`./configure --without-cuda`

The resulting binary is `./src/mandelbrot`

## Installing

For installing on a Small-Summit machine, run this on each machine (there are normally 8 jetsons):

`sudo ./install_jetsonTX2.sh`


## Running

To run, use `mpirun`:

`mpirun -n 1 ./src/mandelbrot -h`

to view help.


Run `mpirun -n 6 ./src/mandelbrot -i250 -crandom -F` to do a fullscreen render.

You will need at least 2 threads (1 head and 1 compute), but you can add as many as you'd like.

TODO: Add multi-machine example


## Bundling

To distribute source, run `make dist-gzip`, you should have `small-summit-mandelbrot-VERSION.tar.gz`






