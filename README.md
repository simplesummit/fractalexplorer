# fractalexplorer

This is a repository for simulation and interactive program to be ran on leconte

## Requirements

Currently, this is only known to work on MacOS, Linux, either my personal desktop, or a Jetson TX2 (which is what leconte uses).

You need:

  * C compiler for MPI (`mpicc`)
  * MPI libraries
  * SDL2 and SDL2_ttf
  * LZ4 (a specific version is required, >= 1.7.0, see `install_ubuntu.sh` for a URL)

Optionally, the following are supported:

 * CUDA [default=yes]


## Building

To build, run:

`./configure && make`

to build without CUDA support, run:

`./configure --without-cuda`

The resulting binary is `./src/fractalexplorer`, and must be ran with `mpirun`

## Installing

Install scripts for required software are provided for macOS and ubuntu, they should be ran like: `./install_macos.sh` or `./install_ubuntu.sh` respectively.



## Running

To run, use `mpirun`:

`mpirun -n 1 ./src/fractalexplorer -h`

to view help.


Run `mpirun -n 6 ./src/fractalexplorer -i250 -crandom -F` to do a fullscreen render.

You will need at least 2 threads (1 head and 1 compute), but you can add as many as you'd like.

TODO: Add multi-machine example


## Bundling

To distribute source, run `make dist-gzip`, you should have `fractalexplorer-VERSION.tar.gz`
