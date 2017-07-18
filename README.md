# fractalexplorer

This is a repository for simulation and interactive program to be ran on leconte

## Requirements

Currently, this is only known to work on MacOS, Linux, either my personal desktop, or a Jetson TX2 (which is what leconte uses).

You need:

  * C compiler for MPI (`mpicc`) (use environment variable `MPICC` to set this)
  * MPI libraries
  * SDL2 and SDL2_ttf
  * LZ4 (you need a specific version, use [this tarball](https://github.com/lz4/lz4/archive/v1.7.5.tar.gz))

Optionally, the following are supported:

 * CUDA (use --with-cuda to specify)


## Building

### From Release Tarball

To build, run:

`./configure && make`

to build without CUDA support, run:

`./configure --without-cuda`

The resulting binary is `./src/fractalexplorer`, and must be ran with `mpirun`

### From Repository

First, install GNU Autotools for your platform. Then, run `autoreconf -i`

Now, you can run `./configure && make` just like **From Release Tarball**.


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

To distribute source, run `make dist-gzip` (or `make dist-zip`), you should have `fractalexplorer-VERSION.tar.gz` (or `fractalexplorer-VERSION.zip`).

This is what is included on the [releases](http://github.com/ornl-leconte/fractalexplorer/releases) page.


It includes `configure`, so you don't need autotools to install the program once untarred/unzipped; just use `./configure && make`


## FAQS/Common Problems

Q: During compilation, the compiler complains that `LZ4_compress_default` is undefined/unresolved. How to fix?

A: You either haven't installed LZ4, or have an improper version (even Ubuntu repositories carry old versions). Install using [this tarball](https://github.com/lz4/lz4/archive/v1.7.5.tar.gz).



Q: I'm getting an error that wasn't listed in this section, how do I troubleshoot?

A: Use fractalexplorer's verbose setting: add `-v5` to the argument list when running, and much more info will be printed. Paste the entire program log with `-v5` to [hastebin](https://hastebin.com/) or a similar site, and open up an [issue](https://github.com/ornl-leconte/fractalexplorer/issues/new), describe your problem and link the program output.
