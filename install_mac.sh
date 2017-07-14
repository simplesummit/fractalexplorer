#!/bin/bash


if [ "$EUID" -ne 0 ];
  then echo "Please run as root"
  exit 1
fi

brew install SDL2 pkg-config mpich2 sdl2_ttf || exit 1

curl -L "https://github.com/lz4/lz4/archive/v1.7.5.tar.gz" > lz4.tar.gz
tar xfv lz4.tar.gz
cd lz4-1.7.5
make || exit 1
make install || exit 1
