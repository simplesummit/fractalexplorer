#!/bin/bash

cd /tmp/

{ brew install pkg-config mpich sdl2 sdl2_ttf && \
curl -L "https://github.com/lz4/lz4/archive/v1.7.5.tar.gz" > lz4.tar.gz && \
tar xfv lz4.tar.gz && \
cd lz4-1.7.5 && \
make && \
sudo make install; } || 
{ echo "error while installing"; exit 1; }


