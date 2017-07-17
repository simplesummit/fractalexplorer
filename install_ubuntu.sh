#!/bin/bash

# script to install on jetsons


if [ "$EUID" -ne 0 ];
  then echo "Please run as root"
  exit 1
fi


CDIR="$PWD"
cd /tmp/

apt install libsdl2-dev libcr-dev mpich2 nfs-kernel-server nfs-common libsdl-ttf2.0-dev

curl -L "https://github.com/lz4/lz4/archive/v1.7.5.tar.gz" > lz4.tar.gz
tar xfv lz4.tar.gz
cd lz4-1.7.5
make
make install


