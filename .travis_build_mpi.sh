#!/bin/bash

# Prepare various MPI installations

BUILD_DIR=$HOME/mpi-build
mkdir -p ~/mpi
mkdir -p $BUILD_DIR

function install {
    PREFIX=$HOME/mpi/$1
    DIST=$2
    FILE=$(basename $DIST)

    echo $PREFIX
    echo $DIST

    if [ ! -d $PREFIX ]; then
        rm -rf $BUILD_DIR/*
        cd $BUILD_DIR
        mkdir -p $PREFIX
        curl $DIST >$FILE
        tar -xf $FILE
        cd $(find . -maxdepth 1 -mindepth 1 -type d)
        ./configure --prefix=$PREFIX && make -j4 && make install
    fi
}

install mpich30 "http://www.mpich.org/static/downloads/3.0.4/mpich-3.0.4.tar.gz"
install mpich32 "http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz"
install ompi110 "https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.6.tar.bz2"
install ompi108 "https://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.8.tar.bz2"
install ompi106 "https://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.gz"
