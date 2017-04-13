#!/bin/bash

# Prepare various MPI installations

BUILD_DIR=$HOME/mpi-build
mkdir -p ~/mpi
mkdir -p $BUILD_DIR

function install_mpich {
    PREFIX=$HOME/mpi/$1
    DIST=$2
    FILE=$(basename $DIST)

    echo PREFIX=$PREFIX
    echo DIST=$DIST

    echo "-------------------------------------------------"
    echo "Installing $1"
    echo "-------------------------------------------------"

    if [ ! -x $PREFIX/bin/mpicxx ]; then
        rm -rf $BUILD_DIR/*
        cd $BUILD_DIR
        mkdir -p $PREFIX
        curl $DIST >$FILE
        tar -xf $FILE
        cd $(find . -maxdepth 1 -mindepth 1 -type d)
        ./configure --prefix=$PREFIX \
                    --disable-f77 \
                    --disable-fc \
                    --enable-cxx \
                    --enable-shared=yes
        make -j4 && make install
    fi

    $PREFIX/bin/mpicxx -show
}

function install_ompi {
    PREFIX=$HOME/mpi/$1
    DIST=$2
    FILE=$(basename $DIST)

    echo PREFIX=$PREFIX
    echo DIST=$DIST

    echo "-------------------------------------------------"
    echo "Installing $1"
    echo "-------------------------------------------------"

    if [ ! -x $PREFIX/bin/mpicxx ]; then
        rm -rf $BUILD_DIR/*
        cd $BUILD_DIR
        mkdir -p $PREFIX
        curl $DIST >$FILE
        tar -xf $FILE
        cd $(find . -maxdepth 1 -mindepth 1 -type d)
        ./configure --prefix=$PREFIX \
                    --disable-mpi-fortran \
                    --enable-mpi-cxx \
                    --enable-shared=yes
        make -j4 && make install
    fi

    $PREFIX/bin/mpicxx --showme:version
}

install_mpich mpich-3.0.4 "http://www.mpich.org/static/downloads/3.0.4/mpich-3.0.4.tar.gz"
#install_mpich mpich-3.2 "http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz"
install_ompi openmpi-1.10.6 "https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.6.tar.bz2"
#install_ompi openmpi-1.8.8 "https://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.8.tar.bz2"
# install ompi106 "https://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.gz"
