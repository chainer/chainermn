#!/bin/bash

# Prepare various MPI installations

BUILD_DIR=$HOME/mpi-build
mkdir -p ~/mpi
mkdir -p $BUILD_DIR

function install_mpich {
    MPI=$1

    if [ "$MPI" == "mpich-3.0.4" ]; then
        DIST="http://www.mpich.org/static/downloads/3.0.4/mpich-3.0.4.tar.gz"
    elif [ "$MPI" == "mpich-3.2" ]; then
        DIST="http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz"
    else
        echo "Unknown MPICH version: $MPI"
        exit -1
    fi

    PREFIX=$HOME/mpi/$MPI
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
    MPI=$1

    if [ "$MPI" == "openmpi-1.10.6" ]; then
        DIST="https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.6.tar.bz2"
    elif [ "$MPI" == "openmpi-1.8.8" ]; then
        DIST="https://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.8.tar.bz2"
    elif [ "$MPI" == "openmpi-1.6.5" ]; then
        DIST="https://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.gz"
    else
        echo "Unknown Open MPI version: $MPI"
        exit -1
    fi

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

if echo $MPI | grep -iq "openmpi"; then
    install_ompi $MPI
elif echo $MPI | grep -iq "mpich"; then
    install_mpich $MPI
fi
