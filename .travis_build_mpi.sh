#!/bin/bash

# Prepare various MPI installations

BUILD_DIR=$HOME/mpi-build
mkdir -p ~/mpi
mkdir -p $BUILD_DIR

function install_mpich {
    MPI=$1
    VER=$(echo "$MPI" | grep -Eo "[0-9.]+$")
    DIST="http://www.mpich.org/static/downloads/${VER}/mpich-${VER}.tar.gz"
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
                    --disable-fortran \
                    --enable-cxx \
                    --enable-shared=yes
        make -j4 && make install
    fi

    $PREFIX/bin/mpicxx -show
}

function install_ompi {
    MPI=$1
    VER=$(echo "$MPI" | grep -Eo "[0-9.]+$")
    MAJOR=$(echo "$VER" | grep -Eo "^[0-9]+\.[0-9]+")
    DIST="https://www.open-mpi.org/software/ompi/v${MAJOR}/downloads/openmpi-${VER}.tar.bz2"

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
