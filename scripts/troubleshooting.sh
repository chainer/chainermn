#!/bin/sh

#=====================================================================
# This script check if the environment setup is an automated version
# of the troubleshooting documentation.
# http://chainermn.readthedocs.io/en/latest/installation/troubleshooting.html
#=====================================================================

MPIEXEC=$(which mpiexec)
if [ -z "${MPIEXEC}" ]; then
    echo "Error: mpiexec is not found: MPI is not installed?"
    exit -1
fi

mpiexec --version 2>&1 >/dev/null

MPICC=$(which mpicc)
if [ -z "${MPICC}" ]; then
    echo "Error: mpicc is not found: MPI is not installed?"
    exit -1
fi




