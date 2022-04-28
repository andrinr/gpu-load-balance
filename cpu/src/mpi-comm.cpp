#include "mpi-comm.h"

void MPI_Comm::init() {
    MPI_Init(Null, Null);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype MPI_CELL;
    const int nitems=4;
    int  blocklengths[nitems] = {1, 1, DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, leftChildId);
    offsets[2] = offsetof(Cell, lower);
    offsets[3] = offsetof(Cell, upper);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);

    return MPI_CELL;
}

void MPI_Comm::prepareCut(int& nCells, int *cells) {
    MPI_Bcast(&nCells, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (nCells != -1) {
        MPI_Bcast(cells, nCells, MPI_CELL, 0, MPI_COMM_WORLD);
    }
}

void MPI_Comm::updateCut(int nCells, int *axis, float *pos) {
    MPI_Bcast(&axis, nCells, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&pos, nCells, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void MPI_Comm::reduceCut(int *count) {

}

void MPI_Comm::destroy() {
    MPI_Finalize();
}