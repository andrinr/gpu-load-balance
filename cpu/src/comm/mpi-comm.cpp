#include "mpi-comm.h"

void MPI_Comm::MPI_Comm() {
    MPI_Init(Null, Null);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype MPI_CELL;
    const int nItems = 7;
    int blockLengths[nItems] = {1, 1, 1, 1, 1,  DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nItems] = {
            MPI_INT,
            MPI_INT,
            MPI_INT,
            MPI_INT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT
    };

    MPI_Aint offsets[nItems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, leftChildId);
    offsets[2] = offsetof(Cell, nCells);
    offsets[3] = offsetof(Cell, cutAxis);
    offsets[4] = offsetof(Cell, cutPos);
    offsets[5] = offsetof(Cell, lower);
    offsets[6] = offsetof(Cell, upper);

    MPI_Type_create_struct(nItems, blockLengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);
}

void MPI_COMM::dispatchWork(int &n, blitz::Array<Cell, 1> cells) {
    MPI_Bcast(&nCells.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (nCells != -1) {
        MPI_Bcast(cells.data(), nCells, MPI_CELL, 0, MPI_COMM_WORLD);
    }
}

OutData MPI_Comm::dispatchService(
        InData (*func)(OutData),
        InData inData,
        int nInData,
        OutData outData,
        int nOutData,
        int source) {

    MPI_Bcast(&nCells.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    OutData result func(inData);



}

void MPI_Comm::concludeWork(int &n, int *count) {
    //todo
    MPI_Reduce(&count, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void MPI_Comm::destroy() {
    MPI_Finalize();
}