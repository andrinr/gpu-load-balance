#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <blitz/array.h>   

struct Cell {
    int id;
    int leftChildId;
    float lower[3], upper[3];
    Cell(
        int id,
        int leftChildId,
        float lower[3], 
        float upper[3]) {
        id = id;
        leftChildId = leftChildId;
        lower = lower;
        upper = upper;
    }
};

MPI_Datatype createMPICell() {
    MPI_Datatype MPI_CELL;
    const int nitems=4;
    int  blocklengths[nitems] = {1, 1, DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Cell, id);
    offsets[1] = offsetof(Cell, leftChildId);
    offsets[2] = offsetof(Cell, lower);
    offsets[3] = offsetof(Cell, upper);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);

    return MPI_CELL;
}

#endif // CELL_H