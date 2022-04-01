#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <blitz/array.h>   

struct Cell {
    int begin;
    int end;
    int id;
    int leftChildId;
    float lower[3], upper[3];
    Cell(
        int begin,
        int end, 
        int id,
        int leftChildId,
        blitz::TinyVector<float, DIMENSIONS> l, 
        blitz::TinyVector<float, DIMENSIONS> u) {
        begin = being;
        end = end;
        id = id;
        leftChildId = leftChildId;
        lower[0] = l(0);
        lower[1] = l(1);
        lower[2] = l(2);
        upper[0] = u(0);
        upper[1] = u(1);
        upper[2] = u(2);
    }
};

MPI_Datatype createMPICell() {
    MPI_Datatype MPI_CELL;
    const int nitems=6;
    int  blocklengths[nitems] = {1, 1, 1, 1, DIMENSIONS, DIMENSIONS};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Cell, begin);
    offsets[1] = offsetof(Cell, end);
    offsets[2] = offsetof(Cell, id);
    offsets[3] = offsetof(Cell, leftChildId);
    offsets[4] = offsetof(Cell, lower);
    offsets[5] = offsetof(Cell, upper);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);

    return MPI_CELL;
}

struct Cut {
    int cellId;
    int axis;
    float pos;
};

MPI_Datatype createMPICut() {
    MPI_Datatype MPI_CUT;
    const int nitems=3;
    int  blocklengths[nitems] = {1, 1, 1};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_FLOAT};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Cut, cellId);
    offsets[1] = offsetof(Cut, axis);
    offsets[2] = offsetof(Cut, pos);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CUT);
    MPI_Type_commit(&MPI_CUT);
}

#endif // CELL_H