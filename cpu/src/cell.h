#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <blitz/array.h>   

struct Cell {
    int id;
    int leftChildId;
    int nCells;
    int cutAxis;
    float cutPos;
    float lower[3], upper[3];

    Cell (
        int id_,
        int leftChildId_,
        int nCells_,
        int cutAxis_,
        float cutPos_,
        float lower_[3], 
        float upper_[3]
    ) : id(id_),
        leftChildId(leftChildId_),
        nCells(nCells_),
        cutAxis(cutAxis_),
        cutPos(cutPos_)
        {
        lower[0] = lower_[0];
        lower[1] = lower_[1];
        lower[2] = lower_[2];
        upper[0] = upper_[0];
        upper[1] = upper_[1];
        upper[2] = upper_[2];
    }
};

#endif // CELL_H