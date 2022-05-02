#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <blitz/array.h>   
#include <cstdint>

class Cell {
public:
    int nCells;
    uint8_t cutAxis;
    float left;
    float right;
    float lower[3], upper[3];

    cut(float position, int axis);
    Cell (
        int nCells_,
        uint8_t cutAxis_,
        float left_,
        float right_,
        float lower_[3], 
        float upper_[3]
    ) :
        nCells(nCells_),
        cutAxis(cutAxis_),
        left(left_),
        right(right_),
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