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

#endif // CELL_H