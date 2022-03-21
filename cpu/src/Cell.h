#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
/**
 * @brief The Cells class is basically a wrapper around the array datastructure which 
 * holds all data of the unwrapped Cell tree datastructure.
 * 
 */
struct Cell {
    int begin;
    int end;
    int id;
    int leftChildId;
    float lower[DIMENSIONS];
    float upper[DIMENSIONS];
};

struct Cut {
    int begin;
    int end;
    int axis;
    float pos;
};

#endif // CELL_H