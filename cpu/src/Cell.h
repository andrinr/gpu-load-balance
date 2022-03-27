#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <blitz/array.h>   

struct Cell {
    int begin;
    int end;
    int id;
    int leftChildId;
    blitz::TinyVector<float, DIMENSIONS> lower;
    blitz::TinyVector<float, DIMENSIONS> upper;
};

struct Cut {
    int begin;
    int end;
    int axis;
    float pos;
};

#endif // CELL_H