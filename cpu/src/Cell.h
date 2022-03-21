#ifndef CELL_H // include guard
#define CELL_H

/**
 * @brief The Cells class is basically a wrapper around the array datastructure which 
 * holds all data of the unwrapped Cell tree datastructure.
 * 
 */
struct Cell {
    int begin;
    int end;
    int leftChildId;
    int id;
    float* lower;
    float* upper;
};

#endif // CELL_H