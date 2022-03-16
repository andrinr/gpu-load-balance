#ifndef CELLS_H // include guard
#define CELLS_H

#include "constants.h"

/**
 * @brief The Cells class is basically a wrapper around the array datastructure which 
 * holds all data of the unwrapped Cell tree datastructure.
 * 
 */
class Cells {
public:
    Cells(float left, float right);
    float getCornerA(int id, int axis);
    void setCornerA(int id, int axis, float value);
    float getCornerB(int id, int axis);
    void setCornerB(int id, int axis, float value);
    int getBegin(int id);
    void setBegin(int id, int value);
    int getEnd(int id);
    void setEnd(int id, int value);
    int getLeftChild(int id);
    int setLeftChild(int id, int value);
    int getRightChild(int id);

private:
    int* cellInfo = new int[MAX_CELL_COUNT * 3];
    float* cellGeometry = new float[MAX_CELL_COUNT * DIMENSIONS * 2];

};

#endif // CELLS_H