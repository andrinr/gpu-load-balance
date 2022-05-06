#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <cstdint>
#include "tuple"

class Cell {
public:
    int id;
    int nLeafCells;
    int nLeftOfCut;
    int8_t cutAxis;
    float cutMarginLeft;
    float cutMarginRight;
    float lower[3], upper[3];

    std::tuple<Cell, Cell> cut();
    void setCutAxis();
    void setCutMargin();
    int getLeftChildId();
    int getRightChildId();
    int getParentId();

    Cell();

    Cell (
        int id_,
        int nLeafCells_,
        float lower_[3], 
        float upper_[3]
    );
};

#endif // CELL_H