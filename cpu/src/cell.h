#ifndef CELL_H // include guard
#define CELL_H

#include "constants.h"
#include <cstdint>
#include "tuple"

struct Cell {
    int id;
    int nLeafCells;
    int cutAxis;
    float cutMarginLeft;
    float cutMarginRight;
    float lower[3], upper[3];

    Cell::Cell(
            int id_,
            int nLeafCells_,
            float *lower_,
            float *upper_) : id(id_), nLeafCells(nLeafCells_) {

#if DEBUG
        if (nLeafCells < 1) {
        throw std::invalid_argument("Cell.Cell: nLeafCells is less than one.");
    }
#endif // DEBUG

        cutAxis = -1;
        cutMarginLeft = 0.0;
        cutMarginRight = 0.0;
        lower[0] = lower_[0];
        lower[1] = lower_[1];
        lower[2] = lower_[2];
        upper[0] = upper_[0];
        upper[1] = upper_[1];
        upper[2] = upper_[2];
    };
};

namespace Cell {
    static std::tuple<Cell, Cell> cut(Cell &cell);
    static void setCutAxis(Cell &cell);
    static void setCutMargin(Cell &cell);
    static int getLeftChildId(Cell &cell);
    static int getRightChildId(Cell &cell);
    static int getParentId(Cell &cell);

    static void log(Cell &cell);
};

#endif // CELL_H