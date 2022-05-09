#include "cell.h"
#include <iostream>
#include <math.h>

Cell::Cell() {

};

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

std::tuple <Cell, Cell> Cell::cut(Cell &cell) {
    int nCellsLeft = ceil(cell.nLeafCells / 2.0);
    int nCellsRight = cell.nLeafCells - nCellsLeft;

    Cell leftChild (Cell::getLeftChildId(cell), nCellsLeft, cell.lower, cell.upper);
    leftChild.upper[cell.cutAxis] = (cell.cutMarginRight - cell.cutMarginLeft) / 2.0;

    Cell rightChild (Cell::(), nCellsRight, lower, upper);
    rightChild.lower[cutAxis] = (cutMarginRight - cutMarginLeft) / 2.0;

    return std::make_tuple(leftChild, rightChild);
}


int Cell::getLeftChildId() {
    return id * 2 - 1;
}

int Cell::getRightChildId() {
    return id * 2;
}

int Cell::getParentId() {
    //todo
    return id;
}

void Cell::setCutAxis() {
    uint8_t maxD = DIMENSIONS;
    float maxSize = 0.0;
    for (int d = 0; d < DIMENSIONS; d++) {
        float size = upper[d] - lower[d];

#if DEBUG
        if (size == 0.0) {
            throw std::invalid_argument("Cell.SetCutAxis: size is zero.");
        }
#endif // DEBUG

        if (size > maxSize) {
            maxSize = size;
            maxD = d;
        }
    }

    cutAxis = int(maxD);
}


void Cell::setCutMargin() {
    cutMarginLeft = lower[cutAxis];
    cutMarginRight = upper[cutAxis];
}

void Cell::log() {
    std::cout << "------" << std::endl;
    std::cout << id << " cell id" << std::endl;
    std::cout << lower[0] << " " << lower[1] << " " << lower[2] << " lower " << std::endl;
    std::cout << upper[0] << " " << upper[1] << " " << upper[2] << " upper " << std::endl;
    std::cout << nLeafCells << " nCells " << cutAxis << " cutAxis " << std::endl;
    std::cout << "------" << std::endl;
}