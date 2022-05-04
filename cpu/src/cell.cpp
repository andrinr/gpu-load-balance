#include "cell.h"

Cell::Cell(
        int id_,
        int nLeafCells_,
        float *lower_,
        float *upper_) :
        id(id_),
        nLeafCells(nLeafCells_)
{
#if DEBUG
    if (nLeafCells < 1) {
        throw std::invalid_argument("Cell.Cell: nLeafCells is less than one.");
    }
#end // DEBUG

    cutAxis_ = -1;
    left = 0.0;
    right = 0.0;
    lower[0] = lower_[0];
    lower[1] = lower_[1];
    lower[2] = lower_[2];
    upper[0] = upper_[0];
    upper[1] = upper_[1];
    upper[2] = upper_[2];
};

std::tuple <Cell, Cell> Cell::cut() {
    int nCellsLeft = cei(nLeafCells / 2.0);
    int nCellsRight = nLeafCells - nCellsLeft;

    Cell leftChild (getLeftChildId(), nCellsLeft, lower, upper);
    leftChild.upper[axis] = (right - left) / 2.0;

    Cell rightChild (getRightChildId() * 2 + 1, nCellsRight, lower, upper);
    rightChild.lower[axis] = (right - left) / 2.0;
}


int Cell::getLeftChildId() {
    return id * 2;
}

int Cell::getRightChildId() {
    return cell.id * 2 + 1;
}

int Cell::getParentId() {
    //todo
}

void Cell::setCutAxis() {
    uint8_t maxD = DIMENSIONS;
    float maxSize = 0.0;
    for (uint8_t d = 0; d < DIMENSIONS; d++) {
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

    cutAxis = d;
}

void Cell::setCutAxis() {
    cutMarginLeft = lower[cutAxis];
    cutMarginRight = upper[cutAxis];
}

void Cell