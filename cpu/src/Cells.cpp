#include <stack>
#include "Cells.h"

Cells::Cells(float left, float right) {
    // Init values
    cellGeometry[0] = left;
    cellGeometry[1] = left;
    cellGeometry[2] = left;
    cellGeometry[3] = right;
    cellGeometry[4] = right;
    cellGeometry[5] = right;

    cellInfo[0] = 0;
    cellInfo[1] = COUNT;
}

float Cells::getCornerA(int id, int axis) {
    return cellGeometry[id * DIMENSIONS + axis];
}

void Cells::setCornerA(int id, int axis, float value) {
    cellGeometry[id * DIMENSIONS + axis] = value;
}

float Cells::getCornerB(int id, int axis) {
    return cellGeometry[(id + 1) * DIMENSIONS + axis];
}

void Cells::setCornerB(int id, int axis, float value) {
    cellGeometry[(id + 1) * DIMENSIONS + axis] = value;
}

int Cells::getBegin(int id) {
    return cellInfo[id * 3];
}

void Cells::setBegin(int id, int value) {
    cellInfo[id * 3] = value;
}

int Cells::getEnd(int id) {
    return cellInfo[id * 3 + 1];
}

void Cells::setEnd(int id, int value) {
    cellInfo[id * 3 + 1] = value;
}

int Cells::getLeftChild(int id) {
    return cellInfo[id * 3 + 2];
}

int Cells::setLeftChild(int id, int value) {
    cellInfo[id * 3 + 2] = value;
}

int Cells::getRightChild(int id) {
    return getLeftChild(id) + 1;
}
