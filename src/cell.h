#ifndef CELL_H // include guard
#define CELL_H

#include <cstdint>
#include "tuple"
#include <iostream>
#include <math.h>

struct Cell {
    int id;
    int nLeafCells;
    int prevCutAxis;
    int cutAxis;
    bool foundCut;
    float cutMarginLeft;
    float cutMarginRight;
    float lower[3], upper[3];

    Cell(
            int id_,
            int nLeafCells_,
            float *lower_,
            float *upper_) : id(id_), nLeafCells(nLeafCells_) {

#if DEBUG
    if (nLeafCells < 1) {
        throw std::invalid_argument("Cell.Cell: nLeafCells is less than one.");
    }
#endif // DEBUG
        foundCut = false;
        prevCutAxis = -1;
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

    Cell() = default; // Trivial structs need a default constructor

    int getLeftChildId() const {
        return (id + 1) * 2 - 1;
    }

    int getRightChildId() const {
        return (id + 1) * 2;
    }

    int getTotalNumberOfCells() const {
        return 2 * nLeafCells - 1;
    }

    int getNLevels() const {
        return ceil(log2(nLeafCells));
    }

    int getNCellsOnLastLevel() const {
        int depth = getNLevels();
        return 2 * nLeafCells - pow(2, depth);
    }

    float getCut() const {
        return (cutMarginRight + cutMarginLeft) / 2.0;
    }

    std::tuple <Cell, Cell> cut() {
        int nCellsLeft = ceil(nLeafCells / 2.0);
        int nCellsRight = nLeafCells - nCellsLeft;

        float cut = getCut();
        Cell leftChild(
                getLeftChildId(),
                nCellsLeft,
                lower,
                upper);
        leftChild.upper[cutAxis] = cut;

        Cell rightChild(
                getRightChildId(),
                nCellsRight,
                lower,
                upper);
        rightChild.lower[cutAxis] = cut;

        return std::make_tuple(leftChild, rightChild);
    }

    void setCutAxis() {
        int maxD = -1;

        float maxSize = 0.0;
        for (int d = 0; d < 3; d++) {
            float size = upper[d] - lower[d];

#if DEBUG
            if (size < 0.0) {
                throw std::invalid_argument("Cell.SetCutAxis: size is zero or negative.");
            }
#endif // DEBUG

            if (size > maxSize) {
                maxSize = size;
                maxD = d;
            }
        }

        // keep track of history
        prevCutAxis = cutAxis;
        cutAxis = int(maxD);
    }

    void setCutMargin() {
        cutMarginLeft = lower[cutAxis];
        cutMarginRight = upper[cutAxis];
    }

    void log() const {
        printf("Cell ID: %u \n", id);
        printf("lower: %f, %f, %f \n", lower[0], lower[1], lower[2]);
        printf("upper: %f, %f, %f \n", upper[0], upper[1], upper[2]);
        printf("NLeafCells: %u, axis: %i , found: %u\n", nLeafCells, cutAxis, foundCut);
        printf("Margin left %f, right: %f \n", cutMarginLeft, cutMarginRight);
    }
};

#endif // CELL_H
