#ifndef CELL_H // include guard
#define CELL_H

#include <cstdint>
#include "tuple"
#include <iostream>
#include <math.h>

struct Cell {
    int id;
    int nLeafCells;
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

    // todo: do this
    int getLeftChildId() const {// does not modify{
        return id * 2 - 1;
    }
};



namespace CellHelpers {

    static int getTotalNumberOfCells(Cell &cell) {
        return 2 * cell.nLeafCells - 1;
    }

    static int getNLevels(Cell &cell) {
        return ceil(log2(cell.nLeafCells));
    }

    static int getNCellsOnLastLevel(Cell &cell) {
        int depth = getNLevels(cell);
        return 2 * cell.nLeafCells - pow(2, depth);
    }

    static int getLeftChildId(Cell &cell) {
        return (cell.id + 1) * 2 - 1;
    }

    static int getRightChildId(Cell &cell) {
        return (cell.id + 1) * 2;
    }

    static int getParentId(Cell &cell) {
        //todo
        return cell.id;
    }

    static std::tuple <Cell, Cell> cut(Cell &cell) {
        int nCellsLeft = ceil(cell.nLeafCells / 2.0);
        int nCellsRight = cell.nLeafCells - nCellsLeft;

        Cell leftChild(
                CellHelpers::getLeftChildId(cell),
                nCellsLeft,
                cell.lower,
                cell.upper);
        leftChild.upper[cell.cutAxis] = (cell.cutMarginRight - cell.cutMarginLeft) / 2.0;

        Cell rightChild(
                CellHelpers::getRightChildId(cell),
                nCellsRight,
                cell.lower,
                cell.upper);
        rightChild.lower[cell.cutAxis] = (cell.cutMarginRight - cell.cutMarginLeft) / 2.0;

        return std::make_tuple(leftChild, rightChild);
    }

    static void setCutAxis(Cell &cell) {
        int maxD = -1;

        float maxSize = 0.0;
        for (int d = 0; d < 3; d++) {
            float size = cell.upper[d] - cell.lower[d];

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

        cell.cutAxis = int(maxD);
    }

    static void setCutMargin(Cell &cell) {
        cell.cutMarginLeft = cell.lower[cell.cutAxis];
        cell.cutMarginRight = cell.upper[cell.cutAxis];
    }

    static void log(Cell &cell) {
        printf("Cell ID: %u \n", cell.id);
        printf("lower: %f, %f, %f \n", cell.lower[0], cell.lower[1], cell.lower[2]);
        printf("upper: %f, %f, %f \n", cell.upper[0], cell.upper[1], cell.upper[2]);
        printf("NLeafCells: %u, axis: %i , found: %u\n", cell.nLeafCells, cell.cutAxis, cell.foundCut);
        printf("Margin left %f, right: %f \n", cell.cutMarginLeft, cell.cutMarginRight);
    }
}

#endif // CELL_H
