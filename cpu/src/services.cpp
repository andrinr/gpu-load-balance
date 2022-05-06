#include "services.h"
#include "comm/MPIMessaging.h"
#include <math.h>
#include <algorithm>

int* Services::countLeft(Orb &orb, Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    int begin = orb.cellToParticle(cells(0).id, 0);
    int end = orb.cellToParticle(cells(n-1).id, 1);
    int size = end - begin;

    float* slice = &orb.particles(begin, end);

    blitz::Array<int, 1> counts(cells.rows());
    int cellInd = 0;
    for (int i = 0; i < size; i += DIMENSIONS) {
        Cell cell = cells(cellInd);
        if (cell.cutAxis == -1) {
            i += 1;
        }
        float* p = slice + i;

        // todo: We need to ensure to cell is never empty (no particles) for this to work!!
        if (i > orb.cellToParticle(cell.id, 1) - begin) {
            orb.cellToParticle(cell.getLeftChildId(), 1) = counts(cellInd);
            orb.cellToParticle(cell.getRightChildId(), 0) = counts(cellInd);
            cellInd++;
        }

        counts(cellInd) += *p + cell.cutAxis < (cell.cutMarginLeft + cell.cutMarginRight) / 2.0;
    }

    // todo: new boundaries will need to be set

    return counts.data();
}

int* Services::localReshuffle(Orb& orb, Cell* cells, int n) {

    for (int cellPtrOffset = 0; cellPtrOffset < n; cellPtrOffset++){
        int id = (cells + cellPtrOffset)->id;
        int beginInd = orb.cellToParticle(id, 0);
        int endInd = orb.cellToParticle(id, 1);
        int i = beginInd;
        for (int j = beginInd; j < endInd; j++) {
            if (orb.particles(j, (cells + cellPtrOffset)->cutAxis) <
                (cells + cellPtrOffset)->cutMarginLeft) {
                orb.swap(i, j);
                i = i + 1;
            }
        }

        orb.swap(i, endInd - 1);
        mids(cellPtrOffset) = i;
    }

    return mids.data();
}

int* Services::count(Orb &orb, Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));
    blitz::Array<int, 1> count(cells.rows());

    for (int i = 0; i < cells.rows(); ++i) {
        int begin = orb.cellToParticle(cells(i).id, 0);
        int end = orb.cellToParticle(cells(i).id, 1);

        count(i) = begin - end;
    }

    return count.data();
}

int* Services::buildTree(Orb& orb, Cell* c, int n) {

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    Cell root = c[0];

    // loop over levels of tree
    for (int l = 1; l < ceil(log2(root.nLeafCells)); l++) {

        int a = std::pow(2, (l-1));
        int b = std::min((int)std::pow(2, l), root.nLeafCells);

        bool status;
        int* sum;
        std::tie(status, sum) = MPIMessaging::dispatchService(
                orb,
                countService,
                cells(blitz::Range(a, b)).data(),
                b - a,
                sum,
                b - a,
                0);

        blitz::Array<Cell, 1> countBlitz(sum, blitz::shape(b - a));
        // Loop
        bool foundAll = true;

        while(foundAll) {
            int* sumLeft;
            foundAll = true;
            std::tie(status, sumLeft) = MPIMessaging::dispatchService(
                    orb,
                    countLeftService,
                    cells(blitz::Range(a, b)).data(),
                    b - a,
                    sumLeft,
                    b - a,
                    0);

            blitz::Array<int, 1> countLeftBlitz(sumLeft, blitz::shape(b - a));

            for (int i = a; i < b; i++) {

               if (abs(countLeftBlitz(i) - countLeftBlitz(i) / 2.0) < CUTOFF) {
                   cells(i).cutAxis = -1;
               }
               else if (countLeftBlitz(i) - countLeftBlitz(i) / 2.0 > 0){
                    cells(i).cutMarginRight = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                    foundAll = false;
               }
               else {
                   cells(i).cutMarginLeft = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                   foundAll = false;
               }
            }
        }

        // Dispatch reshuffle service
        int * dummy;
        std::tie(status, dummy) = MPIMessaging::dispatchService(
                orb,
                localReshuffleService,
                cells(blitz::Range(a, b)).data(),
                b - a,
                dummy,
                b - a,
                0);

        // Split and store all cells on current heap level
        for (int i = a; i < b; i++) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = cells(i).cut();

            cellRight.setCutAxis();
            cellRight.setCutMargin();
            cellLeft.setCutAxis();
            cellLeft.setCutMargin();

            cells(cells(i).getLeftChildId()) = cellLeft;
            cells(cells(i).getRightChildId()) = cellRight;
        }
    }
    return nullptr;
}
