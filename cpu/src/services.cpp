#include "services.h"
#include "comm/MPIMessaging.h"
#include <math.h>

int* Services::countLeft(Orb &orb, Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    int begin = orb.cellToParticle(cells(0).id, 0);
    int end = orb.cellToParticle(cells(n-1).id, 1);
    int size = end - begin;

    float* slice = &orb.particles(begin, end);

    blitz::Array<int, 1> counts(cells.rows());
    int cellInd = 0;
    for (int i = 0; i < size; i += DIMENSIONS) {
        if (cells(cellInd).cutAxis == -1) {
            i += 1;
        }
        float* p = slice + i;

        // todo: We need to ensure to cell is never empty (no particles) for this to work!!
        cellInd += i > orb.cellToParticle(cells(cellInd).id, 1) - begin;
        Cell& cell = cells(cellInd);

        counts(cellInd) += *p + cell.cutAxis < (cell.cutMarginLeft + cell.cutMarginRight) / 2.0;
    }

    return counts.data();
}

int* Services::count(Orb &orb, Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));
    blitz::Array<int, 1> totals(cells.rows());

    for (int i = 0; i < cells.rows(); ++i) {
        int begin = orb.cellToParticle(cells(i).id, 0);
        int end = orb.cellToParticle(cells(i).id, 1);

        totals(i) = begin - end;
    }

    return totals.data();
}

int* Services::buildTree(Orb& orb, Cell* c, int n) {

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    Cell cell = c[0];

    // loop over levels of tree
    for (int l = 1; l < ceil(log2(cell.nLeafCells)); l++) {
        int begin_prev = pow(2, (l-1));
        int end_prev = pow(2, l);
        int begin = end_prev;
        int end = pow(2,l+1);

        // Init cells
        for (int i = begin; i < end; i++) {
            float maxValue = 0;
            int axis = -1;

            for (int i = 0; i < DIMENSIONS; i++) {
                float size = cells(i).upper[i] - cells(i).lower[i];
                if (size > maxValue) {
                    maxValue = size;
                    axis = i;
                }
            }

            cells(i).cutAxis = axis;
            cells(i).cutMarginLeft = lower[axis];
            cells(i).cutMarginRight = upper[axis];
        }

        comm.signalService(1);
        int* totals_ = comm.dispatchService(
                Services::count,
                cells(blitz::Range(begin, min(nLeafCells, end))).data(),
                end - begin,
                int,
                end - begin,
                0);
        blitz::Array<Cell, 1> total(totals_, blitz::shape(end - begin));
        // Loop
        bool foundAll = true;

        while(!foundAll) {

            comm.signalService(1);
            int* counts_ = comm.dispatchService(
                    Services::countLeft,
                    cells(blitz::Range(begin, min(nLeafCells, end))).data(),
                    end - begin,
                    int,
                    end - begin,
                    0);

            blitz::Array<Cell, 1> counts(counts_, blitz::shape(end - begin));

            for (int i = begin; i < min(nLeafCells, end); i++) {

               if (abs(counts(i) - totals(i) / 2.0) < CUTOFF) {
                   cells(i).axis = -1;
               }
               else if (counts(i) - totals(i) / 2.0 > 0){
                    cells(i).right = (cells(i).left + cells(i).right) / 2.0;
                    foundAll = false;
               }
               else {
                   cells(i).left = (cells(i).left + cells(i).right) / 2.0;
                   foundAll = false;
               }
            }
        }

        // Dispatch reshuffle service

        comm.signalService(1);
        int* totals_ = comm.dispatchService(
                Services::localReshuffle,
                cells(blitz::Range(begin, min(nLeafCells, end))).data(),
                end - begin,
                int,
                end - begin,
                0);

        // Split and store all cells on current heap level
        for (int i = begin; i < min(nLeafCells, end); i++) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = cells(i).cut();

            cellRight.setCutAxis();
            cellRight.setCutMargin();
            cellLeft.setCutAxis();
            cellLeft.setCutMargin();

            cells(cells(i).getLeftChildId) = cellLeft;
            cells(cells(i).getRightChildId) = cellRight;
        }
    }
    return nullptr;
}

int* Services::localReshuffle(Orb& orb, Cell* cells, int n) {

    blitz::Array<int, 1> mids(n);
    for (int cellPtrOffset = 0; cellPtrOffset < n; cellPtrOffset++){
        int id = *(cells + cellPtrOffset).id;
        int beginInd = orb.cellToParticle(id, 0);
        int endInd = orb.cellToParticle(id, 1);
        int i = beginInd;
        for (int j = begin; j < endInd; j++) {
            if (orb.particles(j, axis) < cut) {
                orb.swap(i, j);
                i = i + 1;
            }
        }

        orb.swap(i, endInd - 1);
        mids(cellPtrOffset) = i;

        // todo: also update cellToParticle array
    }

    return mids.data();
}



int Services::control(int data) {

    return nullptr;
}



