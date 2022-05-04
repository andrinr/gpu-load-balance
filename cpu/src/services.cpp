
#include "services.h"

Services::Services(Orb& o) : orb(o) {

}

int* Services::countLeft(Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));
    int nLeft = 0;

    int beginInd = orb.cellToParticle(data.cells(0).id, 0);
    int endInd = orb.cellToParticle(data.cells(n-1).id, 1);
    float *slice = orb.particles(beginInd, endInd);

    blitz::Array<int, 1> counts(data.cells.nrows());
    int cellInd = 0;
    for (int i = 0; i < data.end - data.begin; i += DIMENSIONS) {
        if (cells(cellInd).axis == -1) {
            i +=
        }
        float* p = slice + i;

        // todo: We need to ensure to cell is never empty (no particles) for this to work!!
        cellInd += p > data.cToP(cellInd, 1);
        Cell& cell = cells(cellInd);

        counts(cellInd) += *p + cell.cutAxis < (cell.left + cell.right) / 2.0;
    }

    return counts.data();
}

int* Services::count(Cell* c, int n) {
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));
    int total = 0;

    int beginInd = orb.cellToParticle(data.cells(0).id, 0);
    int endInd = orb.cellToParticle(data.cells(0).id, 1);
    float *slice = orb.particles(beginInd, endInd);

    blitz::Array<int, 1> counts(data.cells.nrows());
    int cellInd = 0;
    for (int i = 0; i < data.end - data.begin; i += DIMENSIONS) {
        float* p = slice + i;

        // todo: We need to ensure to cell is never empty (no particles) for this to work!!
        cellInd += p > data.cToP(cellInd, 1);
        Cell& cell = data.cells(cellInd);

        counts(cellInd) += *p + cell.cutAxis < (cell.left + cell.right) / 2.0;
    }

    return counts.data();
}

int* Services::buildTree(Cell* c, int n) {

    // Blitz: 2.3.7
    blitz::Array<Cell, 1> cells(c, blitz::shape(n));

    // loop over levels of tree
    for (int l = 1; l < ceil(log2(cell->nLeafCells)); l++) {
        int begin_prev = 2**(l-1);
        int end_prev = 2**l;
        int begin = end_prev;
        int end = 2**(l+1);

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
            cells(i).left = lower[axis];
            cells(i).right = upper[axis];
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

int* Services::findCuts() {

    return nullptr;
}

int* Services::localReshuffle(Cell* cells, int n) {

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



