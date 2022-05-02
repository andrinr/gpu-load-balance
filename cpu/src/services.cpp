
#include "services.h"

blitz::Array<int, 1> Services::count(InCount& data) {
    int nLeft = 0;

    float *slice = data.orb.particles(data.begin,0);
    float* pointerStart = slice;
    float* pointerEnd = slice + (data.end-data.begin);

    blitz::Array<int, 1> counts(data.cells.nrows());
    int cellInd = 0;
    for (int i = 0; i < data.end - data.begin; i += DIMENSIONS) {
        float* p = pointerStart + i;

        // todo: We need to ensure to cell is never empty (no particles) for this to work!!
        cellInd += p > data.cToP(cellInd, 1);
        Cell& cell = data.cells(cellInd);

        counts(cellInd) += *p + cell.cutAxis < (cell.left + cell.right) / 2.0;
    }

    return counts;
}

int Services::control(int data) {

}



