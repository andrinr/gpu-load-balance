//
// Created by andrin on 19/05/22.
//

#include "localReshuffleService.h"
#include "serviceManager.h"
#include <blitz/array.h>
#include "../cell.h"

LocalReshuffleService::LocalReshuffleService() {};

void LocalReshuffleService::run(void *inputBuffer, int inputBufferLength, void *outputBuffer, int outputBufferLength) {
    Cell *  cells = (Cell*)inputBuffer;
    int * outputData;

    Orb orb = *manager->orb;
    blitz::Array<int, 1> indices(inputBufferLength);
    for (int cellPtrOffset = 0; cellPtrOffset < inputBufferLength; ++cellPtrOffset){
        Cell cell = *(cells + cellPtrOffset);
        int beginInd = orb.cellToParticle(cell.id, 0);
        int endInd = orb.cellToParticle(cell.id, 1);

        blitz::Array<float,1> particles = orb.particles(blitz::Range(beginInd, endInd), cell.cutAxis);
        float * startPtr = particles.data();
        float * endPtr = startPtr + (endInd - beginInd);

        int i = beginInd;
        for (int j = beginInd; j < endInd; j++) {
            if (orb.particles(j, cell.cutAxis) < (cell.cutMarginLeft + cell.cutMarginRight) / 2.0) {
                orb.swap(i, j);
                i = i + 1;
            }
        }

        orb.swap(i, endInd - 1);
        indices(cellPtrOffset) = i;
    }

    outputBufferLength
    outputBuffer = indices.data();
};

std::tuple<int, int> LocalReshuffleService::getNBytes(int bufferLength) const {
    return std::make_tuple(bufferLength * sizeof Cell, bufferLength * sizeof int)
}

int LocalReshuffleService::getNOutputBytes(int outputBufferLength) const {
    return outputBufferLength * sizeof(int);
}

int LocalReshuffleService::getNInputBytes(int inputBufferLength) const {
    return inputBufferLength * sizeof(Cell)
}