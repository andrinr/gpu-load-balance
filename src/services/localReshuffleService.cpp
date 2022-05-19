//
// Created by andrin on 19/05/22.
//

#include "localReshuffleService.h"
#include "serviceManager.h"
#include <blitz/array.h>
#include "../cell.h"

LocalReshuffleService::LocalReshuffleService() {};

void LocalReshuffleService::run(void *rawInputData, void *rawOutputData) {
    LocalReshuffleServiceInput inputData = *(struct LocalReshuffleServiceInput*)rawInputData;
    LocalServiceServiceOutput outputData;

    Orb orb = *manager->orb;
    blitz::Array<int, 1> indices(inputData.nCells);
    for (int cellPtrOffset = 0; cellPtrOffset < inputData.nCells; ++cellPtrOffset){
        Cell cell = *(inputData.cells + cellPtrOffset);
        int beginInd = orb.cellToParticle(cell.id, 0);
        int endInd = orb.cellToParticle(cell.id, 1);

        blitz::Array<float,1> particles = orb.particles(blitz::Range(beginInd, endInd), cell.cutAxis);
        float * startPtr = particles.data();
        float *endPtr = startPtr + (endInd - beginInd);

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

    outputData.nCutIndices = inputData.nCells;
    outputData.cutIndices = indices.data();
    rawOutputData = &outputData;
};

int LocalReshuffleService::getNInputBytes(void *inputPtr) const {
    LocalReshuffleServiceInput input = *(struct LocalReshuffleServiceInput*)inputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( input.nCells ) * sizeof(Cell);
    nBytes += sizeof(int);
    return nBytes;
};

int LocalReshuffleService::getNOutputBytes(void *outputPtr) const {
    LocalServiceServiceOutput output = *(struct LocalServiceServiceOutput*)outputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( 1 + output.nCutIndices ) * sizeof(int);
    return nBytes;
};
