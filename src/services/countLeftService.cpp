//
// Created by andrin on 19/05/22.
//
#include "countLeftService.h"
#include "serviceManager.h"
#include <blitz/array.h>
#include "../cell.h"

CountLeftService::CountLeftService() {};

void CountLeftService::run(void *inputBuffer, int inputBufferLength, void *outputBuffer, int outputBufferLength) {

    CountLeftServiceInput inputData = *(struct CountLeftServiceInput*)rawInputData;
    CountLeftServiceOutput outputData;

    Orb orb = *manager->orb;

    blitz::Array<int, 1> counts(inputData.nCells);
    for (int cellPtrOffset = 0; cellPtrOffset < inputData.nCells; ++cellPtrOffset){
        Cell cell = *(inputData.cells + cellPtrOffset);
        if (cell.cutAxis == -1) continue;
        int beginInd = orb.cellToParticle(cell.id, 0);
        int endInd = orb.cellToParticle(cell.id, 1);

        blitz::Array<float,1> particles = orb.particles(blitz::Range(beginInd, endInd), cell.cutAxis);
        float * startPtr = particles.data();
        float * endPtr = startPtr + (endInd - beginInd);

        int nLeft = 0;

        float cut = (cell.cutMarginRight + cell.cutMarginLeft) / 2.0;
        for(auto p= startPtr; p<endPtr; ++p) nLeft += *p < cut;

        std::cout << nLeft << " ";
        counts[cellPtrOffset] = nLeft;
    }

    outputData.nCounts = inputData.nCells;
    outputData.counts = counts.data();
    rawOutputData = &outputData;
}

std::tuple<int, int> CountService::getNBytes(int bufferLength) const {
    return std::make_tuple(bufferLength * sizeof(Cell), bufferLength * sizeof (int))
}

int CountLeftService::getNOutputBytes(int outputBufferLength) const {
    return outputBufferLength * sizeof(int);
}

int CountLeftService::getNInputBytes(int inputBufferLength) const {
    return inputBufferLength * sizeof(Cell)
}