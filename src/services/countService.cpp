//
// Created by andrin on 15/05/22.
//
#include "countService.h"
#include "serviceManager.h"

CountService::CountService() {};

void CountService::run(void * rawInputData, void * rawOutputData) {
    CountServiceInput inputData = *(struct CountServiceInput*)rawInputData;
    CountServiceOutput outputData;

    Orb orb = *manager->orb;

    for (int i = 0; i < inputData.nCells; ++i) {

        Cell cell = inputData.cells[i];
        CellHelpers::log(cell);
        int begin = orb.cellToParticle(cell.id, 0);
        int end = orb.cellToParticle(cell.id, 1);

        outputData.sums[i] = begin - end;
    }

    outputData.nSums = inputData.nCells;
    rawOutputData = &outputData;
}


int CountService::getNInputBytes(void *inputPtr) const {
    CountServiceInput input = *(struct CountServiceInput*)inputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( input.nCells ) * sizeof(Cell);
    nBytes += sizeof(int);
    return nBytes;
}

int CountService::getNOutputBytes(void *outputPtr) const {
    CountServiceOutput output = *(struct CountServiceOutput*)outputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( 1 + output.nSums ) * sizeof(int);
    return nBytes;
}