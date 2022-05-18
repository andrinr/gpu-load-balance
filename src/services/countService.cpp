//
// Created by andrin on 15/05/22.
//
#include "countService.h"


void CountService::run(void * rawInputData, void * rawOutputData) {
    ServiceInput inputData = *(struct ServiceInput*)rawInputData;
    ServiceOutput outputData;

    Orb orb = manager->orb;

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


int CountService::getNInputBytes(void *inputPtr) {
    ServiceInput input = *(struct ServiceInput*)inputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( input.nCells ) * sizeof(Cell);
    nBytes += sizeof(int);
    return nBytes;
}

int CountService::getNOutputBytes(void *outputPtr) {
    ServiceOutput output = *(struct ServiceOutput*)outputPtr;
    // we add plus one for the nSums variable itself
    int nBytes = ( 1 + output.nSums ) * sizeof(int);
    return nBytes;
}