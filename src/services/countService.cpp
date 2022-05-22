//
// Created by andrin on 15/05/22.
//
#include "countService.h"
#include "serviceManager.h"

CountService::CountService() {};

void CountService::run(void *inputBuffer, int inputBufferLength, void *outputBuffer, int outputBufferLength) {
    Cell *  cells = (Cell*)inputBuffer;

    Orb orb = *manager->orb;

    blitz::Array<int, 1> counts(inputBufferLength);
    for (int i = 0; i < inputBufferLength; ++i) {

        Cell cell = inputData.cells[i];
        CellHelpers::log(cell);
        int begin = orb.cellToParticle(cell.id, 0);
        int end = orb.cellToParticle(cell.id, 1);

        outputData.sums[i] = begin - end;
    }

    rawOutputData = &outputData;
}

std::tuple<int, int> CountService::getNBytes(int bufferLength) const {
    return std::make_tuple(bufferLength * sizeof(Cell), bufferLength * sizeof (int))
}

int CountService::getNOutputBytes(int outputBufferLength) const {
    return outputBufferLength * sizeof(int);
}

int CountService::getNInputBytes(int inputBufferLength) const {
    return inputBufferLength * sizeof(Cell)
}