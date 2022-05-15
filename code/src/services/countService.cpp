//
// Created by andrin on 15/05/22.
//

#include "countService.h"

void CountService::run(void * rawInputData, void * rawOutputData) {
    ServiceInput inputData = *(struct ServiceInput*)rawInputData;
    ServiceOutput outputData;
    /*std::cout << " counting " << n << std::endl;
*/
    for (int i = 0; i < inputData.nCells; ++i) {
        std::cout << MPIMessaging::rank << "A" << std::endl;
        Cell cell = inputData.cells[i];
        std::cout << MPIMessaging::rank << "B" << std::endl;
        CellHelpers::log(cell);
        int begin = orb.cellToParticle(cell.id, 0);
        int end = orb.cellToParticle(cell.id, 1);

        outputData.sums[i] = begin - end;
    }

    outputData.nSums = inputData.nCells;

    rawOutputData = &output;
}