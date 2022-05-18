//
// Created by andrin on 15/05/22.
//

#ifndef CODE_COUNTSERVICE_H
#define CODE_COUNTSERVICE_H

#include "baseService.h"
#include "../cell.h"
#include "../orb.h"

class CountService : BaseService {
public:
    struct ServiceInput {
        Cell * cells;
        int nCells;

        ServiceInput();
    };

    struct ServiceOutput {
        int * sums;
        int nSums;

        ServiceOutput();
    };

    const int serviceID = 1;

    CountService();

    void run(void * rawInputData, void * rawOutputData);
    int getNInputBytes(void * inputPtr);
    int getNOutputBytes(void * outputPtr);
};


#endif //CODE_COUNTSERVICE_H
