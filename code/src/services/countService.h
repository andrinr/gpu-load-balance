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
        Orb & orb;
        Cell * cells;
        int nCells;

        ServiceInput();
    };

    struct ServiceOutput {
        int * sums;
        int nSums;

        ServiceOutput();
    };

    static const int serviceID = 1;

    CountService();

    virtual void run(void * rawInputData, void * rawOutputData) = 0;
};


#endif //CODE_COUNTSERVICE_H
