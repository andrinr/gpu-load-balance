//
// Created by andrin on 15/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTSERVICE_H

#include "baseService.h"
#include "../cell.h"
#include "../orb.h"

class CountService : public BaseService {
public:
    struct ServiceInput {
        Cell * cells;
        int nCells;
    };

    struct ServiceOutput {
        int * sums;
        int nSums;
    };

    const int serviceID = 1;

    CountService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTSERVICE_H
