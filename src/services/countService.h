//
// Created by andrin on 15/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTSERVICE_H

#include "baseService.h"
#include "../cell.h"
#include "../orb.h"

const int COUNT_SERVICE_ID = 1;

struct CountServiceInput {
    Cell * cells;
    int nCells;
};

struct CountServiceOutput {
    int * sums;
    int nSums;
};

class CountService : public BaseService {
public:

    const int serviceID = COUNT_SERVICE_ID;

    CountService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTSERVICE_H
