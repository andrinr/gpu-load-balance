//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#include "baseService.h"

const int COUNT_LEFT_SERVICE_ID = 2;

struct CountLeftServiceInput {
    Cell * cells;
    int nCells;
};

struct CountLeftServiceOutput {
    int * counts;
    int nCounts;
};

class CountLeftService : public BaseService {
public:

    const int serviceID = COUNT_LEFT_SERVICE_ID;

    CountLeftService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
