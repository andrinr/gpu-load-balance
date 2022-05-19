//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#include "baseService.h"

class CountLeftService : public BaseService {
public:
    struct ServiceInput {
        Cell * cells;
        int nCells;
    };

    struct ServiceOutput {
        int * counts;
        int nCounts;
    };

    const int serviceID = 2;

    CountLeftService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
