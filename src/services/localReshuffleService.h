//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H
#define GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H

#include "baseService.h"

const int LOCAL_RESHUFFLE_SERVICE_ID = 3;

struct LocalReshuffleServiceInput {
    Cell * cells;
    int nCells;
};

struct LocalServiceServiceOutput {
    int * cutIndices;
    int nCutIndices;
};

class LocalReshuffleService : public BaseService {
public:

    const int serviceID = LOCAL_RESHUFFLE_SERVICE_ID;

    LocalReshuffleService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};



#endif //GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H
