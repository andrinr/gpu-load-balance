//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H
#define GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H

#include "baseService.h"

const int LOCAL_RESHUFFLE_SERVICE_ID = 3;

class LocalReshuffleService : public BaseService {
public:

    const int serviceID = LOCAL_RESHUFFLE_SERVICE_ID;

    LocalReshuffleService();

    void run(const void * inputBuffer,
             const int nInputElements,
             void * outputBuffer,
             int nOutputElements) override;

    int getNInputBytes(int inputBufferLength) const override;
    int getNOutputBytes(int outputBufferLength) const override;
};

#endif //GPU_LOAD_BALANCE_LOCALRESHUFFLESERVICE_H
