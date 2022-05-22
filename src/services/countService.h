//
// Created by andrin on 15/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTSERVICE_H

#include "baseService.h"
#include "../cell.h"
#include "../orb.h"

const int COUNT_SERVICE_ID = 1;

class CountService : public BaseService {
public:

    const int serviceID = COUNT_SERVICE_ID;

    CountService();

    void run(const void * inputBuffer,
             const int nInputElements,
             void * outputBuffer,
             int nOutputElements) override;

    virtual std::tuple<int, int> getNBytes(int bufferLength) const override = 0;

    int getNInputBytes(int inputBufferLength) const override;
    int getNOutputBytes(int outputBufferLength) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTSERVICE_H
