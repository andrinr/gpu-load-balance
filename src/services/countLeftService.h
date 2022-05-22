//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#define GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
#include "baseService.h"

const int COUNT_LEFT_SERVICE_ID = 2;

class CountLeftService : public BaseService {
public:

    const int serviceID = COUNT_LEFT_SERVICE_ID;

    CountLeftService();

    void run(const void * inputBuffer,
             const int nInputElements,
             void * outputBuffer,
             int nOutputElements) override;

    int getNInputBytes(int inputBufferLength) const override;
    int getNOutputBytes(int outputBufferLength) const override;
};


#endif //GPU_LOAD_BALANCE_COUNTLEFTSERVICE_H
