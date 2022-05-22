//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_BUILDTREESERVICE_H
#define GPU_LOAD_BALANCE_BUILDTREESERVICE_H


#include "baseService.h"
#include "../comm/MPIMessaging.h"

const int BUILD_TREE_SERVICE_ID = 4;

class BuildTreeService : public BaseService {
public:

    const int serviceID = BUILD_TREE_SERVICE_ID;

    BuildTreeService();

    void run(const void * inputBuffer,
             const int nInputElements,
             void * outputBuffer,
             int nOutputElements) override;

    int getNInputBytes(int inputBufferLength) const override;
    int getNOutputBytes(int outputBufferLength) const override;
};


#endif //GPU_LOAD_BALANCE_BUILDTREESERVICE_H
