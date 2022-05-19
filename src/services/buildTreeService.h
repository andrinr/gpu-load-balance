//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_BUILDTREESERVICE_H
#define GPU_LOAD_BALANCE_BUILDTREESERVICE_H


#include "baseService.h"
#include "../comm/MPIMessaging.h"

class BuildTreeService : public BaseService {
public:
    struct ServiceInput {
        Cell * root;
        Messaging * messaging;
    };

    const int serviceID = 2;

    BuildTreeService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_BUILDTREESERVICE_H
