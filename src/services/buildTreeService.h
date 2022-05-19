//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_BUILDTREESERVICE_H
#define GPU_LOAD_BALANCE_BUILDTREESERVICE_H


#include "baseService.h"
#include "../comm/MPIMessaging.h"

const int BUILD_TREE_SERVICE_ID = 4;

struct BuildTreeServiceInput {
    Cell * root;
    Messaging * messaging;
};

class BuildTreeService : public BaseService {
public:

    const int serviceID = BUILD_TREE_SERVICE_ID;

    BuildTreeService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_BUILDTREESERVICE_H
