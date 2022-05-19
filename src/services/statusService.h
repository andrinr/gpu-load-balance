//
// Created by andrin on 19/05/22.
//

#ifndef GPU_LOAD_BALANCE_STATUSSERVICE_H
#define GPU_LOAD_BALANCE_STATUSSERVICE_H

const int STATUS_SERVICE_ID = 5;

struct CountServiceInput {
    int status
};

struct CountServiceOutput {
    int status
};

class StatusService : public BaseService {
public:

    const int serviceID = STATUS_SERVICE_ID;

    StatusService();

    void run(void * rawInputData, void * rawOutputData) override;
    int getNInputBytes(void * inputPtr) const override;
    int getNOutputBytes(void * outputPtr) const override;
};


#endif //GPU_LOAD_BALANCE_STATUSSERVICE_H
