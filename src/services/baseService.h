#ifndef GPU_LOAD_BALANCE_BASESERVICE_H
#define GPU_LOAD_BALANCE_BASESERVICE_H

#include "../comm/messaging.h"
#include "../orb.h"
#include "vector"
class ServiceManager;
class Messaging;

class BaseService {
public:
    const int serviceID = -1;
    ServiceManager * manager;

    virtual void run(void * rawInputData, void * rawOutputData) = 0;
    virtual int getNInputBytes(void * inputPtr) const = 0;
    virtual int getNOutputBytes(void * outputPtr) const = 0;

    void setManager(ServiceManager * m) {
        manager = m;
    }
private:

};

#endif //GPU_LOAD_BALANCE_BASESERVICE_H