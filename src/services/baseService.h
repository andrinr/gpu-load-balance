#ifndef CODE_BASESERVICE_H
#define CODE_BASESERVICE_H

#include "../comm/messaging.h"
#include "../orb.h"
#include "vector"

class ServiceManager;
class Messaging;

class BaseService {
public:
    const int serviceID = -1;
    ServiceManager * manager;

    void run(void * rawInputData, void * rawOutputData) = 0;
    int getNInputBytes(void * inputPtr) const = 0;
    int getNOutputBytes(void * outputPtr) const = 0;

    void setManager(ServiceManager * m) {
        manager = m;
    }
private:

};

#endif //CODE_BASESERVICE_H