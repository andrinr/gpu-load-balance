#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "../services/services.h"
#include "tuple"
#include "../services/baseService.h"
#include "../services/serviceManager.h"

class Messaging {
public:
    Messaging();
    void dispatchService(ServiceManager * manager, int serviceID, void * rawInputData, void * rawOutputData);
    void workService(ServiceManager * manager, void * rawOutputData);
    void workService(ServiceManager * manager);
    virtual void finalize() = 0;
};

#endif //MESSAGING_H