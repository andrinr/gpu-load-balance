#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "../services/services.h"
#include "tuple"

class ServiceManager;

class Messaging {
public:
    virtual void dispatchService(ServiceManager * manager, int serviceID, void * rawInputData, void * rawOutputData) = 0;
    virtual void workService(ServiceManager * manager, void * rawOutputData) = 0;
    virtual void workService(ServiceManager * manager);
    virtual void finalize() = 0;
};

#endif //MESSAGING_H