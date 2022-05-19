#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "tuple"

class ServiceManager;

class Messaging {
public:
    Messaging() {};
    virtual void dispatchService(ServiceManager * manager, int serviceID, void * rawInputData, void * rawOutputData) = 0;
    virtual void workService(ServiceManager * manager, void * rawOutputData) = 0;
    virtual void workService(ServiceManager * manager) = 0;
    virtual void finalize() = 0;
};

#endif //MESSAGING_H