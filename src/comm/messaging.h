#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "tuple"

class ServiceManager;

class Messaging {
public:
    Messaging() {};
    virtual void dispatchService(
            ServiceManager * manager,
            int serviceID,
            void * rawInputData,
            void * rawOutputData) const = 0;

    virtual void dispatchService(ServiceManager * manager,
            int serviceID,
            void *  inputBuffer,
            int inputBufferLength,
            void * outputBuffer,
            int outputBufferLength) const = 0;

    virtual void workService(ServiceManager * manager, void * rawOutputData) const = 0;
    virtual void workService(ServiceManager * manager) const = 0;
    virtual void finalize() const = 0;
};

#endif //MESSAGING_H