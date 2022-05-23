#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>
#include "../services/serviceManager.h"

class MPIMessaging : public Messaging {
public:
    MPIMessaging();
    void dispatchService(ServiceManager * manager,
            int serviceID,
            void *  inputBuffer,
            int inputBufferLength,
            void * outputBuffer,
            int outputBufferLength) const override;

    void workService(ServiceManager * manager, void * rawOutputData) const override;
    void workService(ServiceManager * manager) const override;
    void finalize() const override;

    int rank;
    int np;
};
#endif //MPI_MESSAGING_H