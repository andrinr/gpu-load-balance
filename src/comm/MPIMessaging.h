#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>
#include "../services/serviceManager.h"

class MPIMessaging : public Messaging {
public:
    MPIMessaging();
    void dispatchService(ServiceManager * manager, int serviceID, void * rawInputData, void * rawOutputData);
    void workService(ServiceManager * manager, void * rawOutputData);
    void workService(ServiceManager * manager);

    void finalize() override;
    int rank;
    int np;
};
#endif //MPI_MESSAGING_H