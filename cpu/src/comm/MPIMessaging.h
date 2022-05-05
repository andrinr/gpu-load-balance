#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>

class MPIMessaging : Messaging {
public:
    MPIMessaging();
    int rank;
    int np;

private:
    MPI_Datatype MPI_CELL;
};
#endif //MPI_MESSAGING_H