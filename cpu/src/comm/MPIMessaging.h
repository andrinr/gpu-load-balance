#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>

class MPIMessaging : public Messaging {
public:
    static void Init();
    static int* dispatchService(
            Orb& orb,
            int *(*func)(Orb&, Cell *, int),
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int source);

    static void signalDataSize(int size);
    static void signalServiceId(int flag);
    static void destroy();

    static int rank;
    static int np;

private:
    static MPI_Datatype MPI_CELL;
};
#endif //MPI_MESSAGING_H