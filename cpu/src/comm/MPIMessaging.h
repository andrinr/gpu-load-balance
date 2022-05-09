#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>


class MPIMessaging : public Messaging {
public:
    static void Init();
    static std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            std::tuple<int, int> target,
            int source);

    static std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int target,
            int source);

    static void signalDataSize(int size);
    static void signalServiceId(int flag);
    static void destroy();

    static inline int rank;
    static inline int np;

    static inline MPI_Datatype MPI_CELL;
};
#endif //MPI_MESSAGING_H