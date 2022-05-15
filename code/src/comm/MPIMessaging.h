#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>


class MPIMessaging : public Messaging {
public:
    virtual void Init() = 0;

    virtual std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            std::tuple<int, int> target,
            int source) = 0;

    static std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int target,
            int source);

    virtual void finalize() = 0;

    static inline int rank;
    static inline int np;

    static inline MPI_Datatype MPI_CELL;
};
#endif //MPI_MESSAGING_H