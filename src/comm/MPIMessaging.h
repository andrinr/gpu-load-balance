#ifndef MPI_MESSAGING_H // include guard
#define MPI_MESSAGING_H
#include "messaging.h"
#include <mpi.h>


class MPIMessaging : public Messaging {
public:
    MPIMessaging();

    void Init() override;
    std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            std::tuple<int, int> target,
            int source) override;

    std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int target,
            int source) override;

    void finalize() override;
    int rank;
    int np;

    MPI_Datatype MPI_CELL;
};
#endif //MPI_MESSAGING_H