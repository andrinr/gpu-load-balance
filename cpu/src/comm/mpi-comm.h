#ifndef MPI_COMM_H // include guard
#define MPI_COMM_H
#include "comm.h"
#include <mpi.h>

class MPIComm : public Comm {
public:
    MPIComm();
    int rank;
    int np;

private:
    MPI_Datatype MPI_CELL;
};
#endif //MPI_COMM_H