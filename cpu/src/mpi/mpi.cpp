#include <mpi.h>

namespace mpi {
    std::tuple<int, int> init() {
        int rank, np, i;
        MPI_Init(NULL, NULL); 

        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::cout << "Processor ID: " << rank << " Number of processes: " << np << std::endl;

        return {rank, np};
    }

    void finallize() {
        MPI_Finalize();
    }

}
