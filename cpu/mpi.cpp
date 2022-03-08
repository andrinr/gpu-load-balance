#include <mpi.h>

int init() {
    int myid, numprocs, i;
    MPI_Init(NULL, NULL); 

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    std::cout << "Processor ID: " << myid << " Number of processes: " << numprocs << std::endl;

    return myid;
}

void finallize() {
    MPI_Finalize();
}

int main(int argc, char *argv[]){
    init();
    finallize();
    return 0;
}
