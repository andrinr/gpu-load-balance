#include <iostream>
#include "Orb.h"
#include "constants.h"
#include <blitz/array.h>   

int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Process " << rank + 1 << " out of " << np << std::endl;

    // Init positions
    blitz::Array<float, 2> p(COUNT, 3);
    p = 0;
    
    printf("Initializing... \n");

    srand(rank);
    for (int i = 0; i < COUNT; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p(i,d) = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    printf("Initiate ORB... \n");
    
    // call orb()
    Orb orb(rank, np);

    printf("Build ORB... \n");

    orb.build(p);

    printf("Done.\n");       

    
    MPI_Finalize();                                       

    return 0;
}
