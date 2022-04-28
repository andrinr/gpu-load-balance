#include <services.h>
#include <services.h>
#include "mpi-comm.h"

void Services::operate()


    
}

void Services::work(Orb& orb, Cell cells*) {
    int n;
    Cell cells[nLeafCells * 2];

    while(true) {
        MPI_Comm::dispatchWork(&n, &cells);

        if (n == -1) {
            break;
        }

        int* count = orb.count();

        MPI_Comm::concludeWork(n, count);
    }
}