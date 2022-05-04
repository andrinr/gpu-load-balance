#include <tasks.h>
#include "comm/comm.h"
#include <math.h>
#include <blitz/array.h>
#include "services.h"

void Tasks::operate(
        Orb& orb,
        Comm comm,
        blitz::Array<Cell, 1> cells,
        int nLeafCells
    ){


    
}

void Tasks::work(Orb& orb, MPI_Comm comm) {
    int n;
    Cell cells[nLeafCells * 2];

    while(true) {
        MPI_Comm::dispatchWork(&n, &cells);

        // Check weather still work to do
        if (n == -1) {
            break;
        }

        int* count = orb.count();

        MPI_Comm::concludeWork(n, count);
    }
}

