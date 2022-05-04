#ifndef SERVICES_H
#define SERVICES_H
#include <blitz/array.h>
#include "cell.h"
#include "orb.h"
#include "comm/mpi-comm.h"

struct InControl {
    Orb& orb,
    Comm comm,
    blitz::Array<Cell, 1> cells,
    int nLeafCells
};

class Services {
public:
    Services(Orb& orb);
    int* count(Cell* cells, int n);
    int* countLeft(Cell* cells, int n);
    int* localReshuffle(Cell* cells, int n);
    int* buildTree(Cell* cell, int n);
    int* findCuts(Cell* cells, int n);

private:
    Orb orb;
};

#endif //SERVICES_H
