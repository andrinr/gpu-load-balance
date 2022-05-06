#ifndef SERVICES_H
#define SERVICES_H
#include <blitz/array.h>
#include "cell.h"
#include "orb.h"

enum ServiceIDs {
    countLeftService,
    countService,
    buildTreeService,
    localReshuffleService,
    terminateService
};

class Services {
public:
    static int* count(Orb& orb, Cell* cells, int n);
    static int* countLeft(Orb& orb, Cell* cells, int n);
    static int* localReshuffle(Orb& orb, Cell* cells, int n);
    static int* buildTree(Orb& orb, Cell* cell, int n);
};

#endif //SERVICES_H
