#ifndef SERVICES_H
#define SERVICES_H
#include <blitz/array.h>
#include "../cell.h"
#include "../orb.h"

enum ServiceIDs {
    countLeftService,
    countService,
    buildTreeService,
    localReshuffleService,
    terminateService
};

class Services {
public:
    static void count(Orb& orb, Cell* cells, int* results, int n);
    static void countLeft(Orb& orb, Cell* cells, int* results, int n);
    static void localReshuffle(Orb& orb, Cell* cells, int* results, int n);
    static void buildTree(Orb& orb, Cell* cell, int* results, int n);
};

#endif //SERVICES_H
