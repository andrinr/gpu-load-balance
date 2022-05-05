#ifndef SERVICES_H
#define SERVICES_H
#include <blitz/array.h>
#include "cell.h"
#include "orb.h"

class Services {
public:
    static int* count(Orb& orb, Cell* cells, int n);
    static int* countLeft(Orb& orb, Cell* cells, int n);
    static int* localReshuffle(Orb& orb, Cell* cells, int n);
    static int* buildTree(Orb& orb, Cell* cell, int n);
    static int* findCuts(Orb& orb, Cell* cells, int n);
};

#endif //SERVICES_H
