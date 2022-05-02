#ifndef SERVICES_H
#define SERVICES_H
#include <blitz/array.h>
#include "cell.h"

struct InCount {
   blitz::Array<Cell, 1> cells, // slice of cells
   int begin,
   int end
};

struct InControl {
    Orb& orb,
    Comm comm,
    blitz::Array<Cell, 1> cells,
    int nLeafCells
};

struct InControl {

};

class Services {
    static blitz::Array<int, 1> count(InCount& data);
    static OutData control(InData data);

};

#endif //SERVICES_H
