#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "../services/services.h"
#include "tuple"

class Messaging {
public:
    virtual void Init() = 0;
    virtual std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            std::tuple<int, int> target,
            int source) = 0;

    virtual std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int target,
            int source) = 0;

    virtual void finalize() = 0;

};

#endif //MESSAGING_H