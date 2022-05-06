#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"
#include "../services.h"
#include "tuple"

class Messaging {
public:
    static void Init();
    static std::tuple<bool, int*> dispatchService(
            Orb& orb,
            ServiceIDs id,
            Cell* cells,
            int nCells,
            int* results,
            int nResults,
            int source);

    static void signalDataSize(int size);
    static void signalServiceId(int flag);

    static void destroy();

};

#endif //MESSAGING_H