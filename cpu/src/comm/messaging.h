#ifndef MESSAGING_H // include guard
#define MESSAGING_H
#include "../cell.h"
#include "../orb.h"

class Messaging {
public:
    static void Init();
    static int* dispatchService(
            Orb& orb,
            int *(*func)(Orb&, Cell *, int),
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