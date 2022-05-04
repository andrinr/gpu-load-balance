#ifndef COMM_H // include guard
#define COMM_H

#include "../services.h"
template <class T>;

class Range {
    int beginRank,
    int endRank
};

template <class InData>;
template <class OutData>;

class Comm {
public:
    void Comm();

    OutData static dispatchService(
            OutData (*func)(InData),
            InData* inData,
            int nInData,
            OutData* outData,
            int nOutData,
            int source);

    void destroy();

};

#endif //COMM_H