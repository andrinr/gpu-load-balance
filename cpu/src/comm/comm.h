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

    virtual OutData static dispatchService(
            InData (*func)(OutData),
            InData inData,
            int nInData,
            OutData outData,
            int nOutData,
            int source);

    virtual void static concludeService(EServices serviceID, int& n, T data, int target);

    virtual void static dispatchWork(int& n, blitz::Array<Cell, 1> cells);

    virtual void static concludeWork(int&n, int* count);

    virtual void destroy();

};

#endif //COMM_H