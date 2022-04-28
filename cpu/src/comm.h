#ifndef COMM_H // include guard
#define COMM_H


class Comm {
public:
    virtual void init();

    virtual void static broadcast(int& nCells, Cell* cells);

    virtual void static updateCut(int nCells, int* axis, float* pos);

    virtual void static reduceCut(int* count);

    virtual void static destroy();

};

#endif //COMM_H