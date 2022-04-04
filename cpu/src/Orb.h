#ifndef ORB_H // include guard
#define ORB_H

#include <mpi.h>
#include "Cell.h"
#include <tuple>
#include <blitz/array.h>   

class Orb {
public:
    blitz::Array<float, 2>* particles;
    blitz::Array<int, 2>* indexBounds;

    Cell* cells;
    int nCells;

    MPI_Datatype MPI_CUT;
    MPI_Datatype MPI_CELL;

    Orb(int rank, int np);
    void build(blitz::Array<float, 2> &particles);

private:
    int rank;
    int np;
    
    void swap(int a, int b);
    int reshuffleArray(int axis, int begin, int end, float split);
    int count(int axis, int start, int end, float cut);
    float findCut(Cell &cell, int axis, int begin, int end);
    void operative();
    void worker();
};

#endif //ORB_H