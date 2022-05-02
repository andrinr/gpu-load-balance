#ifndef ORB_H // include guard
#define ORB_H

#include <mpi.h>
#include "Cell.h"
#include <tuple>
#include <vector>
#include <blitz/array.h>   

class Orb {
public:
    std::vector<int> cellBegin;
    std::vector<int> cellEnd;

    MPI_Datatype MPI_CUT;
    MPI_Datatype MPI_CELL;

    Orb(int rank, int np, blitz::Array<float, 2> &p, int d);

private:
    blitz::Array<float, 2> particles; //JDP
    int rank;
    int np;
    int domainCount;
    
    void swap(int a, int b);
    void assign(int begin, int end, int id);
    int reshuffleArray(int axis, int begin, int end, float split);
    int count(int axis, int start, int end, float cut, int stride);
    float findCut(Cell &cell, int axis, int begin, int end);
    void operative();
    void worker();
};

#endif //ORB_H