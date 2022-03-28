#ifndef ORB_H // include guard
#define ORB_H

#include <mpi.h>
#include "Cell.h"
#include <tuple>
#include <blitz/array.h>   

class Orb {
public:
    blitz::Array<float, 2>* particles;
    Cell* cells;
    MPI_Datatype mpi_cut_type;

    Orb();
    void build(blitz::Array<float, 2> &particles);

private:
    int rank;
    int np;

    static const int TAG_TREE_STATUS = 0;
    static const int TAG_CUT_STATUS = 1;
    static const int TAG_CUT_DATA = 2;
    static const int TAG_CUT_COUNT = 3;
    static const int TAG_PARTICLES = 4;
    
    void reshuffleArray(int axis, int begin, int end, float split);
    int count(int axis, int start, int end, float cut);
    std::tuple<float, int> findCut(int axis, int begin, int end, float left, float right);
    void operative();
    void worker();
};

#endif //ORB_H