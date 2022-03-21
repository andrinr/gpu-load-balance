#ifndef ORB_H // include guard
#define ORB_H

#include "Cell.h"
#include <tuple>

class Orb {
public:
    float* particles;
    Cell* cells;
    MPI_Datatype mpi_cut_type;

    Orb(float* particles);
    void build();

private:
    int rank;
    int np;
    
    void reshuffleArray(int axis, int begin, int end, float split);
    int count(int axis, int start, int end, float cut);
    std::tuple<float, int> findCut(int axis, int begin, int end, float left, float right);
    void operative();
    void worker();
};

#endif //ORB_H