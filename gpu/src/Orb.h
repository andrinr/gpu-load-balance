#ifndef ORB_H // include guard
#define ORB_H

#include "Cell.h"
#include <tuple>
#include <blitz/array.h>   

class Orb {
public:
    blitz::Array<float, 2>* h_particles;
    Cell* cells;

    Orb();
    void build(blitz::Array<float, 2> &h_particles);

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