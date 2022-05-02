#ifndef ORB_H // include guard
#define ORB_H

#include <mpi.h>
#include "Cell.h"
#include <tuple>
#include <vector>
#include <blitz/array.h>   

class Orb {
public:
    blitz::Array<float, 2> particles;
    blitz::Array<int, 2> cellToParticle

    Orb(blitz::Array<float, 2> &p, blitz::Array<float*, 2> &cToP);
    int count(int axis, int start, int end, float cut, int stride);

private:
    
    void swap(int a, int b);
    void assign(int begin, int end, int id);
    int reshuffleArray(int axis, int begin, int end, float split);
    float findCut(Cell &cell, int axis, int begin, int end);
    void operative();
    void worker();
};

#endif //ORB_H