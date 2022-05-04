#ifndef ORB_H // include guard
#define ORB_H

#include <mpi.h>
#include "cell.h"
#include <tuple>
#include <vector>
#include <blitz/array.h>   

class Orb {
public:
    blitz::Array<float, 2> particles;
    blitz::Array<int, 2> cellToParticle;
    int nLeafCells;

    Orb(
            blitz::Array<float, 2> &p,
            blitz::Array<float*, 2> &cToP,
            int nLeafCells);

private:
    
    void swap(int a, int b);
    void assign(int begin, int end, int id);

};

#endif //ORB_H