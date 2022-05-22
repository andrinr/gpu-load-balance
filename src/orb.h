#ifndef ORB_H // include guard
#define ORB_H

#include "cell.h"
#include <tuple>
#include <vector>
#include <memory>
#include <blitz/array.h>   

class Orb {
public:
    std::unique_ptr<blitz::Array<float, 2>> particles;
    std::unique_ptr<blitz::Array<int, 2>> cellToParticle;
    int nLeafCells;

    Orb(    std::unique_ptr<blitz::Array<float, 2>> p,
            std::unique_ptr<blitz::Array<int, 2>> cToP,
            int nLeafCells);

    void swap(int a, int b);
    void assign(int begin, int end, int id);

};

#endif //ORB_H