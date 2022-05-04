#include <iostream>
#include <stack>
#include "orb.h"

Orb::Orb(blitz::Array<float, 2> &p,
         blitz::Array<float*, 2> &cToP,
         int nLeafCells)
    : particles(p), cellToParticle(cToP), nLeafCells(n) {

    int N = particles.nrows();

    cellToParticle(1, 0) = particles(0, 0);
    cellToParticle(1, 1) = particles(N - 1, 0);
}

void Orb::assign(int begin, int end, int id) {
    for (int j = begin; j < end; j++) {
        particles(j, 3) = id;
    }
}

void Orb::swap(int a, int b) {
    for (int d = 0; d < DIMENSIONS + 1; d++) {
        float tmp = particles(a, d);
        particles(a, d) = particles(b, d);
        particles(b, d) = tmp;
    }
}

