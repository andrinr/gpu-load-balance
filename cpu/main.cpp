#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "orb.cpp"

static const int COUNT = 1 << 20;

int main()
{
    // Init positions
    std::vector<std::vector<float>> p;
    //p.reserve(COUNT);

    printf("Initializing... \n");

    for (int i = 0; i < COUNT; i++) {
        
        //printf("%.2f percent initialized\n", i/(float)COUNT * 100);
        std::vector<float> pos;
        for (int d = 0; d < DIMENSIONS; d++) {
            pos.push_back((float)rand()/(float)(RAND_MAX) - 0.5);
        }
        p.push_back(pos);
    }

    printf("Computing ORB... \n");

    // Init tree
    struct Cell root = {
        .center = {0.0, 0.0, 0.0},
        .size = {1.0, 1.0, 1.0},
        .start = 0,
        .end = COUNT
    };
    
    orb(&root, p, 256);
    
    printf("Done.\n");
    return 0;
}
