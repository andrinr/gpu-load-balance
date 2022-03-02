#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orb.c"

static int COUNT = 1 << 18;

int main()
{
    // Init positions
    float p [COUNT][DIMENSIONS];

    for (int i = 0; i < COUNT; i++) {
        printf("%.2f percent initialized\n", i/(float)COUNT * 100);
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i][d] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    printf("Initialized \n");

    // Init tree
    struct Cell root = {
        .center = {0.0},
        .size = {1.0},
        .start = 0,
        .end = COUNT
    };
    
    orb(&root, p);

    printf("Done.\n");
    return 0;
}
