#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orb.c"

static int COUNT = 1 << 25;

int main()
{
    // Init positions
    float (*p) [COUNT][DIMENSIONS] = malloc(sizeof(float[COUNT][DIMENSIONS]));

    printf("Initializing... \n");

    for (int i = 0; i < COUNT; i++) {
        
        //printf("%.2f percent initialized\n", i/(float)COUNT * 100);

        for (int d = 0; d < DIMENSIONS; d++) {
            (*p)[i][d] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    printf("Computing ORB... \n");

    // Init tree
    struct Cell root = {
        .center = {0.0, 0.0, 0.0},
        .size = {1.0, 1.0, 1.0},
        .start = 0,
        .end = COUNT
    };
    
    orb(&root, *p, 256);

    free(p);
    
    printf("Done.\n");
    return 0;
}
