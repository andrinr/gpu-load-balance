#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orb.c"

static int COUNT = 1 << 10;

void testSplit()
{
    float p [COUNT][DIMENSIONS];
    for (int i = 0; i < COUNT; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i][d] = i / (float) COUNT -0.5;
        }
    }

    float split = findSplit(p, 0, 0, COUNT, -0.5, 0.5);

    printf("%.3f", split);
}

int main()
{
    testSplit();
    return 0;
}
