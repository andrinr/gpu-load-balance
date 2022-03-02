#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orb.c"
#include <assert.h>

static int COUNT = 1 << 10;

void testFindMaxIndex() {
    float p [5] = {0.1, 0.9, 0.2, 0.3, -0.2};

    int index = findMaxIndex(p);
    assert(index == 1);
}

void testSplit()
{
    float p [COUNT][DIMENSIONS];
    for (int i = 0; i < COUNT; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i][d] = i / (float) COUNT -0.5;
        }
    }

    float split = findSplit(p, 0, 0, COUNT, -0.5, 0.5);

    assert(split == 0.0);
}

void testReshuffleArray() {
    float p [6][3] = {
        {-0.5, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {4.5, 0.0, 0.0},
        {-2.5, 0.0, 0.0},
        {3.5, 0.0, 0.0},
        {-10.0, 0.0, 0.0},
    };

    int mid = reshuffleArray(p,  0, 0, 6, 0.0);

    assert(mid == 3);
}

int main()
{
    testSplit();
    testFindMaxIndex();
    testReshuffleArray();
    return 0;
}
