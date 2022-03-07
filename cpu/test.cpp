#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orb.cpp"
#include <assert.h>

static int COUNT_LARGE = 1 << 10;
static int COUNT_SMALL = 1 << 5;

void testOrb()
{    
    float p [COUNT_SMALL][DIMENSIONS];
    for (int i = 0; i < COUNT_SMALL; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i][d] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    // Init tree
    struct Cell root = {
        .center = {0.0, 0.0, 0.0},
        .size = {1.0, 1.0, 1.0},
        .start = 0,
        .end = COUNT_SMALL
    };
    
    orb(&root, p, 2);
}

void testFindMaxIndex() {
    float p [5] = {0.1, 0.9, 0.2, 0.3, -0.2};

    int index = findMaxIndex(p);
    assert(index == 1);
}

void testSplit()
{
    float p [COUNT_LARGE][DIMENSIONS];
    for (int i = 0; i < COUNT_LARGE; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i][d] = i / (float) COUNT_LARGE -0.5;
        }
    }

    float split = findSplit(p, 0, 0, COUNT_LARGE, -0.5, 0.5);

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
    testOrb();
    return 0;
}
