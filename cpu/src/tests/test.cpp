#include "orb.cpp"
#include <assert.h>

static int COUNT_LARGE = 1 << 10;
static int COUNT_SMALL = 1 << 5;

void testOrb()
{    
    float* p = new float[COUNT_SMALL * DIMENSIONS]{0.0};
    for (int i = 0; i < COUNT_SMALL; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i * DIMENSIONS + d] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }
    
    orb(p, 3);
}

void testSplit()
{
    float* p = new float[COUNT_LARGE * DIMENSIONS]{0.0};
    for (int i = 0; i < COUNT_LARGE; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i * DIMENSIONS + d] = i / (float) COUNT_LARGE -0.5;
        }
    }

    float split = findCut(p, 0, 0, COUNT_LARGE, -0.5, 0.5);

    assert(split == 0.0);
}

void testReshuffleArray() {
    float* p = new float[6 * DIMENSIONS] {
        -0.5, 0.0, 0.0,
        1.5, 0.0, 0.0,
        4.5, 0.0, 0.0,
       -2.5, 0.0, 0.0,
        3.5, 0.0, 0.0,
        -10.0, 0.0, 0.0,
    };

    int mid = reshuffleArray(p,  0, 0, 6, 0.0);

    assert(mid == 3);
}

int main()
{
    testSplit();
    testReshuffleArray();
    testOrb();
    return 0;
}
