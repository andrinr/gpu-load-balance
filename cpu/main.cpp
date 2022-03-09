#include "orb.cpp"

static const int COUNT = 1 << 20;

int main()
{
    
    // Init positions
    float* p = new float[COUNT * DIMENSIONS]{0.0};

    printf("Initializing... \n");

    for (int i = 0; i < COUNT; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p[i * DIMENSIONS + d] = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    printf("Computing ORB... \n");

    // Init tree
    struct Cell root = {
        .cornerA = {-0.5, -0.5, -0.5},
        .cornerB = {0.5, 0.5, 0.5},
        .start = 0,
        .end = COUNT
    };
    
    orb(&root, p, 256);

    printf("Done.\n");                                              
    

    //mpi::finallize();
    return 0;
}
