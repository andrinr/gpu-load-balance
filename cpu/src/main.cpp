#include "orb.cpp"
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
    
    orb(p, 8);

    printf("Done.\n");                                              
    
    //mpi::finallize();
    return 0;
}
