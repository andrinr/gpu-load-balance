#include <iostream>
#include "Orb.h"
#include "constants.h"
#include <blitz/array.h>   

int main()
{
    // Init positions
    blitz::Array<float, 2> p(COUNT, 3);
    p = 0;
    //blitz::allocateArrays(blitz::shape(COUNT,3), p);

    printf("Initializing... \n");

    // Set a seed
    for (int i = 0; i < COUNT; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p(i,d) = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
    }

    printf("Computing ORB... \n");
    
    Orb orb = Orb(p);
    orb.build();

    printf("Done.\n");                                              

    return 0;
}
