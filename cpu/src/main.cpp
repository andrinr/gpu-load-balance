#include <iostream>
#include <fstream>
#include "Orb.h"
#include "constants.h"
#include <blitz/array.h>   
#include <chrono>
using namespace std::chrono;
int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = floor(COUNT / np);
    std::cout << "Process " << rank << " processing " << N / 1000 << "K particles." << std::endl;

    // Init positions
    blitz::Array<float, 2> p(N, DIMENSIONS + 1);
    p = 0;
    
    srand(rank);
    for (int i = 0; i < p.rows(); i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p(i,d) = (float)rand()/(float)(RAND_MAX) - 0.5;
        }
        p(i,3) = 0.;
    }

    // call orb()
    Orb orb(rank, np);

    std::cout << "Process " << rank << " building tree..." << std::endl;

    auto start = high_resolution_clock::now();
    orb.build(p);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Process " << rank << " duration: " << duration.count() / 1000 << " ms" << std::endl;

    printf("Done.\n");       

    
    MPI_Finalize();     

    std::fstream file( "../data/splitted" + std::to_string(rank) + ".dat", std::fstream::out);
   
    for (int i = 0; i < N; i++){
        file << p(i,0) << "\t" << p(i,1) << "\t" << p(i,2) << "\t" << p(i,3) << std::endl;
    }

    file.close();                           

    return 0;
}
