#include <iostream>
#include <fstream>
#include "Orb.h"
#include "constants.h"
#include <blitz/array.h>   
#include <chrono>
using namespace std::chrono;

float r01() {
    return (float)(rand())/(float)(RAND_MAX);
}

int main(int argc, char** argv) {

    if (strlen(argv[1]) == 0) {
        return 1; // empty string
    }
    char* p1;
    char* p2;
    errno = 0; // not 'int errno', because the '#include' already defined it
    long arg1 = strtol(argv[1], &p1, 10);
    long arg2 = strtol(argv[2], &p2, 10);
    if (*p1 != '\0' || *p2 != '\0' || errno != 0) {
        return 1; // In main(), returning non-zero means failure
    }

    if (arg1 < INT_MIN || arg1 > INT_MAX || arg2 < INT_MIN || arg2 > INT_MAX) {
        return 1;
    }
    int count = arg1 * 1000;
    int nCells = arg2;

    int rank, np;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = floor(count / np);
    std::cout << "Process " << rank << " processing " << N / 1000 << "K particles." << std::endl;

    // Init positions
    blitz::Array<float, 2> p(N, DIMENSIONS + 1);
    p = 0;
    
    srand(rank);
    for (int i = 0; i < p.rows(); i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            p(i,d) = (r01()-0.5)*(r01()-0.5);
        }
        p(i,3) = 0.;
    }

    std::cout << "Process " << rank << " building tree..." << std::endl;

    auto start = high_resolution_clock::now();
    Orb orb(rank, np, p, nCells);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Process " << rank << " duration: " << duration.count() / 1000 << " ms" << std::endl;

    printf("Done.\n");       

    
    MPI_Finalize();     

    std::fstream file( "../data/splitted" + std::to_string(rank) + ".dat", std::fstream::out);
   
    for (int i = 0; i < N; i += 64){
        file << p(i,0) << "\t" << p(i,1) << "\t" << p(i,2) << "\t" << p(i,3) << std::endl;
    }

    file.close();                           

    return 0;
}
