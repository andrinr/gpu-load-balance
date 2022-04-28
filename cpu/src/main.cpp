#include <iostream>
#include <fstream>
#include <filesystem>
#include "Orb.h"
#include "IO.h"
#include "constants.h"
#include "services.h"
#include <blitz/array.h>   
#include <chrono>
#include "mpi-comm.h"

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
    long arg1 = strtol(argv[1], &p1, 10);
    long arg2 = strtol(argv[2], &p2, 10);

    int count = arg1 * 1000;
    int nLeafCells = arg2;

    int N = floor(count / np);
    blitz::Array<float, 2> particles = IO::generateData(N);

    MPI_Comm mpiComm;

    std::cout << "Process " << mpiComm.rank << " processing " << N / 1000 << "K particles." << std::endl;


    Cell cells[nLeafCells + 1];

    if (mpiComm.rank == 0) {
        const float lowerInit = -0.5;
        const float upperInit = 0.5;

        float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
        float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

        Cell cell(0, -1, domainCount, lower, upper);
        cells[1] = cell;

        Services::operate();
    }
    else {
        Services::work();
    }

    std::cout << "Process " << mpiComm.rank << " building tree..." << std::endl;

    auto start = high_resolution_clock::now();
    Orb orb(rank, np, p, nCells);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Process " << mpiComm.rank << " duration: " << duration.count() / 1000 << " ms" << std::endl;

    int l_duration = duration.count() / 1000.0;
    int g_duration;

    MPI_Reduce(&l_duration, &g_duration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


    std::cout << "Process " << mpiComm.rank << " done. "  << std::endl;



    return 0;
}
