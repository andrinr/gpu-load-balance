#include <iostream>
#include <fstream>
#include <filesystem>
#include "Orb.h"
#include "IO.h"
#include "cell.h"
#include "constants.h"
#include "tasks.h"
#include <blitz/array.h>   
#include <chrono>
#include "communication/mpi-comm.h"

using namespace std::chrono;

float r01() {
    return (float)(rand())/(float)(RAND_MAX);
}

int main(int argc, char** argv) {

    // read params
    if (strlen(argv[1]) == 0) {
        return 1; // empty string
    }
    char* p1;
    char* p2;
    long arg1 = strtol(argv[1], &p1, 10);
    long arg2 = strtol(argv[2], &p2, 10);

    int count = arg1 * 1000;
    int nLeafCells = arg2;

    // Number of particles for current processor
    int N = floor(count / np);
    blitz::Array<float, 2> particles = IO::generateData(N);

    // We add +1 due to heap storage order
    int nCells = nLeafCells * 2 + 1;
    // root cell is at index 1
    blitz::Array<Cell, 1> cells(nCells);
    blitz::Array<int, 2> cellToParticle(nCells);
    Orb orb(particles, cellToParticle);

    // Init comm
    MPI_Commparticles,  mpiComm;
    std::cout << "Process " << mpiComm.rank << " processing " << N / 1000 << "K particles." << std::endl;
    std::cout << "Process " << mpiComm.rank << " starting task..." << std::endl;

    auto start = high_resolution_clock::now();
    if (mpiComm.rank == 0) {
        const float lowerInit = -0.5;
        const float upperInit = 0.5;

        float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
        float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

        Cell cell(domainCount, lower, upper);
        cells(1) = cell;

        Tasks::operate(&orb, nLeafCells);
    }
    else {
        Tasks::work(&orb);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Process " << mpiComm.rank << " duration: " << duration.count() / 1000 << " ms" << std::endl;

    int l_duration = duration.count() / 1000.0;
    int g_duration;

    MPI_Reduce(&l_duration, &g_duration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


    std::cout << "Process " << mpiComm.rank << " done. "  << std::endl;

    mpiComm.destroy();

    return 0;
}
