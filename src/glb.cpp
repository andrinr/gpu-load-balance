#include <iostream>
#include <fstream>
#include <filesystem>
#include <blitz/array.h>
#include <chrono>
#include <cstdlib>

#include "comm/MPIMessaging.h"
#include "constants.h"
#include "cell.h"
#include "IO.h"
#include "orb.h"
#include "services/services.h"

using namespace std::chrono;

int main(int argc, char** argv) {

    const double inputValue = std::stod(argv[1]);

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

    // Init comm
    MPIMessaging mpiMessaging;
    mpiMessaging.Init();

    std::cout << "Process " << mpiMessaging.rank << " processing " << count / 1000 << "K particles." << std::endl;
    std::cout << "Process " << mpiMessaging.rank << " starting task..." << std::endl;

    int N = (count+mpiMessaging.np-1) / mpiMessaging.np ;
    if (N * mpiMessaging.rank >= count) N = 0;
    else if (N * (mpiMessaging.rank+1) >= count) N = count - mpiMessaging.rank*N;

    // Set row major -> can enable AVX
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;

    blitz::Array<float, 2> particles = IO::generateData(N, mpiMessaging.rank);

    // We add +1 due to heap storage order
    int nCells = nLeafCells * 2 + 1;
    blitz::Array<int, 2> cellToParticle(nCells);
    cellToParticle(0,0) = 0;
    cellToParticle(0,1) = N-1;
    Orb orb(particles, cellToParticle, nLeafCells);

    auto start = high_resolution_clock::now();
    if (mpiMessaging.rank == 0) {

        // root cell is at index 1
        blitz::Array<Cell, 1> cells(nCells);

        const float lowerInit = -0.5;
        const float upperInit = 0.5;

        float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
        float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

        Cell root(1, nLeafCells, lower, upper);
        CellHelpers::setCutAxis(root);
        CellHelpers::setCutMargin(root);
        cells(0) = root;

        int* results;
        mpiMessaging.dispatchService(
                orb,
                buildTreeService,
                cells.data(),
                1,
                results,
                0,
                0,
                0);
    }
    else {
        Cell* emptyCells;
        int* results;
        int nResults;
        ServiceIDs id;
        bool status = true;
        while(status) {
            std::tie(status, results) = mpiMessaging.dispatchService(
                    orb,
                    id,
                    emptyCells,
                    nCells,
                    results,
                    nResults,
                    std::make_tuple(1, mpiMessaging.np-1),
                    0);
        }
    }

    mpiMessaging.finalize();

    return 0;
}
