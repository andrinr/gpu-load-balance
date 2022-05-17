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

    // report version
    std::cout << argv[0] << " Version " << glb_VERSION_MAJOR << "." << glb_VERSION_MINOR << std::endl;

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
    MPIMessaging::Init();
    std::cout << "Process " << MPIMessaging::rank << " processing " << count / 1000 << "K particles." << std::endl;
    std::cout << "Process " << MPIMessaging::rank << " starting task..." << std::endl;

    int N = (count+MPIMessaging::np-1) / MPIMessaging::np ;
    if (N * MPIMessaging::rank >= count) N = 0;
    else if (N * (MPIMessaging::rank+1) >= count) N = count - MPIMessaging::rank*N;

    // Set row major -> can enable AVX
    blitz::GeneralArrayStorage<2> storage;
    storage.ordering() = 0,1;
    storage.base() = 0, 0;
    storage.ascendingFlag() = true, true;

    blitz::Array<float, 2> particles = IO::generateData(N, MPIMessaging::rank);

    // We add +1 due to heap storage order
    int nCells = nLeafCells * 2 + 1;
    blitz::Array<int, 2> cellToParticle(nCells);
    cellToParticle(0,0) = 0;
    cellToParticle(0,1) = N-1;
    Orb orb(particles, cellToParticle, nLeafCells);

    auto start = high_resolution_clock::now();
    if (MPIMessaging::rank == 0) {

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
        MPIMessaging::dispatchService(
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
            std::tie(status, results) = MPIMessaging::dispatchService(
                    orb,
                    id,
                    emptyCells,
                    nCells,
                    results,
                    nResults,
                    std::make_tuple(1, MPIMessaging::np-1),
                    0);
        }
    }

    MPIMessaging::finalize();

    return 0;
}
