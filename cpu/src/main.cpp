#include <iostream>
#include <fstream>
#include <filesystem>
#include <blitz/array.h>
#include <chrono>

#include "comm/MPIMessaging.h"
#include "constants.h"
#include "cell.h"
#include "IO.h"
#include "orb.h"
#include "services.h"

using namespace std::chrono;

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

    // Init comm
    MPIMessaging mpiMessaging;
    std::cout << "Process " << mpiMessaging.rank << " processing " << count / 1000 << "K particles." << std::endl;
    std::cout << "Process " << mpiMessaging.rank << " starting task..." << std::endl;


    // Number of particles for current processor
    int N = floor(count / mpiMessaging.np);
    blitz::Array<float, 2> particles = IO::generateData(N, mpiMessaging.rank);

    // We add +1 due to heap storage order
    int nCells = nLeafCells * 2 + 1;
    blitz::Array<float*, 2> cellToParticle(nCells);
    Orb orb(particles, cellToParticle, nLeafCells);

    auto start = high_resolution_clock::now();
    if (mpiMessaging.rank == 0) {

        // root cell is at index 1
        blitz::Array<Cell, 1> cells(nCells);

        const float lowerInit = -0.5;
        const float upperInit = 0.5;

        float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
        float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

        Cell cell(1, nLeafCells, lower, upper);
        cells(1) = cell;

        Messaging::dispatchService(
                orb,
                Services::buildTree,
                cells.data(),
                1,
                nullptr,
                0,
                0);
    }
    else {
        Cell* empty_cells;
        int* results;
        while(true) {
            int id;
            Messaging::signalServiceId(id);
            if (id == -1) {
                break;
            };

            int nCells;
            int nResults;
            Messaging::signalDataSize(nCells);
            switch(id) {
                case 0:
                    Messaging::dispatchService(
                            orb,
                            &Services::count,
                            empty_cells,
                            nCells,
                            results,
                            nResults,
                            0);
                    break;
                case 1:
                    Messaging::dispatchService(
                            orb,
                            &Services::localReshuffle,
                            empty_cells,
                            nCells,
                            results,
                            nResults,
                            0);
                    break;
                default:
                    throw std::invalid_argument(
                            "Main.main: Service ID unknown.");

            }
        }
    }

    mpiMessaging.destroy();

    return 0;
}
