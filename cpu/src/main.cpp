#include <iostream>
#include <fstream>
#include <filesystem>
#include "orb.h"
#include "IO.h"
#include "cell.h"
#include "constants.h"
#include "services.h"
#include <blitz/array.h>   
#include <chrono>
#include "comm/mpi-comm.h"

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
    MPIComm mpiComm;
    std::cout << "Process " << mpiComm.rank << " processing " << count / 1000 << "K particles." << std::endl;
    std::cout << "Process " << mpiComm.rank << " starting task..." << std::endl;


    // Number of particles for current processor
    int N = floor(count / mpiComm.np);
    blitz::Array<float, 2> particles = IO::generateData(N, mpiComm.rank);

    // We add +1 due to heap storage order
    int nCells = nLeafCells * 2 + 1;
    // root cell is at index 1
    blitz::Array<Cell, 1> cells(nCells);
    blitz::Array<int, 2> cellToParticle(nCells);
    Orb orb(particles, cellToParticle, nLeafCells);


    auto start = high_resolution_clock::now();
    if (mpiComm.rank == 0) {
        const float lowerInit = -0.5;
        const float upperInit = 0.5;

        float lower[DIMENSIONS] = {lowerInit, lowerInit, lowerInit};
        float upper[DIMENSIONS] = {upperInit, upperInit, upperInit};

        Cell cell(domainCount, lower, upper);
        cells(1) = cell;

        mpiComm.dispatchService(Services::buildTree, cells, 1, nullptr, 0, 0);
    }
    else {
        Cell* cells;
        while(true) {
            int id;
            mpiComm.signalServiceId(&id);
            if (id == -1) {
                break;
            };

            int size;
            mpiComm.signalDataSize(&size);
            switch(id) {
                case 0:
                    mpiComm.dispatchService(Services::count, &cells, size, Null, size);
                    break;
                case 1:
                    mpiComm.dispatchService(Services::localReshuffle, &cells, size, Null, size);
                    break;
                default:
                    throw std::invalid_argument(
                            "Main.main: Service ID unknown."

            }
        }
    }

    mpiComm.destroy();

    return 0;
}
