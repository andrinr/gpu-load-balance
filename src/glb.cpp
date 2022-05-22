#include <iostream>
#include <fstream>
#include <filesystem>
#include <blitz/array.h>
#include <chrono>
#include <cstdlib>
#include <memory>

#include "constants.h"
#include "cell.h"
#include "IO.h"
#include "orb.h"
#include "services/serviceManager.h"
#include "services/baseService.h"
#include "services/countService.h"
#include "services/countLeftService.h"
#include "services/localReshuffleService.h"
#include "services/buildTreeService.h"
#include "comm/MPIMessaging.h"

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
    auto mpiMessaging = std::make_shared<MPIMessaging>;

    std::cout << "Process " << mpiMessaging->rank << " processing " << count / 1000 << "K particles." << std::endl;
    std::cout << "Process " << mpiMessaging->rank << " starting task..." << std::endl;

    int N = (count+mpiMessaging.np-1) / mpiMessaging->np ;
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
    auto orb std::make_shared<Orb>(particles, cellToParticle, nLeafCells);

    // Init all services
    auto countService = std::make_unique<CountService>;
    auto countLeftService = std::make_unique<CountLeftService>;
    auto localReshuffleService = std::make_unique<LocalReshuffleService>;
    auto buildTreeService = std::make_unique<BuildTreeService>;

    ServiceManager manager(&orb, &mpiMessaging);
    manager.addService(countService);
    manager.addService(countService);
    manager.addService(localReshuffleService);
    manager.addService(buildTreeService);

    std::cout << "Process " << mpiMessaging.rank << " start building tree" << std::endl;

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

        std::cout << "building input" << std::endl;
        BuildTreeServiceInput btsi {
            cells.data(),
            &mpiMessaging
        };
        mpiMessaging.dispatchService(
                &manager,
                BUILD_TREE_SERVICE_ID,
                &btsi,
                nullptr);
    }
    else {
        bool status = true;
        while(status) {
            mpiMessaging.workService(
                    &manager
                    );
        }
    }

    mpiMessaging.finalize();

    return 0;
}
