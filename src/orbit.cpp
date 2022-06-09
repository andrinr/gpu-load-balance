#include <stdio.h>
#include <blitz/array.h>
#include "cell.h"
#include "mdl.h"
#include "services/countService.h"
#include "services/pst.h"
#include "services/setadd.h"
#include "services/countLeftService.h"
#include "services/initService.h"

int master(MDL vmdl,void *vpst) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    auto pst = reinterpret_cast<PST*>(vpst);
    printf("Launched with %d threads\n",mdl->Threads());
    // Build the PST tree structure
    ServiceSetAdd::input inAdd(mdl->Threads());
    mdl->RunService(PST_SETADD,sizeof(inAdd),&inAdd);

    int n = 1 << 25;
    int d = 1 << 10;

    const float lowerInit = -0.5;
    const float upperInit = 0.5;

    float lower[3] = {lowerInit, lowerInit, lowerInit};
    float upper[3] = {upperInit, upperInit, upperInit};

    // root cell is at index 1
    blitz::Array<Cell, 1> cells(d);

    // user code
    Cell root(0, d, lower, upper);
    CellHelpers::setCutAxis(root);
    CellHelpers::setCutMargin(root);
    cells(0) = root;

    ServiceInit::input iNParticles[1];
    ServiceInit::output oNParticles[1];
    mdl->RunService(PST_INIT, sizeof (int), iNParticles, oNParticles);

    for (int l = 1; l < CellHelpers::getNLevels(root); ++l) {

        int a = std::pow(2, (l - 1)) - 1;
        int b = std::min(
                CellHelpers::getNCellsOnLastLevel(root),
                (int) std::pow(2, l)) - 1;

        int nCells = b - a;

        ServiceCount::input *iCount = cells.data();
        ServiceCount::output oCount[nCells];
        mdl->RunService(PST_COUNT, nCells * sizeof(ServiceCount::input), iCount, oCount);
        printf("ServiceCount returned: %lu\n", oCount[0]);

        // Loop
        bool foundAll = false;

        int j = 0;
        while(!foundAll && j < 32) {
            j++;
            int *sumLeft;
            foundAll = true;

            ServiceCountLeft::input *iCountLeft = cells.data();
            ServiceCountLeft::output oCountLeft[nCells];
            mdl->RunService(PST_COUNTLEFT, nCells * sizeof(ServiceCountLeft::input), iCountLeft, oCountLeft);
            printf("ServiceCountLeft returned: %lu\n", oCountLeft[0]);

            std::cout << a << " " << b << std::endl;
            for (int i = a; i < b; ++i) {
                std::cout << oCountLeft[i] << " " << oCount[i] / 2.0 << std::endl;
                //std::cout << " i " << i << "\n";
                if (abs(oCountLeft[i] - oCount[i] / 2.0) < 1) {
                    cells(i).cutAxis = -1;
                } else if (oCountLeft[i] - oCount[i] / 2.0 > 0) {
                    cells(i).cutMarginRight = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                    //std::cout << " i " << cells(i).cutMarginRight << "\n";
                    foundAll = false;
                } else {
                    cells(i).cutMarginLeft = (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0;
                    foundAll = false;
                }
            }
            std::cout << foundAll << std::endl;
        }

        // Split and store all cells on current heap level
        for (int i = a; i < b; ++i) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = CellHelpers::cut(cells(i));

            CellHelpers::setCutAxis(cellRight);
            CellHelpers::setCutAxis(cellLeft);
            CellHelpers::setCutMargin(cellLeft);
            CellHelpers::setCutMargin(cellRight);

            cells(CellHelpers::getLeftChildId(cells(i))) = cellLeft;
            cells(CellHelpers::getRightChildId(cells(i))) = cellRight;
        }

    }
    return 0;
}

void *worker_init(MDL vmdl) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    // Construct a PST node for this thread. The SetAdd service will be called in "master" to contruct a tree of them.
    auto pst = new pstNode(mdl);

    pst->lcl = new LocalData();

    mdl->AddService(std::make_unique<ServiceSetAdd>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeft>(pst));
    mdl->AddService(std::make_unique<ServiceInit>(pst));
    mdl->AddService(std::make_unique<ServiceCount>(pst));

    return pst;
}

void worker_done(MDL mdl, void *ctx) {
    auto pst = reinterpret_cast<PST*>(ctx);
    delete pst;
}

int main(int argc,char **argv) {
    // This will run "worker_init" on every thread on every node. The worker init must register services and will return
    // a new "PST" object. The "master" function is then called on thread 0. When the "master" function returns, then
    // the "worker_done" routine is called on every thread.
    return mdlLaunch(argc,argv,master,worker_init,worker_done);
}
