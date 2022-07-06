#include <stdio.h>
#include <blitz/array.h>
#include "cell.h"
#include "mdl.h"
#include "services/init.cuh"
#include "services/count.h"
#include "services/pst.h"
#include "services/setadd.h"
#include "services/countLeft.h"
#include "services/countLeft.cuh"
#include "services/copyToDevice.cuh"
#include "services/finalize.cuh"
#include "services/reshuffle.cuh"
#include "services/init.h"
#include "services/makeAxis.h"

#include "constants.h"

int master(MDL vmdl,void *vpst) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    auto pst = reinterpret_cast<PST*>(vpst);
    printf("Launched with %d threads\n",mdl->Threads());
    // Build the PST tree structure
    ServiceSetAdd::input inAdd(mdl->Threads());
    mdl->RunService(PST_SETADD,sizeof(inAdd),&inAdd);

    float lower[3] = {-0.5, -0.5, -0.5};
    float upper[3] = {0.5, 0.5, 0.5};

    // user code
    Cell root(0, d, lower, upper);
    root.cutAxis = 0;
    root.setCutMargin();

    root.log();

    // root cell is at index 1
    blitz::Array<Cell, 1> cellHeap(root.getTotalNumberOfCells());

    cellHeap(0) = root;

    ServiceInit::input iInit {N/mdl->Threads()};
    ServiceInit::output oInit[1];
    mdl->RunService(PST_INIT, sizeof (ServiceInit::input), &iInit, oInit);

    ServiceInitGPU::input iInitGpu{mdl->Threads(), N/mdl->Threads()};
    ServiceInitGPU::output oInitGpu[1];
    mdl->RunService(PST_INITGPU, sizeof (ServiceInitGPU::input), &iInitGpu, oInitGpu);

    for (int l = 1; l < root.getNLevels(); ++l) {

        int a = std::pow(2, (l - 1)) - 1;
        int b = std::min(
               root.getNCellsOnLastLevel(),
                (int) std::pow(2, l)) - 2;

        int nCells = b - a + 1;
        blitz::Array<Cell, 1> cells = cellHeap(blitz::Range(a, b));
        ServiceCount::input *iCells = cells.data();

        ServiceReshuffle::output oSwaps[1];
        mdl->RunService(PST_AXISSWAP, nCells * sizeof(ServiceReshuffle::input), iCells, oSwaps);

        ServiceCount::output oCount[nCells];
        mdl->RunService(PST_COUNT, nCells * sizeof(ServiceCount::input), iCells, oCount);

        ServiceCopyToDevice::input iCopy[1];
        ServiceCopyToDevice::output oCopy[1];
        mdl->RunService(PST_COPYTODEVICE, sizeof (int), iCopy, oCopy);

        // Loop
        bool foundAll = false;

        int j = 0;
        while(!foundAll && j < 32) {
            j++;
            int *sumLeft;
            foundAll = true;

            ServiceCountLeft::output oCountLeft[nCells];
            //mdl->RunService(PST_COUNTLEFT, nCells * sizeof(ServiceCountLeft::input), iCells, oCountLeft);
            mdl->RunService(PST_COUNTLEFTGPU, nCells * sizeof(ServiceCountLeft::input), iCells, oCountLeft);

            for (int i = 0; i < nCells; ++i) {
                if (cells(i).foundCut) continue;
                printf(
                        "counted left: %u, of %u. cut %f, axis %d, level %u, cell %u \n",
                        oCountLeft[i],
                        oCount[i] / 2,
                        (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0,
                        cells(i).cutAxis,
                        l,
                        cells(i).id
                        );
                //CellHelpers::log(cells(i));

                float ratio = ceil(cells(i).nLeafCells / 2.0) / cells(i).nLeafCells;
                int difference = oCountLeft[i] - oCount[i] * ratio;
                float diffPct = (float) difference / oCountLeft[i];
                printf("diff %f %, diff %i \n", diffPct, difference);
                if (abs(difference) < 32) {
                    cells(i).foundCut = true;
                } else if (difference > 0) {
                    // good optimization, but can it be proven to give a result?
                    // Take more constrained version of both, which should ensure it?
                    // Could maybe also be proven.
                    cells(i).cutMarginRight -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    //cells(i).cutMarginRight = cells(i).getCut();
                    foundAll = false;
                } else {
                    cells(i).cutMarginLeft -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    //cells(i).cutMarginLeft = cells(i).getCut();
                    foundAll = false;
                }
            }
            printf("%i \n", j);
            //free(oCountLeft);
        }

        // Split and store all cells on current heap level
        for (int i = 0; i < nCells; ++i) {
            Cell cellLeft;
            Cell cellRight;
            std::tie(cellLeft, cellRight) = cells(i).cut();

            cellRight.setCutAxis();
            cellRight.setCutMargin();
            cellLeft.setCutAxis();
            cellLeft.setCutMargin();

            cellHeap(cells(i).getLeftChildId()) = cellLeft;
            cellHeap(cells(i).getRightChildId()) = cellRight;

            //cellLeft.log();
            //cellRight.log();
        }

        ServiceReshuffle::output oCutIndices[1];
        mdl->RunService(PST_RESHUFFLE, nCells * sizeof(ServiceReshuffle::input), iCells, oCutIndices);
    }

    ServiceFreeDevice::input iFree[1];
    ServiceFreeDevice::output oFree[1];
    mdl->RunService(PST_FREE, sizeof (int), iFree, oFree);

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
    mdl->AddService(std::make_unique<ServiceInitGPU>(pst));
    mdl->AddService(std::make_unique<ServiceCount>(pst));
    mdl->AddService(std::make_unique<ServiceCopyToDevice>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeftGPU>(pst));
    mdl->AddService(std::make_unique<ServiceReshuffle>(pst));
    mdl->AddService(std::make_unique<ServiceFreeDevice>(pst));
    mdl->AddService(std::make_unique<ServiceAxisSwap>(pst));

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
