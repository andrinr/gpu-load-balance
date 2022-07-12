#include <stdio.h>
#include <blitz/array.h>
#include "cell.h"
#include "mdl.h"
#include "services/count.h"
#include "services/pst.h"
#include "services/setadd.h"
#include "services/countLeft.h"
#include "services/countLeft.cuh"
#include "services/countLeftAxis.cuh"
#include "services/copyToDevice.cuh"
#include "services/finalize.cuh"
#include "services/partition.cuh"
#include "services/partition.h"
#include "services/init.cuh"
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

    const GPU_ACCELERATION acceleration = COUNT_PARTITION;

    // user code
    Cell root(0, d, lower, upper);
    root.cutAxis = 0;
    root.setCutMargin();

    root.log();

    // root cell is at index 1
    blitz::Array<Cell, 1> cellHeap(root.getTotalNumberOfCells());

    cellHeap(0) = root;

    ServiceInit::input iInit {N/mdl->Threads(), acceleration};
    ServiceInit::output oInit[1];
    mdl->RunService(PST_INIT, sizeof (ServiceInit::input), &iInit, oInit);

    // Only copy once
    if (acceleration == COUNT_PARTITION) {
        ServiceCopyToDevice::input iCopy {acceleration};
        ServiceCopyToDevice::output oCopy[1];
        mdl->RunService(PST_COPYTODEVICE, sizeof (ServiceCopyToDevice::input), &iCopy, oCopy);
    }

    for (int l = 1; l < root.getNLevels(); ++l) {

        printf("done part root 2 \n");
        int a = std::pow(2, (l - 1)) - 1;
        int b = std::min(
               root.getNCellsOnLastLevel(),
                (int) std::pow(2, l)) - 2;

        int nCells = b - a + 1;
        blitz::Array<Cell, 1> cells = cellHeap(blitz::Range(a, b));
        ServiceCount::input *iCells = cells.data();

        if (acceleration == NONE || acceleration == COUNT) {
            ServiceMakeAxis::output oSwaps[1];
            mdl->RunService(PST_MAKEAXIS, nCells * sizeof(ServicePartition::input), iCells, oSwaps);
        }

        printf("done part root 2.5 \n");
        ServiceCount::output oCount[nCells];
        mdl->RunService(PST_COUNT, nCells * sizeof(ServiceCount::input), iCells, oCount);

        printf("done part root 3 \n");
        // Copy with each iteration as partition is done on GPU
        if (acceleration == COUNT) {
            ServiceCopyToDevice::input iCopy {acceleration};
            ServiceCopyToDevice::output oCopy[1];
            mdl->RunService(PST_COPYTODEVICE, sizeof (ServiceCopyToDevice::input), &iCopy, oCopy);
        }

        // Loop
        bool foundAll = false;

        int j = 0;
        while(!foundAll && j < 32) {
            j++;
            int *sumLeft;
            foundAll = true;

            ServiceCountLeft::output oCountLeft[nCells];
            if (acceleration == COUNT_PARTITION) {
                mdl->RunService(
                        PST_COUNTLEFTGPU,
                        nCells * sizeof(ServiceCountLeft::input),
                        iCells,
                        oCountLeft);
            }
            else if (acceleration == COUNT) {
                mdl->RunService(
                        PST_COUNTLEFTAXISGPU,
                        nCells * sizeof(ServiceCountLeft::input),
                        iCells,
                        oCountLeft);
            }
            else {
                mdl->RunService(
                        PST_COUNTLEFT,
                        nCells * sizeof(ServiceCountLeft::input),
                        iCells,
                        oCountLeft);
            }

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
                    //cells(i).cutMarginRight -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    cells(i).cutMarginRight = cells(i).getCut();
                    foundAll = false;
                } else {
                    //cells(i).cutMarginLeft -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    cells(i).cutMarginLeft = cells(i).getCut();
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

        if (acceleration == COUNT_PARTITION) {
            ServicePartitionGPU::output oPartition[1];
            mdl->RunService(PST_PARTITIONGPU, nCells * sizeof(ServicePartitionGPU::input), iCells, oPartition);
            printf("done part root 1 \n");
        }
        else {
            ServicePartition::output oPartition[1];
            mdl->RunService(PST_PARTITION, nCells * sizeof(ServicePartition::input), iCells, oPartition);
        }
    }

    if (acceleration == COUNT || acceleration == COUNT_PARTITION) {
        ServiceFinalize::input iFree[1];
        ServiceFinalize::output oFree[1];
        mdl->RunService(PST_FINALIZE, sizeof (int), iFree, oFree);
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
    mdl->AddService(std::make_unique<ServiceCopyToDevice>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeftGPU>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeftAxisGPU>(pst));
    mdl->AddService(std::make_unique<ServicePartition>(pst));
    mdl->AddService(std::make_unique<ServicePartitionGPU>(pst));
    mdl->AddService(std::make_unique<ServiceFinalize>(pst));
    mdl->AddService(std::make_unique<ServiceMakeAxis>(pst));

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
