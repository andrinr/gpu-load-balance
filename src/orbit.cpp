#include <stdio.h>
#include <blitz/array.h>
#include "cell.h"
#include "mdl.h"
#include "services/count.h"
#include "services/pst.h"
#include "services/setadd.h"
#include "services/countLeft.h"
#include "services/countLefGPU.h"
#include "services/countLeftGPUAxis.h"
#include "services/copyParticles.h"
#include "services/copyCells.h"
#include "services/finalize.h"
#include "services/partitionGPU.h"
#include "services/partition.h"
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

    //mdl->argc
    //mdl->argv

    float lower[3] = {-0.5, -0.5, -0.5};
    float upper[3] = {0.5, 0.5, 0.5};

    META_PARAMS params{
            true,
            true,
            false,
    };

    // user code
    Cell root(0, d, lower, upper);
    root.cutAxis = 0;
    root.setCutMargin();

    root.log();

    // root cell is at index 1
    blitz::Array<Cell, 1> cellHeap(root.getTotalNumberOfCells());

    cellHeap(0) = root;

    ServiceInit::input iInit {N/mdl->Threads(), params};
    ServiceInit::output oInit[1];
    mdl->RunService(PST_INIT, sizeof (ServiceInit::input), &iInit, oInit);

    // Only copy once
    if (params.GPU_PARTITION) {
        ServiceCopyParticles::input iCopy {params};
        ServiceCopyParticles::output oCopy[1];
        mdl->RunService(PST_COPYPARTICLES, sizeof (ServiceCopyParticles::input), &iCopy, oCopy);
    }

    unsigned int oCounts[MAX_CELLS];
    unsigned int oCountsLeft[MAX_CELLS];


    for (int l = 1; l < root.getNLevels(); ++l) {

        int a = std::pow(2, (l - 1)) - 1;
        int b = std::min(
               root.getNCellsOnLastLevel(),
                (int) std::pow(2, l)) - 2;

        int nCells = b - a + 1;
        blitz::Array<Cell, 1> cells = cellHeap(blitz::Range(a, b));
        ServiceCount::input *iCells = cells.data();

        if (not params.GPU_PARTITION) {
            ServiceMakeAxis::output oSwaps[1];
            mdl->RunService(PST_MAKEAXIS, nCells * sizeof(ServicePartition::input), iCells, oSwaps);
        }

        mdl->RunService(PST_COUNT, nCells * sizeof(ServiceCount::input), iCells, oCounts);

        // Copy with each iteration as partition is done on CPU
        if (params.GPU_COUNT && not params.GPU_PARTITION) {
            ServiceCopyParticles::input iCopy {params};
            ServiceCopyParticles::output oCopy[1];
            mdl->RunService(PST_COPYPARTICLES, sizeof (ServiceCopyParticles::input), &iCopy, oCopy);
        }

        if (params.GPU_COUNT) {
            ServiceCopyCells::output oCopy[1];
            mdl->RunService(PST_COPYCELLS, nCells * sizeof (ServiceCopyCells::input), iCells, oCopy);
        }

        // Loop
        bool foundAll = false;

        int j = 0;
        while(!foundAll && j < 32) {
            j++;
            int *sumLeft;
            foundAll = true;

            if (params.GPU_PARTITION) {
                mdl->RunService(
                PST_COUNTLEFTGPU,
                nCells * sizeof(ServiceCountLeftGPU::input),
                iCells,
                oCountsLeft);
            }
            else if (params.GPU_COUNT) {
                mdl->RunService(
                PST_COUNTLEFTAXISGPU,
                nCells * sizeof(ServiceCountLeftGPUAxis::input),
                iCells,
                oCountsLeft);
            }
            else {
                mdl->RunService(
                PST_COUNTLEFT,
                nCells * sizeof(ServiceCountLeft::input),
                iCells,
                oCountsLeft);
            }

            for (int i = 0; i < nCells; ++i) {
                if (cells(i).foundCut) continue;
                printf(
                        "counted left: %u, of %u. cut %f, axis %d, level %u, cell %u \n",
                        oCountsLeft[i],
                        oCounts[i] / 2,
                        (cells(i).cutMarginLeft + cells(i).cutMarginRight) / 2.0,
                        cells(i).cutAxis,
                        l,
                        cells(i).id
                        );
                //CellHelpers::log(cells(i));

                float ratio = ceil(cells(i).nLeafCells / 2.0) / cells(i).nLeafCells;
                int difference = oCountsLeft[i] - oCounts[i] * ratio;
                float diffPct = (float) difference / oCountsLeft[i];
                //printf("diff %f %, diff %i \n", diffPct, difference);
                if (abs(difference) < 32) {
                    cells(i).foundCut = true;
                } else if (difference > 0) {
                    // good optimization, but can it be proven to give a result?
                    // Take more constrained version of both, which should ensure it?
                    // Could maybe also be proven.
                    if (params.FAST_MEDIAN)
                    {
                        cells(i).cutMarginRight -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    }
                    else {
                        cells(i).cutMarginRight = cells(i).getCut();
                    }
                    foundAll = false;
                } else {
                    if (params.FAST_MEDIAN) {
                        cells(i).cutMarginLeft -= diffPct * (cells(i).cutMarginRight - cells(i).cutMarginLeft);
                    }
                    else {
                        cells(i).cutMarginLeft = cells(i).getCut();
                    }
                    foundAll = false;
                }
            }
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

        if (params.GPU_PARTITION) {
            ServicePartitionGPU::output oPartition[1];
            mdl->RunService(
                    PST_PARTITIONGPU,
                    nCells * sizeof(ServicePartitionGPU::input),
                    iCells,
                    oPartition);
        }
        else {
            ServicePartition::output oPartition[1];
            mdl->RunService(
                    PST_PARTITION,
                    nCells * sizeof(ServicePartition::input),
                    iCells,
                    oPartition);
        }
    }

    if (params.GPU_COUNT){
        ServiceFinalize::input iFree {params};
        ServiceFinalize::output oFree[1];
        mdl->RunService(PST_FINALIZE, sizeof (ServiceFinalize::input), &iFree, oFree);
    }

    return 0;
}

void *worker_init(MDL vmdl) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    // Construct a PST node for this thread. The SetAdd service will be called in "master" to contruct a tree of them.
    auto pst = new pstNode(mdl);

    pst->lcl = new LocalData();

    mdl->AddService(std::make_unique<ServiceSetAdd>(pst));
    mdl->AddService(std::make_unique<ServiceInit>(pst));
    mdl->AddService(std::make_unique<ServiceCount>(pst));
    mdl->AddService(std::make_unique<ServiceCopyParticles>(pst));
    mdl->AddService(std::make_unique<ServiceCopyCells>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeft>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeftGPU>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeftGPUAxis>(pst));
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
