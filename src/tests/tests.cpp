//
// Created by andrin on 8/1/22.
//


#include "../src/services/partitionGPU.h"
#include "../src/services/init.h"
#include "../src/services/copyParticles.h"
#include "catch2.h"
#include <blitz/array.h>

TEST_CASE("GPU Partition", "Regular Function")
{


    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    auto pst = new pstNode(mdl);
    pst->lcl = new LocalData();
    mdl->AddService(std::make_unique<ServiceSetAdd>(pst));
    mdl->AddService(std::make_unique<ServiceInit>(pst));
    mdl->AddService(std::make_unique<ServicePartitionGPU>(pst));
    mdl->AddService(std::make_unique<ServiceCopyParticles>(pst));

    META_PARAMS params{true, true, false};

    const N = 1 << 15;
    ServiceInit::input iInit {n/mdl->Threads(), params};
    ServiceInit::output oInit[1];
    mdl->RunService(PST_INIT, sizeof (ServiceInit::input), &iInit, oInit);

    ServiceCopyParticles::input iCopy {params};
    ServiceCopyParticles::output oCopy[1];
    mdl->RunService(PST_COPYPARTICLES, sizeof (ServiceCopyParticles::input), &iCopy, oCopy);

    Cell root(0, d, lower, upper);
    root.cutAxis = 0;
    root.setCutMargin();

    root.log();

    // root cell is at index 1
    blitz::Array<Cell, 1> cellHeap(root.getTotalNumberOfCells());

    cellHeap(0) = root;

    ServicePartition::output oPartition[1];
    mdl->RunService(
            PST_PARTITION,
            nCells * sizeof(ServicePartition::input),
            iCells,
            oPartition);

    int nLeft = 0;
    for (int i = 0; i < N; i++) {
        nLeft +=
    }
    REQUIRE_FALSE(function1(0));
    REQUIRE_FALSE(function1(5));
    REQUIRE(function1(10));
}