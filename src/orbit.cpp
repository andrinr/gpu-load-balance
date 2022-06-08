#include <stdio.h>
#include <blitz/array.h>
#include "cell.h"
#include "mdl.h"
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

    /*for (int l = 1; l < CellHelpers::getNLevels(*inputData.root); ++l) {

        int a = std::pow(2, (l-1)) - 1;
        int b = std::min(
                CellHelpers::getNCellsOnLastLevel(*inputData.root),
                inputData.root->nLeafCells) - 1;

        int nCells = 1;*/


    ServiceInit::input iNParticles[1];
    ServiceInit::output oNParticles[1];
    mdl->RunService(PST_INIT, sizeof (int), iNParticles, oNParticles);

    int nCells = 1;
    ServiceCountLeft::input * iCountLeft = cells.data();
    ServiceCountLeft::output oCountLeft[nCells];
    mdl->RunService(PST_COUNTLEFT,nCells*sizeof(ServiceCountLeft::input),iCountLeft,oCountLeft);
    printf("ServiceCountLeft returned: %lu\n",oCountLeft[0]);
    return 0;
}

void *worker_init(MDL vmdl) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    // Construct a PST node for this thread. The SetAdd service will be called in "master" to contruct a tree of them.
    auto pst = new pstNode(mdl);

    // Put that in a service

    // Services:
    // Initialize particles
    // Copy to device
    // Count Left
    // Reshuffle

    mdl->AddService(std::make_unique<ServiceSetAdd>(pst));
    mdl->AddService(std::make_unique<ServiceCountLeft>(pst));
    mdl->AddService(std::make_unique<ServiceInit>(pst));

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
