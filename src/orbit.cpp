#include <stdio.h>
#include "mdl.h"
#include "services/pst.h"
#include "services/setadd.h"
#include "services/countLeftService.h"

int master(MDL vmdl,void *vpst) {
    auto mdl = static_cast<mdl::mdlClass *>(vmdl);
    auto pst = reinterpret_cast<PST*>(vpst);
    printf("Launched with %d threads\n",mdl->Threads());
    // Build the PST tree structure
    ServiceSetAdd::input inAdd(mdl->Threads());
    mdl->RunService(PST_SETADD,sizeof(inAdd),&inAdd);

    int N = 1 << 25;
    // user code

    //IO::generateData(N, )

    int nCells = 1;
    ServiceCountLeft::input incl[nCells];
    ServiceCountLeft::output counts[nCells];
    mdl->RunService(PST_COUNTLEFT,nCells*sizeof(incl[0]),incl,counts);
    printf("ServiceCountLeft returned: %lu\n",counts[0]);
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
