#include <stdio.h>
#include "mdl.h"
#inlcude "IO.h"
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

    int n = 1 << 20;
    int k = 4;
    int maxD = 1024;
    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, k);
    blitz::Array<float, 2> CTRM = blitz::Array<float, 2>(maxD, 2);
    p = 0;
    CTRM = -1;
    // srand(vmdl);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
    }

    pst->lcl = new LocalData {
        p,
        CTRM
    };

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
