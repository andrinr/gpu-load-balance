#include "setadd.h"

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceSetAdd::input>()  || std::is_trivial<ServiceSetAdd::input>());
static_assert(std::is_void<ServiceSetAdd::output>() || std::is_trivial<ServiceSetAdd::output>());

// This is a weird service. Normally we use TraversePST, but here
// we are actually setting up the PST structure.
int ServiceSetAdd::operator()(int nIn, void *pIn, void *pOut) {
    assert(nIn == sizeof(input));
    SetAdd(node_pst,static_cast<input *>(pIn));
    return 0;
}

void ServiceSetAdd::SetAdd(PST pst,input *in) {
    PST pstNew;
    int n, idMiddle,iProcLower,iProcUpper;
    mdlassert(pst->mdl,pst->nLeaves==1);
    mdlassert(pst->mdl,in->idLower==mdlSelf(pst->mdl));
    n = in->idUpper - in->idLower;
    idMiddle = (in->idUpper + in->idLower) / 2;
    if ( n > 1 ) {
        int rID;
        /* Make sure that the pst lands on core zero */
        iProcLower = mdlThreadToProc(pst->mdl,in->idLower);
        iProcUpper = mdlThreadToProc(pst->mdl,in->idUpper-1);
        if (iProcLower!=iProcUpper) {
            idMiddle = mdlProcToThread(pst->mdl,mdlThreadToProc(pst->mdl,idMiddle));
        }
        pst->nLeaves += n - 1;
        pst->nLower = idMiddle - in->idLower;
        pst->nUpper = in->idUpper - idMiddle;

        in->idLower = idMiddle;
        pst->idUpper = in->idLower;
        rID = mdlReqService(pst->mdl,pst->idUpper,getServiceID(),in,sizeof(input));
        in->idLower = mdlSelf(pst->mdl);
        in->idUpper = idMiddle;
        pst->pstLower = new pstNode(pst->mdl);
	pst->pstLower->lcl = pst->lcl;
        SetAdd(pst->pstLower,in);
        mdlGetReply(pst->mdl,rID,NULL,NULL);
    }
}
