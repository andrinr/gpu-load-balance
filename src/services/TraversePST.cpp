#include "TraversePST.h"

int TraversePST::operator()(int nIn, void *pIn, void *pOut) {
    return Traverse(node_pst,pIn,nIn,pOut,getMaxBytesOut());
}

int TraversePST::Traverse(PST pst,void *vin,int nIn,void *vout,int nOut) {
    if (pst->AmCore())       return Service(pst,vin,nIn,vout,nOut);
    else if (pst->OffNode()) return OffNode(pst,vin,nIn,vout,nOut);
    else if (pst->AmNode())  return  AtNode(pst,vin,nIn,vout,nOut);
    else                     return Recurse(pst,vin,nIn,vout,nOut);
}

// This is a static version of traverse that we use when we want to traverse a subtree,
// but for a difference service. We look it up in the MDL and call the real traverse.
// This relies on the service being a "TraversePST" and if PST was converted to C++ then
// there would be a more elegant way of doing this.
int TraversePST::Traverse(unsigned sid, PST pst,void *vin,int nIn,void *vout,int nOut) {
    auto mdl = static_cast<mdl::mdlClass *>(pst->mdl);
    auto service = dynamic_cast<TraversePST *>(mdl->GetService(sid));
    assert(service); // It would be a serious error if this was not a TraversePST service.
    return service->Traverse(pst,vin,nIn,vout,nOut);
}

// This is the default function for when we are still traversing the PST
// (OffNode,AtNode,Recurse) and will just naively recurse. Once we reach
// the leaf (AmCore) then Service() is called and must be provided.
int TraversePST::Recurse(PST pst,void *vin,int nIn,void *vout,int nOut) {
    auto mdl = reinterpret_cast<mdl::mdlClass *>(pst->mdl);
    auto rID = mdl->ReqService(pst->idUpper,getServiceID(),vin,nIn);
    Traverse(pst->pstLower,vin,nIn,vout,nOut);
    return mdl->GetReply(rID,vout);
}

// This class is used for services that pass the same input (or none),
// and need to combine output results of the same size and type.
int TraverseCombinePST::Recurse(PST pst,void *vin,int nIn,void *vout,int nOut) {
    auto mdl = reinterpret_cast<mdl::mdlClass *>(pst->mdl);
    auto rID = mdl->ReqService(pst->idUpper,getServiceID(),vin,nIn);
    Traverse(pst->pstLower,vin,nIn,vout,nOut);
    void *vout2 = alloca(nOut);
    auto nOut2 = mdl->GetReply(rID,vout2);
    return Combine(vout,vout2,nIn,nOut,nOut2);
}

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
//static_assert(std::is_void<TraverseCountN::input>()  || std::is_trivial<TraverseCountN::input>());
static_assert(std::is_void<TraverseCountN::output>() || std::is_trivial<TraverseCountN::output>());
int TraverseCountN::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    *out += *out2;
    return sizeof(output);
}
