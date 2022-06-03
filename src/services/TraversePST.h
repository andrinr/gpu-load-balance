#ifndef TRAVERSEPST_H
#define TRAVERSEPST_H
#include "pst.h"
#include <string>
#include <type_traits>

// This class will traverse the PST calling Recurse() at each level.
// The Recurse method is responsible for making a request,
// calling and calling Traverse() recursively. When a leaf is reach
// the Service() function is called.
class TraversePST : public mdl::BasicService {
    PST node_pst;
public:
    explicit TraversePST(PST node_pst,int service_id,
                         int nInBytes, int nOutBytes, const char *service_name="")
        : BasicService(service_id, nInBytes, nOutBytes, service_name), node_pst(node_pst) {}
    explicit TraversePST(PST node_pst,int service_id,
                         int nInBytes, const char *service_name="")
        : BasicService(service_id, nInBytes, 0, service_name), node_pst(node_pst) {}
    explicit TraversePST(PST node_pst,int service_id,const char *service_name="")
        : BasicService(service_id, 0, 0, service_name), node_pst(node_pst) {}
    virtual ~TraversePST() = default;
protected:
    virtual int operator()(int nIn, void *pIn, void *pOut) final;
    virtual int Traverse(PST pst,void *vin,int nIn,void *vout,int nOut);

    virtual int OffNode(PST pst,void *vin,int nIn,void *vout,int nOut) {return Recurse(pst,vin,nIn,vout,nOut);}
    virtual int  AtNode(PST pst,void *vin,int nIn,void *vout,int nOut) {return Recurse(pst,vin,nIn,vout,nOut);}
    virtual int Recurse(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut) = 0;
protected:
    static  int Traverse(unsigned sid, PST pst,void *vin,int nIn,void *vout,int nOut);
};

// This class is for services where the input does not change as the PST is walked,
// but where a customized Combine for a fixed size output is needed.
class TraverseCombinePST : public TraversePST {
public:
    explicit TraverseCombinePST(PST node_pst,int service_id,
                                int nInBytes=0, int nOutBytes=0, const char *service_name="")
        : TraversePST(node_pst,service_id, nInBytes, nOutBytes, service_name) {}
    virtual ~TraverseCombinePST() = default;
protected:
    virtual int Recurse(PST pst,void *vin,int nIn,void *vout,int nOut) final;
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) = 0;
};

// It is common for services to return a count of the number of particles
// (total or updated) so this provides an appropriate Combine function
class TraverseCountN : public TraverseCombinePST {
public:
    typedef uint64_t output;
    explicit TraverseCountN(PST pst,int service_id,int nInBytes, const char *service_name="")
        : TraverseCombinePST(pst,service_id,nInBytes,sizeof(output),service_name) {}
    explicit TraverseCountN(PST pst,int service_id,const char *service_name="")
        : TraverseCombinePST(pst,service_id,0,sizeof(output),service_name) {}
protected:
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) final;
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut) override = 0;
};
#endif
