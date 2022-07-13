#ifndef PST_H
#define PST_H
#include "mdl.h"
#include <blitz/array.h>

class LocalData {
public:
    blitz::Array<float, 2> particles;
    blitz::Array<float, 1> particlesT;
    blitz::Array<unsigned int, 2> cellToRangeMap;
    blitz::Array<unsigned int, 1> h_countsLeft;
    blitz::Array<cudaStream_t , 1> streams;
    float * d_particlesT;
    float * d_particlesX;
    float * d_particlesY;
    float * d_particlesZ;
    unsigned int * d_permutations;
    unsigned int * d_results;
    unsigned int * h_results;
    unsigned int * d_offsetLeq;
    unsigned int * d_offsetG;
    int nThreads;
    int nBlocks;

    LocalData() = default;
};

//#include "bound.h"
class pstNode {
public:
    class pstNode *pstLower;
    LocalData *lcl;
    MDL mdl;
    int idSelf;
    int idUpper;
    int nLeaves;
    int nLower;
    int nUpper;
    //Bound bnd;
    pstNode(MDL mdl) : pstLower(nullptr), mdl(mdl),
        idSelf(mdlSelf(mdl)), nLeaves(1), nLower(0), nUpper(0) {}
    bool OffNode() const { return nLeaves > mdlCores(mdl); }
    bool OnNode()  const { return nLeaves <= mdlCores(mdl); }
    bool AmNode()  const { return nLeaves == mdlCores(mdl); }
    bool NotNode() const { return nLeaves != mdlCores(mdl); }
    bool AmCore()  const { return nLeaves == 1; }
    bool NotCore() const { return nLeaves > 1; }
};
typedef pstNode *PST;

enum pst_service {
    PST_SRV_STOP=0, /* service 0 is always STOP and handled by MDL */
    PST_SETADD,
    PST_INIT,
    PST_INITGPU,
    PST_COPYTODEVICE,
    PST_COUNTLEFTGPU,
    PST_COUNTLEFTAXISGPU,
    PST_COUNTLEFT,
    PST_PARTITION,
    PST_PARTITIONGPU,
    PST_COUNT,
    PST_FINALIZE,
    PST_MAKEAXIS
};
#endif
