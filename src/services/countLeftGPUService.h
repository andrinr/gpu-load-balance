#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"
#define CUDA_CHECK(f,a) {cudaError_t rc = (f)a; if (rc!=cudaSuccess) CUDA_Abort(rc,#f,__FILE__,__LINE__);}

class ServiceCountLeftGPU : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef uint output;   // Array of counts
    explicit ServiceCountLeftGPU(PST pst)
        : TraverseCombinePST(pst,PST_COUNTLEFTGPU,MAX_CELLS*sizeof(input),MAX_CELLS*sizeof(output),"CountLeftGPU") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
