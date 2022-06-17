#include "TraversePST.h"
#include "cell.h"

class ServiceCountLeftGPU : public TraverseCombinePST {
public:
    static constexpr int max_cells = 8192;
    typedef struct Cell input; // Array of Cells
    typedef uint64_t output;   // Array of counts
    explicit ServiceCountLeftGPU(PST pst)
        : TraverseCombinePST(pst,PST_COUNTLEFTGPU,max_cells*sizeof(input),max_cells*sizeof(output),"CountLeftGPU") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};