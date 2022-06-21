#include "TraversePST.h"
#include "cell.h"

class ServiceBuildTreeGPU : public TraverseCombinePST {
public:
    typedef int  input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceBuildTreeGPU(PST pst)
        : TraverseCombinePST(pst,PST_COPYTODEVICE,sizeof(input), sizeof(output),"CountLeft") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
