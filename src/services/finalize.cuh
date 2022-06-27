#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"
class ServiceFreeDevice : public TraverseCombinePST {
public:
    typedef int  input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceFreeDevice(PST pst)
        : TraverseCombinePST(pst,PST_FREE,sizeof(input),sizeof(output),"CountLeft") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
