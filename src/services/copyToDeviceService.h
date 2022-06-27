#include "TraversePST.h"
#include "cell.h"

class ServiceCopyToDevice : public TraverseCombinePST {
public:
    struct input {
        int nCells;
    }; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceCopyToDevice(PST pst)
        : TraverseCombinePST(pst,PST_COPYTODEVICE,sizeof(input),sizeof(output),"CountLeft") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
