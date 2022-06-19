#include "TraversePST.h"
#include "cell.h"

class ServiceCopyToDevice : public TraverseCombinePST {
public:
    static constexpr int max_cells = 8192;
    typedef int  input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceCopyToDevice(PST pst)
        : TraverseCombinePST(pst,PST_COPYTODEVICE,max_cells*sizeof(input),max_cells*sizeof(output),"CountLeft") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
