#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"
class ServiceCountLeftGPUAxis : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef unsigned int output;   // Array of counts
    explicit ServiceCountLeftGPUAxis(PST pst)
        : TraverseCombinePST(pst,PST_COUNTLEFTAXISGPU,MAX_CELLS*sizeof(input),MAX_CELLS*sizeof(output),"CountLeftAxis") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
