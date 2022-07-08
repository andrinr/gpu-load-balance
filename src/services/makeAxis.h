#include "TraversePST.h"
#include "cell.h"

class ServiceMakeAxis : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceMakeAxis(PST pst)
        : TraverseCombinePST(pst,PST_MAKEAXIS,sizeof(input),sizeof(output),"Reshuffle") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
