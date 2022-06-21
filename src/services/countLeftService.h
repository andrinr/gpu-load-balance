#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"
class ServiceCountLeft : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef uint output;   // Array of counts
    explicit ServiceCountLeft(PST pst)
        : TraverseCombinePST(pst,PST_COUNTLEFT,MAX_CELLS*sizeof(input),MAX_CELLS*sizeof(output),"CountLeft") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
