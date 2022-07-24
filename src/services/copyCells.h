#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"

class ServiceCopyCells : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceCopyCells(PST pst)
        : TraverseCombinePST(pst,PST_COPYCELLS,sizeof(input),sizeof(output),"CopyCells") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
