#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"

class ServiceFinalize : public TraverseCombinePST {
public:
    struct input {
        META_PARAMS params;
    }; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceFinalize(PST pst)
        : TraverseCombinePST(pst,PST_FINALIZE,sizeof(input),sizeof(output),"Finalize") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
