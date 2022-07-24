#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"

class ServiceCopyParticles : public TraverseCombinePST {
public:
    struct input {
        META_PARAMS params;
    }; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceCopyParticles(PST pst)
        : TraverseCombinePST(pst,PST_COPYPARTICLES,sizeof(input),sizeof(output),"CopyToDevice") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
