#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"

class ServiceInit : public TraverseCombinePST {
public:
    struct input {
        int nParticles;
        int d;
        bool generate;
        META_PARAMS params;
    };
    typedef int output;
    explicit ServiceInit(PST pst)
            : TraverseCombinePST(pst,PST_INIT,sizeof (input),sizeof (int),"Init") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
