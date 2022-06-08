#include "TraversePST.h"
#include "cell.h"

class ServiceInit : public TraverseCombinePST {
public:
    static constexpr int max_cells = 8192;
    typedef int input; // Number of particles
    typedef int output; // Number of particles
    explicit ServiceInit(PST pst)
            : TraverseCombinePST(pst,PST_INIT,sizeof (int),sizeof (int),"Init") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
