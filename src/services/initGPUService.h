#include "TraversePST.h"
#include "cell.h"

class ServiceInitGPU : public TraverseCombinePST {
public:
    struct input {
        int nThreads;
    }; // Number of particles
    typedef int output; // Number of particles
    explicit ServiceInitGPU(PST pst)
            : TraverseCombinePST(pst,PST_INITGPU,sizeof (int),sizeof (int),"Init") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
