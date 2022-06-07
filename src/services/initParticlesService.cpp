#include "initParticlesService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceCountLeft::input>()  || std::is_trivial<ServiceCountLeft::input>());
static_assert(std::is_void<ServiceCountLeft::output>() || std::is_trivial<ServiceCountLeft::output>());

int ServiceInitParticles::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    int n = 1 << 20;
    int k = 4;
    int maxD = 1024;
    blitz::Array<float, 2> p = blitz::Array<float, 2>(n, k);
    blitz::Array<float, 2> CTRM = blitz::Array<float, 2>(maxD, 2);
    p = 0;
    CTRM = -1;
    // srand(vmdl);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            p(i,d) = (float)(rand())/(float)(RAND_MAX);
        }
    }

    CTRM(0, 0) = 0;
    CTRM(0, 1) = n - 1;

    pst->lcl = new LocalData {
            p,
            CTRM
    };

}

int ServiceInitParticles::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    auto out  = static_cast<output *>(vout);
    auto out2 = static_cast<output *>(vout2);
    int nCounts = nIn / sizeof(input);
    assert(nOut1 >= nCounts*sizeof(output));
    assert(nOut2 >= nCounts*sizeof(output));
    for(auto i=0; i<nCounts; ++i)
	    out[i] += out2[i];
    return 0;
}
