#include "initGPUService.h"
#include <blitz/array.h>

// Make sure that the communication structure is "trivial" so that it
// can be moved around with "memcpy" which is required for MDL.
static_assert(std::is_void<ServiceInit::input>()  || std::is_trivial<ServiceInit::input>());
static_assert(std::is_void<ServiceInit::output>() || std::is_trivial<ServiceInit::output>());

int ServiceInitGPU::Service(PST pst,void *vin,int nIn,void *vout, int nOut) {

    auto streams = blitz::Array<cudaStream_t, 1>(pst->nLeaves);

    for (int i = 0; i < pst->nLeaves; i++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams(i) = stream;
    }

    lcl->streams.reference(streams);

    for (int i = 0; i < 32; i++) {

    }

    return 0;
}

int ServiceInit::Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2) {
    return 0;
}
